# 예측 모델 정의 (ridge, rf, xgb, lgbm, gru, cnn)
#   사용: spec = MODEL_SPECS["lgbm"]
#         model = spec["build"](params, n_jobs)
#         model.fit(X_tab, X_seq, y)       y = 실제 15분 평균 - 기준 계산식
#         model.predict(X_tab, X_seq)      보정값 예측
#   - 최종 예측 = 기준 계산식 + 보정값
#   - model.pkl에 클래스 이름이 저장되므로 TabModel, SeqModel, GRUNet, CNNNet 이름 변경 금지
#   - 멀티프로세싱에서 pickle 되도록 함수는 전부 최상위 def (lambda 금지)

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config
from common.features import EXTRA_COLS, TAB_COLS

BATCH = 512
SEED = 42

# 근거 논문 (학습 결과 엑셀에 기록)
PAPERS = {
    "ridge": "Hoerl & Kennard (1970), Technometrics",
    "rf": "Breiman (2001), Machine Learning",
    "xgb": "Chen & Guestrin (2016), KDD",
    "lgbm": "Ke et al. (2017), NeurIPS",
    "gru": "Cho et al. (2014), EMNLP",
    "cnn": "Bai, Kolter & Koltun (2018), arXiv:1803.01271",
    "optuna": "Akiba et al. (2019), KDD",
    "가중 평균 앙상블": "Bates & Granger (1969), Operational Research Quarterly",
}


# 트리·선형 모델 (요약 특징 입력)
class TabModel:

    def __init__(self, name, params, model):
        self.name = name
        self.params = params
        self.model = model

    def fit(self, X_tab, X_seq, y):
        self.model.fit(X_tab, y)

    def predict(self, X_tab, X_seq):
        return self.model.predict(X_tab)


# GRU: 최근 10분 흐름을 순서대로 학습
class GRUNet(nn.Module):

    def __init__(self, hidden, n_extra):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_extra, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, extra):
        out, _ = self.gru(seq.unsqueeze(-1))   # (배치, 60, 1) -> (배치, 60, hidden)
        last = out[:, -1]                      # 마지막 시점
        return self.head(torch.cat([last, extra], dim=1)).squeeze(1)


# 1D-CNN: 짧은 급변(설비 on/off) 포착
class CNNNet(nn.Module):

    def __init__(self, channels, n_extra):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(channels + n_extra, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, extra):
        feat = self.conv(seq.unsqueeze(1)).squeeze(-1)
        return self.head(torch.cat([feat, extra], dim=1)).squeeze(1)


# 딥러닝 모델 (시퀀스 + 보조 특징 입력)
#   fit에 valid를 주면 에포크마다 loss를 history에 기록 (학습 자체는 같음)
class SeqModel:

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.net = None
        self.history = []       # [{"epoch", "train_loss", "valid_loss"}, ...]
        self.best_epoch = None  # valid loss가 가장 낮았던 에포크 (기록만, 가중치는 마지막 에포크)

    # 보조 특징 추출 (kW 값은 NORM으로 정규화)
    def make_extra(self, X_tab):
        extra = np.zeros((len(X_tab), len(EXTRA_COLS)), dtype="float32")
        for j in range(len(EXTRA_COLS)):
            col = EXTRA_COLS[j]
            extra[:, j] = X_tab[:, TAB_COLS.index(col)]
            if col == "proj_base" or col == "kw_now":
                extra[:, j] = extra[:, j] / config.NORM
        return extra

    # valid 전체 loss (학습에 영향 없음)
    def valid_loss(self, seq, extra, target, loss_fn):
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(target), BATCH):
                end = start + BATCH
                loss = loss_fn(self.net(seq[start:end], extra[start:end]), target[start:end])
                total += loss.item() * len(target[start:end])
        return total / len(target)

    def fit(self, X_tab, X_seq, y, valid=None):
        torch.manual_seed(SEED)
        if self.name == "gru":
            self.net = GRUNet(self.params["hidden"], len(EXTRA_COLS))
        else:
            self.net = CNNNet(self.params["channels"], len(EXTRA_COLS))

        seq = torch.tensor(X_seq, dtype=torch.float32)
        extra = torch.tensor(self.make_extra(X_tab))
        target = torch.tensor(y / config.NORM, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        loss_fn = nn.HuberLoss()

        if valid is not None:
            X_tab_va, X_seq_va, y_va = valid
            seq_va = torch.tensor(X_seq_va, dtype=torch.float32)
            extra_va = torch.tensor(self.make_extra(X_tab_va))
            target_va = torch.tensor(y_va / config.NORM, dtype=torch.float32)

        self.history = []
        for epoch in range(self.params["epochs"]):
            self.net.train()
            order = torch.randperm(len(target))
            total = 0.0
            for start in range(0, len(target), BATCH):
                idx = order[start:start + BATCH]
                optimizer.zero_grad()
                loss = loss_fn(self.net(seq[idx], extra[idx]), target[idx])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)

            row = {"epoch": epoch + 1, "train_loss": total / len(target), "valid_loss": None}
            if valid is not None:
                row["valid_loss"] = self.valid_loss(seq_va, extra_va, target_va, loss_fn)
            self.history.append(row)

        if valid is not None:
            losses = [row["valid_loss"] for row in self.history]
            self.best_epoch = int(np.argmin(losses)) + 1

    def predict(self, X_tab, X_seq):
        self.net.eval()
        seq = torch.tensor(X_seq, dtype=torch.float32)
        extra = torch.tensor(self.make_extra(X_tab))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(seq), BATCH):
                pred = self.net(seq[start:start + BATCH], extra[start:start + BATCH])
                outputs.append(pred.numpy())
        return np.concatenate(outputs) * config.NORM


# 모델별 Optuna 탐색 범위(suggest_xxx), 모델 생성(build_xxx)
#   n_jobs: 모델 내부 병렬 스레드 수 (ridge, 딥러닝은 사용 안 함)

def suggest_ridge(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


def build_ridge(params, n_jobs=-1):
    # 특징 단위가 달라 표준화 후 적용
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=params["alpha"]))])
    return TabModel("ridge", params, model)


def suggest_rf(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
    }


def build_rf(params, n_jobs=-1):
    model = RandomForestRegressor(**params, n_jobs=n_jobs, random_state=SEED)
    return TabModel("rf", params, model)


def suggest_xgb(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def build_xgb(params, n_jobs=-1):
    model = XGBRegressor(**params, tree_method="hist", n_jobs=n_jobs, random_state=SEED)
    return TabModel("xgb", params, model)


def suggest_lgbm(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def build_lgbm(params, n_jobs=-1):
    model = LGBMRegressor(**params, subsample_freq=1, n_jobs=n_jobs, verbose=-1, random_state=SEED)
    return TabModel("lgbm", params, model)


def suggest_gru(trial):
    return {
        "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_categorical("epochs", [3, 5]),
    }


def build_gru(params, n_jobs=-1):
    return SeqModel("gru", params)


def suggest_cnn(trial):
    return {
        "channels": trial.suggest_categorical("channels", [16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_categorical("epochs", [3, 5]),
    }


def build_cnn(params, n_jobs=-1):
    return SeqModel("cnn", params)


# kind: ML = 요약 특징 입력, DL = 시퀀스 입력
MODEL_SPECS = {
    "ridge": {"kind": "ML", "suggest": suggest_ridge, "build": build_ridge},
    "rf": {"kind": "ML", "suggest": suggest_rf, "build": build_rf},
    "xgb": {"kind": "ML", "suggest": suggest_xgb, "build": build_xgb},
    "lgbm": {"kind": "ML", "suggest": suggest_lgbm, "build": build_lgbm},
    "gru": {"kind": "DL", "suggest": suggest_gru, "build": build_gru},
    "cnn": {"kind": "DL", "suggest": suggest_cnn, "build": build_cnn},
}

SEQ_MODELS = [name for name in MODEL_SPECS if MODEL_SPECS[name]["kind"] == "DL"]


# 기존 호출 방식 유지용
def suggest_params(name, trial):
    return MODEL_SPECS[name]["suggest"](trial)


def make_model(name, params, n_jobs=-1):
    return MODEL_SPECS[name]["build"](params, n_jobs)


# 앙상블 예측 (model.pkl 형식, df에 TAB_COLS 필요)
def predict_ensemble(ens, df, X_seq):
    X_tab = df[TAB_COLS].to_numpy(dtype="float32")
    base = df["proj_base"].to_numpy(dtype="float64")
    total = np.zeros(len(df))
    for i in range(len(ens["members"])):
        name = ens["members"][i]
        weight = ens["weights"][i]
        total = total + weight * (base + ens["models"][name].predict(X_tab, X_seq))
    return total
