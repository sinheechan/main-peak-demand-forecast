# 예측 모델 정의 (ridge, rf, xgb, lgbm, gru, cnn)
#   사용: model = make_model("lgbm", params)
#         model.fit(X_tab, X_seq, y)       y = 실제 15분 평균 - 기준 계산식
#         model.predict(X_tab, X_seq)      보정값 예측
#   - 최종 예측 = 기준 계산식 + 보정값
#   - model.pkl에 클래스 이름이 저장되므로 TabModel, SeqModel, GRUNet, CNNNet 이름 변경 금지

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

SEQ_MODELS = ["gru", "cnn"]   # 시퀀스 입력 모델
BATCH = 512

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


# Optuna 탐색 범위
def suggest_params(name, trial):
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        }

    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if name == "gru":
        return {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_categorical("epochs", [3, 5]),
        }

    if name == "cnn":
        return {
            "channels": trial.suggest_categorical("channels", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_categorical("epochs", [3, 5]),
        }


# 트리·선형 모델 (요약 특징 입력)
class TabModel:

    def __init__(self, name, params):
        self.name = name
        self.params = params

        if name == "ridge":
            # 특징 단위가 달라 표준화 후 적용
            self.model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=params["alpha"]))])
        elif name == "rf":
            self.model = RandomForestRegressor(**params, n_jobs=-1, random_state=config.SEED)
        elif name == "lgbm":
            self.model = LGBMRegressor(**params, subsample_freq=1, n_jobs=-1, verbose=-1, random_state=config.SEED)
        elif name == "xgb":
            self.model = XGBRegressor(**params, tree_method="hist", n_jobs=-1, random_state=config.SEED)

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
class SeqModel:

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.net = None

    # 보조 특징 추출 (kW 값은 NORM으로 정규화)
    def make_extra(self, X_tab):
        extra = np.zeros((len(X_tab), len(EXTRA_COLS)), dtype="float32")
        for j in range(len(EXTRA_COLS)):
            col = EXTRA_COLS[j]
            extra[:, j] = X_tab[:, TAB_COLS.index(col)]
            if col == "proj_base" or col == "kw_now":
                extra[:, j] = extra[:, j] / config.NORM
        return extra

    def fit(self, X_tab, X_seq, y):
        torch.manual_seed(config.SEED)
        if self.name == "gru":
            self.net = GRUNet(self.params["hidden"], len(EXTRA_COLS))
        else:
            self.net = CNNNet(self.params["channels"], len(EXTRA_COLS))

        seq = torch.tensor(X_seq, dtype=torch.float32)
        extra = torch.tensor(self.make_extra(X_tab))
        target = torch.tensor(y / config.NORM, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        loss_fn = nn.HuberLoss()

        for epoch in range(self.params["epochs"]):
            self.net.train()
            order = torch.randperm(len(target))
            for start in range(0, len(target), BATCH):
                idx = order[start:start + BATCH]
                optimizer.zero_grad()
                loss = loss_fn(self.net(seq[idx], extra[idx]), target[idx])
                loss.backward()
                optimizer.step()

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


def make_model(name, params):
    if name in SEQ_MODELS:
        return SeqModel(name, params)
    return TabModel(name, params)


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
