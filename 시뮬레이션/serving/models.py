# 서빙용 모델 클래스 (model.pkl 로드 + 예측에 필요한 부분만)
#   - 루트 models.py와 클래스 이름·구조가 같아야 model.pkl이 열림
#     (pkl에 "models.TabModel" 처럼 파일 이름 + 클래스 이름으로 저장됨)
#   - 학습(fit) 코드는 루트 models.py에만 있음

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

BATCH = 512


# 트리·선형 모델 (요약 특징 입력)
class TabModel:

    def __init__(self, name, params, model):
        self.name = name
        self.params = params
        self.model = model

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
        extra = np.zeros((len(X_tab), len(config.EXTRA_COLS)), dtype="float32")
        for j in range(len(config.EXTRA_COLS)):
            col = config.EXTRA_COLS[j]
            extra[:, j] = X_tab[:, config.TAB_COLS.index(col)]
            if col == "proj_base" or col == "kw_now":
                extra[:, j] = extra[:, j] / config.NORM
        return extra

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


# 앙상블 예측 (model.pkl 형식, df에 TAB_COLS 필요)
def predict_ensemble(ens, df, X_seq):
    X_tab = df[config.TAB_COLS].to_numpy(dtype="float32")
    base = df["proj_base"].to_numpy(dtype="float64")
    total = np.zeros(len(df))
    for i in range(len(ens["members"])):
        name = ens["members"][i]
        weight = ens["weights"][i]
        total = total + weight * (base + ens["models"][name].predict(X_tab, X_seq))
    return total
