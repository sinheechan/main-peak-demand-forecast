# 학습 샘플 생성 및 날짜 기준 train / valid / test 분할
#   입력: 01_데이터관리/machines/MainTerminal/ 최신 parquet
#   출력: 03_전처리_모델학습/outputs/
#         Train/, Valid/, Test/   {분할}_{생성시간}.parquet  예측 시점별 특징, 정답
#                                 {분할}_seq_{생성시간}.npy  최근 10분 시퀀스 (딥러닝 입력)
#         전처리/meta_{생성시간}.json   배율, 목표전력, 분할 기간
#   실행: python 03_전처리_모델학습/preprocess.py

import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from common import features

# 1. 공장 전체 전력 로드 (10초 평균)
steps, scale, source_path = features.load_main_steps()
config.log("10초 칸 " + str(len(steps)) + "개, 빈 칸 " + str(round(steps.isna().mean() * 100, 2)) + "%, 배율 x" + str(round(scale, 2)))

# 2. 샘플 생성 (15분 구간별, 2~14분 시점 1분 간격)
values = steps.to_numpy(dtype="float64")
first = steps.index[0].ceil("15min")
last = steps.index[-1].floor("15min") - pd.Timedelta(minutes=15)
window_list = pd.date_range(first, last, freq="15min")

rows = []
seqs = []
for ws in tqdm(window_list):
    start = steps.index.get_loc(ws)
    window = values[start:start + config.WINDOW]

    # 90칸이 모두 있는 구간만 사용 (정답 계산 가능)
    if len(window) < config.WINDOW or np.isnan(window).any():
        continue
    actual = float(window.mean())

    for k in range(config.FIRST, config.LAST + 1, config.EVERY):
        point = features.make_point(values, start, k, ws)
        if point is None:
            continue
        tab, seq = point
        tab["window_start"] = ws
        tab["date"] = str(features.to_local(ws).date())   # 현지 날짜
        tab["elapsed_min"] = k * config.STEP_SEC / 60
        tab["actual_kw"] = actual
        rows.append(tab)
        seqs.append(seq)

df = pd.DataFrame(rows)
X_seq = np.stack(seqs)
print("샘플 수:", len(df))

# 3. 날짜 기준 분할 (train 70% / valid 15% / test 15%)
#   - 무작위 분할 시 같은 날이 train, test에 섞여 성능이 과대평가됨
dates = sorted(df["date"].unique())
n_train = int(len(dates) * config.TRAIN_RATIO)
n_valid = int(len(dates) * config.VALID_RATIO)
train_dates = dates[:n_train]
valid_dates = dates[n_train:n_train + n_valid]

df["split"] = "test"
df.loc[df["date"].isin(train_dates), "split"] = "train"
df.loc[df["date"].isin(valid_dates), "split"] = "valid"

# 4. 학습 타깃: 잔차 = 실제 - 기준 계산식
df["residual"] = df["actual_kw"] - df["proj_base"]

# 5. 목표전력: train 15분 평균 상위 5%
train_windows = df[df["split"] == "train"].drop_duplicates("window_start")["actual_kw"]
target_kw = float(np.percentile(train_windows, config.TARGET_PCT))
df["is_peak"] = df["actual_kw"] >= target_kw
config.log("목표전력 " + str(round(target_kw)) + " kW")

# 분할별 기간, 피크 비율 확인
split_info = {}
for split in ["train", "valid", "test"]:
    part = df[df["split"] == split]
    split_info[split] = {
        "start": part["date"].min(),
        "end": part["date"].max(),
        "days": int(part["date"].nunique()),
        "samples": int(len(part)),
        "peak_ratio": float(part["is_peak"].mean()),
    }
    config.log(split + " " + part["date"].min() + " ~ " + part["date"].max()
               + " | 샘플 " + str(len(part)) + " | 피크 비율 " + str(round(part["is_peak"].mean() * 100, 1)) + "%")
    if part["is_peak"].mean() < 0.01:
        config.log("  -> " + split + " 기간은 피크가 거의 없음 (크리스마스 연휴)")

# 6. 분배 확인
print()
print("[분배 확인]")
print(f"{'분할':6s} {'샘플':>8s} {'비율':>7s} {'일수':>5s}  {'기간':23s} {'피크 비율':>8s}")
for split in ["train", "valid", "test"]:
    part = df[df["split"] == split]
    print(f"{split:6s} {len(part):8d} {len(part) / len(df):7.1%} {part['date'].nunique():5d}  "
          f"{part['date'].min()} ~ {part['date'].max()} {part['is_peak'].mean():8.1%}")

overlap = (set(df.loc[df["split"] == "train", "date"]) & set(df.loc[df["split"] != "train", "date"])) \
    | (set(df.loc[df["split"] == "valid", "date"]) & set(df.loc[df["split"] == "test", "date"]))
print("분할 간 겹치는 날짜:", len(overlap), "일")
print("합계 일치:", (df["split"].value_counts().sum() == len(df)))
print()

# 7. 저장 (분할별 폴더 + meta)
run_stamp = config.stamp()

for split in ["train", "valid", "test"]:
    mask = (df["split"] == split).to_numpy()
    folder = config.SPLIT_DIRS[split]
    os.makedirs(folder, exist_ok=True)
    df[mask].to_parquet(os.path.join(folder, split + "_" + run_stamp + ".parquet"), index=False)
    np.save(os.path.join(folder, split + "_seq_" + run_stamp + ".npy"), X_seq[mask])

os.makedirs(config.PREP_DIR, exist_ok=True)
meta = {
    "stamp": run_stamp,
    "source": os.path.basename(source_path),
    "scale": float(scale),
    "target_kw": target_kw,
    "split": split_info,
    "n_samples": int(len(df)),
    "features": features.TAB_COLS,
}
with open(os.path.join(config.PREP_DIR, "meta_" + run_stamp + ".json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

config.log("완료: Train/, Valid/, Test/ 에 {분할}_" + run_stamp + ".parquet, {분할}_seq_" + run_stamp + ".npy, 전처리/meta_" + run_stamp + ".json")
