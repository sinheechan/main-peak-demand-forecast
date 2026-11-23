# 학습 샘플 생성 및 날짜 기준 train / valid / test 분할
#   입력: data/machines/MainTerminal/ 최신 parquet
#   출력: outputs/
#         Train/, Valid/, Test/   {분할}_{생성시간}.parquet  예측 시점별 특징, 정답
#                                 {분할}_seq_{생성시간}.npy  최근 10분 시퀀스 (딥러닝 입력)
#         전처리/meta_{생성시간}.json   배율, 목표전력, 분할 기간
#   실행: python 3_preprocess.py
#   - make_point는 시뮬레이션/serving/serve.py에도 같은 코드가 있음 (학습과 서빙 특징이 같아야 함)

import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import config

# 전처리 설정
P95_KW = 1400        # 가동 중 상위 5% 전력 기준 규모 (kW)
LAST = 84            # 예측 끝: 구간 14분 (칸)
EVERY = 6            # 예측 간격: 1분 (칸)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15   # 나머지 0.15는 test
TARGET_PCT = 95      # 목표전력: train 15분 평균 상위 5%
SPLITS = ["train", "valid", "test"]

logger = config.write_log_print("preprocess")


# 현지 시각 변환 (시간대 없는 값은 이미 현지 시각으로 간주)
def to_local(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert(config.LOCAL_TZ)


# 공장 전체 전력 로드 -> (10초 평균 kW, 배율)
def load_main_steps(path):
    df = pd.read_parquet(path, columns=["tmp_timestamp", "P_kW"])

    kw = pd.Series(df["P_kW"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(df["tmp_timestamp"]))
    kw = kw.clip(lower=0)  # 음수 = 측정 잡음

    # 규모 맞추기: 정지 구간(최대의 5% 이하) 제외 후 상위 5%를 P95_KW로
    running = kw[kw > kw.max() * 0.05]
    scale = P95_KW / np.percentile(running, 95)

    # 10초 평균 (빈 칸은 보간 없이 NaN)
    steps = (kw * scale).resample(f"{config.STEP_SEC}s").mean()
    return steps, scale


# 예측 시점 1개의 특징 -> (요약 특징, 최근 10분 시퀀스), 빈 칸이 있으면 None
#   values: 10초 평균 배열 / start: 구간 시작 칸 / k: 경과 칸 수 / ws: 구간 시작 시각
#   - 현재 칸(start + k) 직전까지만 사용 (미래 정보 차단)
def make_point(values, start, k, ws):
    now = start + k
    if now - config.HIST < 0 or now > len(values):
        return None

    hist = values[now - config.HIST:now]   # 최근 10분
    inside = values[start:now]             # 이번 구간 경과분
    if len(hist) < config.HIST or np.isnan(hist).any() or np.isnan(inside).any():
        return None

    kw_now = float(hist[-1])
    elapsed_h = k * config.STEP_SEC / 3600
    remain_h = 0.25 - elapsed_h                              # 15분 = 0.25시간
    used_kwh = float(inside.sum()) * config.STEP_SEC / 3600  # 사용 전력량

    # 기준 계산식: 현재 전력 유지 가정 (현장 제어기와 동일)
    proj_base = (used_kwh + kw_now * remain_h) / 0.25

    # 추세 계산식: 최근 5분 기울기 유지 가정
    last5 = hist[-30:]
    slope = float(np.polyfit(np.arange(len(last5)), last5, 1)[0])   # kW / 칸
    slope_h = slope * 3600 / config.STEP_SEC
    future_kwh = kw_now * remain_h + 0.5 * slope_h * remain_h ** 2
    if future_kwh < 0:
        future_kwh = 0.0
    proj_trend = (used_kwh + future_kwh) / 0.25

    now_time = to_local(ws) + pd.Timedelta(seconds=k * config.STEP_SEC)
    hour = now_time.hour + now_time.minute / 60

    tab = {
        "elapsed_ratio": k / config.WINDOW,
        "used_kwh": used_kwh,
        "kw_now": kw_now,
        "proj_base": proj_base,
        "proj_trend": proj_trend,
        "mean_1m": float(hist[-6:].mean()),
        "mean_5m": float(hist[-30:].mean()),
        "mean_10m": float(hist.mean()),
        "std_5m": float(hist[-30:].std()),
        "slope_5m": slope,
        "max_10m": float(hist.max()),
        "min_10m": float(hist.min()),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),   # 시각 순환 표현 (23시 ≈ 0시)
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow": now_time.dayofweek,
    }
    seq = (hist / config.NORM).astype("float32")
    return tab, seq


# 샘플 생성 (15분 구간별, 2~14분 시점 1분 간격)
def make_samples(steps):
    values = steps.to_numpy(dtype="float64")
    first = steps.index[0].ceil("15min")
    last = steps.index[-1].floor("15min") - pd.Timedelta(minutes=15)
    window_list = pd.date_range(first, last, freq="15min")

    rows = []
    seqs = []
    for ws in tqdm(window_list):
        start = steps.index.get_loc(ws)
        window = values[start:start + config.WINDOW]

        # 90칸이 모두 있는 구간만 사용 -- 정답 계산 가능하기 때문
        if len(window) < config.WINDOW or np.isnan(window).any():
            continue
        actual = float(window.mean())

        for k in range(config.FIRST, LAST + 1, EVERY):
            point = make_point(values, start, k, ws)
            if point is None:
                continue
            tab, seq = point
            tab["window_start"] = ws
            tab["date"] = str(to_local(ws).date())
            tab["elapsed_min"] = k * config.STEP_SEC / 60
            tab["actual_kw"] = actual
            rows.append(tab)
            seqs.append(seq)

    return pd.DataFrame(rows), np.stack(seqs)


# 날짜 기준 분할 (train 70% / valid 15% / test 15%)
def split_by_date(df):
    dates = sorted(df["date"].unique())
    n_train = int(len(dates) * TRAIN_RATIO)
    n_valid = int(len(dates) * VALID_RATIO)
    train_dates = dates[:n_train]
    valid_dates = dates[n_train:n_train + n_valid]

    df["split"] = "test"
    df.loc[df["date"].isin(train_dates), "split"] = "train"
    df.loc[df["date"].isin(valid_dates), "split"] = "valid"
    return df


# 분할별 기간, 샘플 수, 피크 비율
def summarize_splits(df):
    split_info = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        info = {
            "start": part["date"].min(),
            "end": part["date"].max(),
            "days": int(part["date"].nunique()),
            "samples": int(len(part)),
            "peak_ratio": float(part["is_peak"].mean()),
        }
        split_info[split] = info
        logger.info(f"{split} {info['start']} ~ {info['end']} | {info['days']}일 | "
                    f"샘플 {info['samples']} | 피크 비율 {info['peak_ratio'] * 100:.1f}%")
    return split_info


def main():
    # 출력 폴더에 이미 파일이 있으면 중단 (덮어쓰지 않음)
    out_folders = [config.PREP_DIR] + list(config.SPLIT_DIRS.values())
    config.stop_if_output_exists(out_folders, logger)

    # 공장 전체 전력 로드 (10초 평균)
    folder = os.path.join(config.MACHINE_DIR, config.MAIN_METER)
    source_path = config.latest_file(folder, config.MAIN_METER, "parquet")
    if source_path is None:
        logger.error(f"{config.MAIN_METER} parquet 없음: {folder} -> 1_load_data.py 먼저 실행")
        sys.exit(1)
    steps, scale = load_main_steps(source_path)
    empty_pct = steps.isna().mean() * 100
    logger.info(f"10초 칸 {len(steps)}개, 빈 칸 {empty_pct:.2f}%, 배율 x{scale:.2f}")

    df, X_seq = make_samples(steps)
    logger.info(f"샘플 수: {len(df)}")

    df = split_by_date(df)

    # 학습 타깃: 잔차 = 실제 - 기준 계산식
    df["residual"] = df["actual_kw"] - df["proj_base"]

    # 목표전력: train 15분 평균 상위 5%
    train_windows = df[df["split"] == "train"].drop_duplicates("window_start")["actual_kw"]
    target_kw = float(np.percentile(train_windows, TARGET_PCT))
    df["is_peak"] = df["actual_kw"] >= target_kw
    logger.info(f"목표전력 {target_kw:.0f} kW")

    split_info = summarize_splits(df)

    # 저장 (분할별 폴더 + meta)
    run_stamp = config.stamp()
    for split in SPLITS:
        mask = (df["split"] == split).to_numpy()
        folder = config.SPLIT_DIRS[split]
        os.makedirs(folder, exist_ok=True)
        df[mask].to_parquet(os.path.join(folder, f"{split}_{run_stamp}.parquet"), index=False)
        np.save(os.path.join(folder, f"{split}_seq_{run_stamp}.npy"), X_seq[mask])

    os.makedirs(config.PREP_DIR, exist_ok=True)
    meta = {
        "stamp": run_stamp,
        "source": os.path.basename(source_path),
        "scale": float(scale),
        "target_kw": target_kw,
        "split": split_info,
    }
    meta_path = os.path.join(config.PREP_DIR, f"meta_{run_stamp}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"완료: {run_stamp}")


if __name__ == "__main__":
    main()
