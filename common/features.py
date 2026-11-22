# 예측 특징 생성 (03 전처리·학습, 04 서빙 공용)
#   - 10초 평균(1칸) 기준, 15분 구간(90칸) 안의 예측 시점마다 특징 생성
#   - 학습과 서빙이 같은 make_point를 써야 결과가 일치함

import os

import numpy as np
import pandas as pd

import config

# 요약 특징 (트리·선형 모델 입력)
TAB_COLS = [
    "elapsed_ratio", "used_kwh", "kw_now", "proj_base", "proj_trend",
    "mean_1m", "mean_5m", "mean_10m", "std_5m", "slope_5m", "max_10m", "min_10m",
    "hour_sin", "hour_cos", "dow",
]
# 보조 특징 (딥러닝에서 시퀀스와 함께 입력)
EXTRA_COLS = ["elapsed_ratio", "proj_base", "kw_now", "hour_sin", "hour_cos"]


# 현지 시각 변환 (시간대 없는 값은 이미 현지 시각으로 간주)
def to_local(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert(config.LOCAL_TZ)


# 10초 평균 (빈 칸은 보간 없이 NaN)
def to_steps(series):
    series = series.sort_index()
    series = series[~series.index.duplicated()]
    return series.resample(str(config.STEP_SEC) + "s").mean()


# 공장 전체 전력 로드 -> (10초 평균 kW, 배율, 파일 경로)
def load_main_steps():
    folder = os.path.join(config.MACHINE_DIR, config.MAIN_METER)
    path = config.latest_file(folder, config.MAIN_METER, "parquet")
    df = pd.read_parquet(path, columns=["ts_utc", "P_kW"])

    kw = pd.Series(df["P_kW"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(df["ts_utc"]))
    kw = kw.clip(lower=0)  # 음수 = 측정 잡음

    # 규모 맞추기: 정지 구간(최대의 5% 이하) 제외 후 상위 5%를 P95_KW로
    running = kw[kw > kw.max() * 0.05]
    scale = config.P95_KW / np.percentile(running, 95)

    return to_steps(kw * scale), scale, path


# 시뮬레이션 DB 전력 로드 -> 10초 평균 kW (서빙용, 배율 적용된 값)
def load_db(db_url, since=None):
    from sqlalchemy import create_engine, text

    sql = "SELECT ts, kw FROM raw.meter_reading"
    params = {}
    if since is not None:
        sql = sql + " WHERE ts >= :since"
        params["since"] = since
    sql = sql + " ORDER BY ts"

    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if len(df) == 0:
        return pd.Series(dtype="float64")
    series = pd.Series(df["kw"].to_numpy(dtype="float64"), index=pd.to_datetime(df["ts"]))
    return to_steps(series)


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
