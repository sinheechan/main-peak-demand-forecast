# ML 그림자 운영: 30초마다 예측 기록, 구간 종료 후 계산식과 오차 비교
#   입력: 시뮬레이션/model/model.pkl (4_train.py --deploy로 배포)
#         시뮬레이션 DB raw.meter_reading, ops.demand_15min
#   출력: ml.forecast (예측 기록), ml.window_eval (구간별 계산식 vs ML 오차)
#   실행: python 시뮬레이션/serving/serve.py (simulator, controller, server 실행 중)
#   - 제어는 controller의 기준 계산식이 담당, ML은 기록만 함
#   - make_point는 루트 3_preprocess.py와 같은 코드 (학습과 서빙 특징이 같아야 함)
#   - model.pkl 로드에 같은 폴더의 models.py가 필요함 (루트 models.py 말고 이 폴더 것을 씀)
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

SERVING_DIR = Path(__file__).resolve().parent
sys.path.append(str(SERVING_DIR.parents[1]))   # 루트(config.py)는 뒤에 추가 -> models는 serving 것이 먼저
import config  # noqa: E402
from models import predict_ensemble  # noqa: E402  (serving/models.py)

INTERVAL = 30          # 예측 주기 (초)

load_dotenv(SERVING_DIR / ".env")
DB_URL = os.getenv("DB_URL")

logger = config.write_log_print("serve")


def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


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
    return series.resample(f"{config.STEP_SEC}s").mean()


# 시뮬레이션 DB 전력 로드 -> 10초 평균 kW (배율 적용된 값)
def load_db(db_url, since=None):
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


def create_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml.forecast (
              ts TIMESTAMP PRIMARY KEY,
              window_start TIMESTAMP,
              elapsed_min REAL,
              formula_kw REAL,
              ml_kw REAL,
              run_id TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml.window_eval (
              window_start TIMESTAMP PRIMARY KEY,
              actual_kw REAL,
              formula_mae REAL,
              ml_mae REAL,
              n_points INT,
              run_id TEXT
            )"""))


def load_model():
    ens = joblib.load(config.SIM_MODEL_PATH)
    members = "+".join(ens["members"])
    logger.info(f"모델 로드: {members} (run {ens['run_id']}, valid MAE {ens.get('valid_mae', 0):.1f} kW)")
    return ens


def predict(engine, ens):
    # 현재 구간 예측 1회
    t = now_kst()
    ws = t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)
    k = int((t - ws).total_seconds() // config.STEP_SEC)
    if k < config.FIRST:
        return                         # 구간 초반 2분은 예측 보류

    steps = load_db(DB_URL, since=t - timedelta(minutes=30))
    if len(steps) == 0 or ws not in steps.index:
        logger.info(f"[{t:%H:%M:%S}] 데이터 부족 -> 건너뜀")
        return

    values = steps.to_numpy(dtype="float64")
    point = make_point(values, steps.index.get_loc(ws), k, ws)
    if point is None:
        logger.info(f"[{t:%H:%M:%S}] 빈 구간 있음 -> 건너뜀")
        return

    tab, seq = point
    ml_kw = float(predict_ensemble(ens, pd.DataFrame([tab]), seq[None, :])[0])
    formula_kw = tab["proj_base"]

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ml.forecast VALUES (:ts, :ws, :em, :f, :m, :run)
            ON CONFLICT (ts) DO NOTHING"""),
            {"ts": t, "ws": ws, "em": k * config.STEP_SEC / 60,
             "f": formula_kw, "m": ml_kw, "run": ens["run_id"]})

    logger.info(f"[{t:%H:%M:%S}] 구간 {ws:%H:%M} | 계산식 {formula_kw:7.0f} kW | ML {ml_kw:7.0f} kW")


def evaluate(engine, ens):
    # 마감 구간 평가 (ops.demand_15min 확정 평균과 비교)
    sql = text("""
        SELECT f.window_start,
               d.avg_kw AS actual_kw,
               avg(abs(f.formula_kw - d.avg_kw)) AS formula_mae,
               avg(abs(f.ml_kw - d.avg_kw)) AS ml_mae,
               count(*) AS n_points
        FROM ml.forecast f
        JOIN ops.demand_15min d ON d.window_start = f.window_start AND d.final
        WHERE f.window_start NOT IN (SELECT window_start FROM ml.window_eval)
        GROUP BY f.window_start, d.avg_kw
        ORDER BY f.window_start""")

    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()
        for r in rows:
            conn.execute(text("""
                INSERT INTO ml.window_eval VALUES (:ws, :a, :fm, :mm, :n, :run)
                ON CONFLICT (window_start) DO NOTHING"""),
                {"ws": r["window_start"], "a": r["actual_kw"], "fm": r["formula_mae"],
                 "mm": r["ml_mae"], "n": r["n_points"], "run": ens["run_id"]})

            winner = "계산식 승"
            if r["ml_mae"] < r["formula_mae"]:
                winner = "ML 승"
            logger.info(f"[구간 평가] {r['window_start']:%H:%M} 실제 {r['actual_kw']:,.0f} kW | "
                        f"계산식 오차 {r['formula_mae']:.0f} kW | ML 오차 {r['ml_mae']:.0f} kW | {winner}")


def main():
    if not os.path.exists(config.SIM_MODEL_PATH):
        logger.error("시뮬레이션/model/model.pkl 이 없습니다. "
                     "먼저 python 4_train.py --deploy 를 실행하세요.")
        return
    if not DB_URL:
        logger.error("serving/.env 에 DB_URL 이 없습니다. .env.example 을 참고하세요.")
        return

    engine = create_engine(DB_URL, pool_pre_ping=True)
    create_tables(engine)
    ens = load_model()

    while True:
        t0 = time.time()
        try:
            predict(engine, ens)
            evaluate(engine, ens)
        except Exception as exc:
            logger.error(f"오류: {exc}")
        time.sleep(max(0.0, INTERVAL - (time.time() - t0)))


if __name__ == "__main__":
    main()
