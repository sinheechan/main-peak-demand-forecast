# ML 그림자 운영: 30초마다 예측 기록, 구간 종료 후 계산식과 오차 비교
#   입력: 04_시뮬레이션/model/model.pkl (train.py --deploy로 배포)
#         시뮬레이션 DB raw.meter_reading, ops.demand_15min
#   출력: ml.forecast (예측 기록), ml.window_eval (구간별 계산식 vs ML 오차)
#   실행: python 04_시뮬레이션/serving/serve.py (simulator, controller, server 실행 중)
#   - 제어는 controller의 기준 계산식이 담당, ML은 기록만 함
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

SERVING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVING_DIR.parents[1]))
import config  # noqa: E402
from common import features  # noqa: E402
from common.models import predict_ensemble  # noqa: E402

INTERVAL = 30          # 예측 주기 (초)

load_dotenv(SERVING_DIR / ".env")
DB_URL = os.getenv("DB_URL")


def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


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
    print(f"모델 로드: {'+'.join(ens['members'])} (run {ens['run_id']}, valid MAE {ens.get('valid_mae', 0):.1f} kW)")
    return ens


def predict(engine, ens):
    # 현재 구간 예측 1회
    t = now_kst()
    ws = t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)
    k = int((t - ws).total_seconds() // config.STEP_SEC)
    if k < config.FIRST:
        return                         # 구간 초반 2분은 예측 보류

    steps = features.load_db(DB_URL, since=t - timedelta(minutes=30))
    if len(steps) == 0 or ws not in steps.index:
        print(f"[{t:%H:%M:%S}] 데이터 부족 -> 건너뜀")
        return

    values = steps.to_numpy(dtype="float64")
    point = features.make_point(values, steps.index.get_loc(ws), k, ws)
    if point is None:
        print(f"[{t:%H:%M:%S}] 빈 구간 있음 -> 건너뜀")
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

    print(f"[{t:%H:%M:%S}] 구간 {ws:%H:%M} | 계산식 {formula_kw:7.0f} kW | ML {ml_kw:7.0f} kW")


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
            print(f"[구간 평가] {r['window_start']:%H:%M} 실제 {r['actual_kw']:,.0f} kW | "
                  f"계산식 오차 {r['formula_mae']:.0f} kW | ML 오차 {r['ml_mae']:.0f} kW | {winner}")


def main():
    if not os.path.exists(config.SIM_MODEL_PATH):
        print("04_시뮬레이션/model/model.pkl 이 없습니다. "
              "먼저 python 03_전처리_모델학습/train.py --deploy 를 실행하세요.")
        return
    if not DB_URL:
        print("serving/.env 에 DB_URL 이 없습니다. .env.example 을 참고하세요.")
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
            print("오류:", exc)
        time.sleep(max(0.0, INTERVAL - (time.time() - t0)))


if __name__ == "__main__":
    main()
