# 기기별 데이터 분석 결과를 엑셀 1개로 저장
#   입력: 01_데이터관리/machines/{기기}/ 최신 parquet
#   출력: 02_데이터분석/reports/데이터분석_{생성시간}.xlsx
#   실행: python 02_데이터분석/analyze_data.py
#   시트: 요약, 기술통계, 이상치, 누락구간, 일별수집률, 시간대패턴, 장비관계

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 1. 기기 목록
machines = sorted(os.listdir(config.MACHINE_DIR))
machines = [m for m in machines if os.path.isdir(os.path.join(config.MACHINE_DIR, m))]
if len(machines) == 0:
    print("기기 데이터가 없음. 먼저 python 01_데이터관리/load_data.py")
    sys.exit()

# 2. 기기별 분석
summary = []
stats = []
outliers = []
gaps = []
coverage = {}
steps = {}       # 기기별 10초 평균 P_kW (장비관계용)
pattern = None

for machine in tqdm(machines):
    path = config.latest_file(os.path.join(config.MACHINE_DIR, machine), machine, "parquet")
    if path is None:
        continue
    df = pd.read_parquet(path)

    # 공통 값
    local = df["ts_utc"].dt.tz_convert(config.LOCAL_TZ).dt.tz_localize(None)   # 현지 시각
    gap = df["ts_utc"].diff().dt.total_seconds()   # 측정 간격 (초)
    power = df["P_kW"]

    # 요약
    summary.append({
        "기기": machine,
        "행 수": len(df),
        "시작 (현지)": local.min(),
        "끝 (현지)": local.max(),
        "간격 중앙값 (초)": gap.median(),
        "1분 넘게 끊긴 곳": int((gap > config.GAP_SEC).sum()),
        "가장 긴 공백 (분)": gap.max() / 60,
        "P_kW 최소": power.min(),
        "P_kW 평균": power.mean(),
        "P_kW 최대": power.max(),
        "P_kW 음수 비율": (power < 0).mean(),
        "결측 칸 수": int(df.isna().sum().sum()),
        "파일": os.path.basename(path),
    })

    # 기술통계, 이상치 (수치 컬럼별)
    for col in df.columns:
        if col == "ts_utc" or col == "tz_offset":
            continue
        s = df[col].astype("float64")

        stats.append({
            "기기": machine, "컬럼": col, "개수": int(s.count()), "결측": int(s.isna().sum()),
            "평균": s.mean(), "표준편차": s.std(), "최소": s.min(),
            "25%": s.quantile(0.25), "중앙값": s.median(), "75%": s.quantile(0.75),
            "99%": s.quantile(0.99), "최대": s.max(),
        })

        # IQR 기준: Q1 - k*IQR 미만 또는 Q3 + k*IQR 초과
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - config.IQR_K * iqr
        upper = q3 + config.IQR_K * iqr
        n_iqr = int(((s < lower) | (s > upper)).sum())

        # z-score 기준: |z| > Z_LIMIT
        n_z = 0
        if s.std() > 0:
            n_z = int((((s - s.mean()) / s.std()).abs() > config.Z_LIMIT).sum())

        outliers.append({
            "기기": machine, "컬럼": col, "IQR 하한": lower, "IQR 상한": upper,
            "IQR 이상치 수": n_iqr, "IQR 이상치 비율": n_iqr / len(s),
            "z-score 이상치 수": n_z, "음수 수": int((s < 0).sum()), "0 수": int((s == 0).sum()),
        })

    # 누락구간 (GAP_SEC 초과)
    for i in np.where(gap > config.GAP_SEC)[0]:
        gaps.append({
            "기기": machine,
            "끊기기 전 (현지)": local.iloc[i - 1],
            "다시 들어옴 (현지)": local.iloc[i],
            "길이 (분)": gap.iloc[i] / 60,
        })

    # 일별수집률 = 실제 행 수 / 예상 행 수(86400초 / 측정 간격)
    expected = 86400 / gap.median()
    per_day = local.dt.date.value_counts().sort_index()
    coverage[machine] = (per_day / expected * 100).clip(upper=100)

    # 장비관계용 10초 평균
    s = pd.Series(power.to_numpy(dtype="float64"), index=pd.DatetimeIndex(df["ts_utc"]))
    steps[machine] = s.resample(str(config.STEP_SEC) + "s").mean()

    # 시간대패턴 (공장 전체, 요일 x 시간 평균)
    if machine == config.MAIN_METER:
        pattern = df.groupby([local.dt.dayofweek, local.dt.hour])["P_kW"].mean().unstack()
        pattern.index = ["월", "화", "수", "목", "금", "토", "일"][:len(pattern)]

# 3. 장비관계 (공장 전체 대비 비중, 상관)
relation = []
main = steps[config.MAIN_METER]
for machine in steps:
    if machine == config.MAIN_METER:
        continue
    s = steps[machine].clip(lower=0)
    relation.append({
        "장비": machine,
        "평균 전력 (kW)": s.mean(),
        "공장 전체 대비 비중": s.mean() / main.mean(),
        "공장 전체와 상관계수": main.corr(s),
        "가동 비율 (최대의 10% 초과)": (s > s.max() * 0.1).mean(),
    })
relation = pd.DataFrame(relation).sort_values("공장 전체와 상관계수", ascending=False)

# 4. 저장
os.makedirs(config.REPORT_DIR, exist_ok=True)
out_path = os.path.join(config.REPORT_DIR, "데이터분석_" + config.stamp() + ".xlsx")
with pd.ExcelWriter(out_path) as writer:
    pd.DataFrame(summary).to_excel(writer, sheet_name="1_요약", index=False)
    pd.DataFrame(stats).to_excel(writer, sheet_name="2_기술통계", index=False)
    pd.DataFrame(outliers).to_excel(writer, sheet_name="3_이상치", index=False)
    pd.DataFrame(gaps).to_excel(writer, sheet_name="4_누락구간", index=False)
    pd.DataFrame(coverage).rename_axis("날짜 (현지)").to_excel(writer, sheet_name="5_일별수집률")
    if pattern is not None:
        pattern.rename_axis("요일 \\ 시").to_excel(writer, sheet_name="6_시간대패턴")
    relation.to_excel(writer, sheet_name="7_장비관계", index=False)

config.log("완료: " + out_path)
