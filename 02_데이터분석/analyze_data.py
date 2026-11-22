# 기기별 데이터 분석
# 입력: 01_데이터관리/machines/{기기}/ 최신 parquet
# 출력: 02_데이터분석/reports/데이터분석_{생성시간}/
#       1~3번은 엑셀, 4~7번은 그래프(png)

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import write_log_print

# 분석 설정
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")   # 결과 저장 폴더
GAP_SEC = 60     # 누락 기준 (초)
IQR_K = 1.5      # IQR 이상치 배수
Z_LIMIT = 3.0    # z-score 이상치 기준


# 기기 목록
machines = []
for name in sorted(os.listdir(config.MACHINE_DIR)):
    if os.path.isdir(os.path.join(config.MACHINE_DIR, name)):
        machines.append(name)

if len(machines) == 0:
    print("기기 데이터 없음 -> load_data.py 먼저 실행")
    sys.exit()


# 결과 담을 곳
summary = []     # 1. 요약
stats = []       # 2. 기술통계
outliers = []    # 3. 이상치
gaps = []        # 4. 누락구간
coverage = {}    # 5. 일별수집률
pattern = None   # 6. 시간대패턴 (MainTerminal만)
steps = {}       # 7. 장비관계 계산용 10초 평균


for machine in tqdm(machines, desc="분석중"):

    # 파일 읽기
    try:
        path = config.latest_file(os.path.join(config.MACHINE_DIR, machine), machine, "parquet")
        if path is None:
            continue
        df = pd.read_parquet(path)
    except Exception as e:
        write_log_print(f"[오류] {machine} 파일 읽기 실패: {e}")
        continue

    # 분석
    try:
        local = df["tmp_timestamp"].dt.tz_convert(config.LOCAL_TZ).dt.tz_localize(None)  # 현지 시각
        gap = df["tmp_timestamp"].diff().dt.total_seconds()  # 측정 간격(초)
        power = df["P_kW"]

        # 1. 요약
        summary.append({
            "기기": machine,
            "파일": os.path.basename(path),
            "행 수": len(df),
            "start_time": local.min(),
            "end_time": local.max(),
            "간격 중앙값(초)": gap.median(),
            "1분 넘게 끊긴 구간": int((gap > GAP_SEC).sum()),
            "가장 긴 공백(분)": gap.max() / 60,
            "P_kW 최소": power.min(),
            "P_kW 평균": power.mean(),
            "P_kW 최대": power.max(),
            "P_kW 음수 비율": (power < 0).mean(),
            "결측 칸 수": int(df.isna().sum().sum()),
        })

        # 2. 기술통계, 3. 이상치 - 컬럼마다
        for col in df.columns:
            if col in ["tmp_timestamp", "tmp_tzoffset"]:
                continue
            s = df[col].astype("float64")

            stats.append({
                "기기": machine,
                "컬럼": col,
                "개수": int(s.count()),
                "결측": int(s.isna().sum()),
                "평균": s.mean(),
                "표준편차": s.std(),
                "최소": s.min(),
                "25%": s.quantile(0.25),
                "중앙값": s.median(),
                "75%": s.quantile(0.75),
                "99%": s.quantile(0.99),
                "최대": s.max(),
            })

            # IQR
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - IQR_K * iqr
            upper = q3 + IQR_K * iqr
            n_iqr = int(((s < lower) | (s > upper)).sum())

            # z-score (표준편차 0이면 계산 안 함)
            n_z = 0
            if s.std() > 0:
                z = (s - s.mean()) / s.std()
                n_z = int((z.abs() > Z_LIMIT).sum())

            outliers.append({
                "기기": machine,
                "컬럼": col,
                "IQR 하한": lower,
                "IQR 상한": upper,
                "IQR 이상치 수": n_iqr,
                "IQR 이상치 비율": n_iqr / len(s),
                "z-score 이상치 수": n_z,
                "음수 수": int((s < 0).sum()),
                "0 수": int((s == 0).sum()),
            })

        # 4. 누락구간
        for i in np.where(gap > GAP_SEC)[0]:
            gaps.append({
                "기기": machine,
                "끊기기 전": local.iloc[i - 1],
                "재가동": local.iloc[i],
                "길이": gap.iloc[i] / 60,
            })

        # 5. 일별수집률 = 하루 실제 행 수 / 하루 예상 행 수
        expected = 86400 / gap.median()
        per_day = local.dt.date.value_counts().sort_index()
        coverage[machine] = (per_day / expected * 100).clip(upper=100)

        # 6. 시간대패턴 - 공장 전체만
        if machine == config.MAIN_METER:
            pattern = df.groupby([local.dt.day_of_week, local.dt.hour])["P_kW"].mean().unstack()
            pattern.index = ["월", "화", "수", "목", "금", "토", "일"][:len(pattern)]

        # 7. 장비관계용 10초 평균
        s = pd.Series(power.to_numpy(dtype="float64"), index=pd.DatetimeIndex(df["tmp_timestamp"]))
        steps[machine] = s.resample(f"{config.STEP_SEC}s").mean()

    except Exception as e:
        write_log_print(f"[오류] {machine} 분석 실패: {e}")
        continue


# 7. 장비관계 - 공장 전체(MainTerminal)와 비교
main = steps[config.MAIN_METER]
relation = []
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


# 저장 폴더 (실행할 때마다 새로)
out_dir = os.path.join(REPORT_DIR, f"데이터분석_{config.stamp()}")
os.makedirs(out_dir, exist_ok=True)

# 1~3 엑셀
pd.DataFrame(summary).to_excel(os.path.join(out_dir, "1.요약.xlsx"), index=False)
pd.DataFrame(stats).to_excel(os.path.join(out_dir, "2.기술통계.xlsx"), index=False)
pd.DataFrame(outliers).to_excel(os.path.join(out_dir, "3.이상치.xlsx"), index=False)

# 4. 누락구간
if len(gaps) > 0:
    gap_df = pd.DataFrame(gaps)
    plt.figure(figsize=(14, 5))
    plt.scatter(gap_df["끊기기 전"], gap_df["기기"])
    plt.title("누락구간")
    plt.savefig(os.path.join(out_dir, "4.누락구간.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 5. 일별수집률 (행: 기기, 열: 날짜)
cov_df = pd.DataFrame(coverage).T
plt.figure(figsize=(36, 6))  # 날짜가 많아서 가로로 길게
plt.imshow(cov_df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
plt.yticks(range(len(cov_df.index)), cov_df.index)
plt.xticks(range(0, len(cov_df.columns), 7), cov_df.columns[::7], rotation=45)  # 날짜는 일주일 간격
for i in range(cov_df.shape[0]):
    for j in range(cov_df.shape[1]):
        plt.text(j, i, f"{cov_df.iloc[i, j]:.0f}", ha="center", va="center", fontsize=7)
plt.colorbar()
plt.title("일별수집률 (%)")
plt.savefig(os.path.join(out_dir, "5.일별수집률.png"), dpi=150, bbox_inches="tight")
plt.close()

# 6. 시간대패턴 (행: 요일, 열: 시)
if pattern is not None:
    plt.figure(figsize=(14, 4))
    plt.imshow(pattern.values, aspect="auto", cmap="YlOrRd")
    plt.yticks(range(len(pattern.index)), pattern.index)
    plt.xticks(range(len(pattern.columns)), pattern.columns)
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            plt.text(j, i, f"{pattern.iloc[i, j]:.0f}", ha="center", va="center", fontsize=7)
    plt.colorbar()
    plt.title("시간대패턴(평균 P_kW)")
    plt.savefig(os.path.join(out_dir, "6.시간대패턴.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 7. 장비관계
plt.figure(figsize=(10, 5))
plt.barh(relation["장비"], relation["공장 전체와 상관계수"])
plt.title("공장 전체와 상관계수")
plt.savefig(os.path.join(out_dir, "7.장비관계.png"), dpi=150, bbox_inches="tight")
plt.close()

write_log_print(f"완료: {out_dir}")
