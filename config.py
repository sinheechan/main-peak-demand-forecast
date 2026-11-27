# 프로젝트 공통 설정 (경로, 상수, 공용 함수)
#   사용: import config

import glob
import os
from datetime import datetime

# 경로
ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(ROOT, "01_데이터관리", "raw")             # 원본 zip
OUTPUT_DIR = os.path.join(ROOT, "output")                       # 모든 산출물

MACHINE_DIR = os.path.join(OUTPUT_DIR, "machines")    # 기기별 parquet, csv
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")      # 분석 엑셀
PREP_DIR = os.path.join(OUTPUT_DIR, "전처리")          # meta
SPLIT_DIRS = {                                        # 분할별 샘플
    "train": os.path.join(OUTPUT_DIR, "Train"),
    "valid": os.path.join(OUTPUT_DIR, "Valid"),
    "test": os.path.join(OUTPUT_DIR, "Test"),
}
TRAIN_DIR = os.path.join(OUTPUT_DIR, "학습결과")
SIM_MODEL_PATH = os.path.join(ROOT, "04_시뮬레이션", "model", "model.pkl")   # 시뮬레이션 사용 모델
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "peak_demand.log")

# 데이터
MAIN_METER = "MainTerminal"     # 공장 전체 미터 (예측 대상)
LOCAL_TZ = "Europe/Berlin"      # 공장 현지 시간대

# 저장할 컬럼 (원본 362개 중 일부, 2상 기기는 없는 컬럼 제외)
KEEP_COLS = ["SensorDateTime", "P_kW", "Q_kvar", "S_kVA",
             "I1_A", "I2_A", "I3_A", "V1_V", "V2_V", "V3_V", "F_Hz", "L_F"]

# 분석
GAP_SEC = 60        # 누락 기준 (초)
IQR_K = 1.5         # IQR 이상치 배수
Z_LIMIT = 3.0       # z-score 이상치 기준

# 전처리 (1칸 = 10초 평균)
P95_KW = 1400       # 가동 중 상위 5% 전력 기준 규모 (kW)
STEP_SEC = 10       # 1칸 = 10초
WINDOW = 90         # 15분 = 90칸
HIST = 60           # 최근 10분 = 60칸
FIRST = 12          # 예측 시작: 구간 2분 후
LAST = 84           # 예측 끝: 구간 14분
EVERY = 6           # 예측 간격: 1분
EARLY_MIN = 8       # 초반 기준 (분)
NORM = 1000.0       # 딥러닝 입력 정규화

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15  # 나머지 0.15는 test

# 학습
MODELS = ["ridge", "rf", "xgb", "lgbm", "gru", "cnn"]
TRIALS = 20         # 모델별 Optuna 시도 횟수 (딥러닝은 1/3)
TUNE_ROWS = 30000   # 튜닝용 샘플 수
TARGET_PCT = 95     # 목표전력: train 15분 평균 상위 5%
SEED = 42

# 시뮬레이션
REPLAY_SPLIT = "valid"   # 재생 기간 (test는 연휴라 피크가 거의 없음)


# 파일명용 생성시간 (예: 20261122_170501)
def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# {name}_{생성시간}.{ext} 중 최신 파일 반환, 없으면 None
def latest_file(folder, name, ext):
    files = sorted(glob.glob(os.path.join(folder, name + "_*." + ext)))
    if len(files) == 0:
        return None
    return files[-1]


# 콘솔 출력 + 로그 파일 기록
def write_log_print(msg):
    line = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " | " + str(msg)
    print(line)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
