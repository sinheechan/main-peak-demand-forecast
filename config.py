# 프로젝트 공통 설정: 경로, 로그, 여러 파일이 같이 쓰는 값
#   (한 파일에서만 쓰는 값은 그 파일 위쪽에 둠)

import glob
import logging
import os
import sys
from datetime import datetime

# 경로
ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(ROOT, "data", "raw")              # 원본 zip (load_data 입력)
MACHINE_DIR = os.path.join(ROOT, "data", "machines")     # 기기별 parquet, csv (load_data 출력)
OUTPUT_DIR = os.path.join(ROOT, "outputs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")         # 분석 엑셀·그래프 (analyze_data 출력)
PREP_DIR = os.path.join(OUTPUT_DIR, "전처리")             # meta (preprocess 출력)
SPLIT_DIR = os.path.join(ROOT, "data", "학습데이터")
SPLIT_DIRS = {                                           # 분할별 샘플 (preprocess 출력)
    "train": os.path.join(SPLIT_DIR, "Train"),
    "valid": os.path.join(SPLIT_DIR, "Valid"),
    "test": os.path.join(SPLIT_DIR, "Test"),
}
TRAIN_DIR = os.path.join(OUTPUT_DIR, "학습결과")          # 모델·결과 엑셀 (train 출력)
SIM_MODEL_PATH = os.path.join(ROOT, "시뮬레이션", "model", "model.pkl")   # 시뮬레이션 사용 모델 (train --deploy, serve)
SIM_INFO_PATH = os.path.join(ROOT, "시뮬레이션", "model", "model_info.json")   # 목표전력·배율·재생 정보 (train --deploy, edge)
LOG_PATH = os.path.join(ROOT, "logs", "main.log")                 # write_log_print 용

# 데이터
MAIN_METER = "MainTerminal"     # 공장 전체 미터 (예측 대상)
LOCAL_TZ = "Europe/Berlin"      # 공장 현지 시간대

# 예측 구간 (1칸 = 10초 평균, 전처리·학습·서빙이 같아야 함)
STEP_SEC = 10       # 1칸 = 10초
WINDOW = 90         # 15분 = 90칸
HIST = 60           # 최근 10분 = 60칸
FIRST = 12          # 예측 시작: 구간 2분 후
NORM = 1000.0       # 딥러닝 입력 정규화

# 예측 특징 (전처리에서 만들고 학습·서빙 모델이 입력으로 씀)
TAB_COLS = [        # 요약 특징 (트리·선형 모델 입력)
    "elapsed_ratio", "used_kwh", "kw_now", "proj_base", "proj_trend",
    "mean_1m", "mean_5m", "mean_10m", "std_5m", "slope_5m", "max_10m", "min_10m",
    "hour_sin", "hour_cos", "dow",
]
EXTRA_COLS = ["elapsed_ratio", "proj_base", "kw_now", "hour_sin", "hour_cos"]   # 딥러닝 보조 특징


# 파일명용 생성시간 (예: 20261122_170501)
def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# {name}_{생성시간}.{ext} 중 최신 파일 반환, 없으면 None
def latest_file(folder, name, ext):
    files = sorted(glob.glob(os.path.join(folder, name + "_*." + ext)))
    if len(files) == 0:
        return None
    return files[-1]


# 터미널 + 로그 파일(logs/main.log)에 같이 기록하는 logger
#   형식: 2026-11-22 18:30:00 | 메시지
def write_log_print(name):
    logger = logging.getLogger(name)
    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file.setFormatter(formatter)
    logger.addHandler(file)
    return logger


# 출력 경로에 이미 파일이 있으면 목록을 보여 주고 중단 (덮어쓰지 않음)
#   paths: 폴더면 안에 있는 것 전부, 파일이면 그 파일이 있는지 확인
def stop_if_output_exists(paths, logger):
    found = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                found.append(os.path.join(path, name))
        elif os.path.exists(path):
            found.append(path)

    if len(found) == 0:
        return

    logger.error("출력 폴더에 이미 파일이 있어서 중단합니다 (덮어쓰지 않음)")
    for path in found:
        logger.error(f"  {path}")
    logger.error("필요 없으면 지우거나 다른 곳으로 옮긴 뒤 다시 실행하세요.")
    sys.exit(1)
