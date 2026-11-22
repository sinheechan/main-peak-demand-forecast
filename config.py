# 프로젝트 공통 설정 (2개 이상 파일이 같이 쓰는 값과 공용 함수만)
#   - 한 파일에서만 쓰는 값은 그 파일 위쪽에 있음
#   사용: import config

import glob
import os
from datetime import datetime

# 경로
ROOT = os.path.dirname(os.path.abspath(__file__))

MACHINE_DIR = os.path.join(ROOT, "01_데이터관리", "machines")    # 기기별 parquet (02 분석, 03 전처리)
PREP_DIR = os.path.join(ROOT, "03_전처리_모델학습", "outputs", "전처리")    # meta (전처리, 학습)
SPLIT_DIRS = {                                                              # 분할별 샘플 (전처리, 학습)
    "train": os.path.join(ROOT, "03_전처리_모델학습", "outputs", "Train"),
    "valid": os.path.join(ROOT, "03_전처리_모델학습", "outputs", "Valid"),
    "test": os.path.join(ROOT, "03_전처리_모델학습", "outputs", "Test"),
}
SIM_MODEL_PATH = os.path.join(ROOT, "04_시뮬레이션", "model", "model.pkl")   # 시뮬레이션 사용 모델 (학습, 서빙)
LOG_PATH = os.path.join(ROOT, "logs", "peak_demand.log")                     # write_log_print 용

# 데이터
MAIN_METER = "MainTerminal"     # 공장 전체 미터 (예측 대상)
LOCAL_TZ = "Europe/Berlin"      # 공장 현지 시간대

# 예측 구간 (1칸 = 10초 평균, 전처리·학습·서빙이 같아야 함)
STEP_SEC = 10       # 1칸 = 10초
WINDOW = 90         # 15분 = 90칸
FIRST = 12          # 예측 시작: 구간 2분 후
NORM = 1000.0       # 딥러닝 입력 정규화


# 파일명용 생성시간 (예: 20261122_170501)
def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# {name}_{생성시간}.{ext} 중 최신 파일 반환, 없으면 None
def latest_file(folder, name, ext):
    files = sorted(glob.glob(os.path.join(folder, name + "_*." + ext)))
    if len(files) == 0:
        return None
    return files[-1]


# 터미널 출력 + 로그 파일(logs/peak_demand.log) 기록
#   사용: write_log_print(f"완료: {경로}")
def write_log_print(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")
