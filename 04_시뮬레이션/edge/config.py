# 엣지 공통 설정 (simulator, controller, scenario 공용)
#   - 비밀번호는 .env, 나머지는 이 파일에서 관리
#   - 재생 데이터: 01_데이터관리/machines/MainTerminal/ 최신 parquet
#   - 재생 시작일: 03 전처리 meta의 REPLAY_SPLIT 시작일 (학습에 안 쓴 기간)

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

EDGE_DIR = Path(__file__).resolve().parent
ROOT = EDGE_DIR.parents[1]

load_dotenv(EDGE_DIR / ".env")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")

MODBUS_HOST = "localhost"
MODBUS_PORT = 5020
MQTT_HOST = "localhost"
MQTT_PORT = 1883
SITE = "plant1"

TARGET_KW = 1600   # 목표 최대수요전력 (15분 평균)
ESS_KW = 200       # ESS 방전 출력

# 차단 가능 부하 (코일 주소 0, 1, 2 = load1, load2, load3)
LOAD_KW = [150, 120, 100]
SHED_ORDER = [2, 1, 0]   # 차단 순서 (중요도 낮은 load3부터)

# 레지스터 주소표 (Modbus는 정수만 저장 -> 배율 적용, 예: 380.5V -> 3805)
#   0: 전압 x10      1: 전류 x10      2: kW x10       3: 역률 x1000    4: Hz x100
#   5, 6: 누적 kWh x10 (32비트 -> 상위/하위 16비트)
#   10: ESS 충전율 x10    11: ESS 방전 kW x10
#   20: ESS 방전 지령 kW x10 (controller 기록)
#   30: 시나리오 번호 (scenario.py 기록)

# 재생 데이터 (HIPE 공장 전체)
MAIN_METER = "MainTerminal"
LOCAL_TZ = "Europe/Berlin"         # 공장 현지 시간대
P95_KW = 1400                      # 규모 기준 (03 전처리와 동일)
REPLAY_SPLIT = "valid"             # 재생 기간 (test는 연휴라 피크가 거의 없음)

def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


# {name}_{생성시간}.{ext} 중 최신 파일 반환, 없으면 None
def latest(folder, name, ext):
    files = sorted(glob.glob(os.path.join(str(folder), name + "_*." + ext)))
    if len(files) == 0:
        return None
    return files[-1]


def replay_path():
    return latest(ROOT / "01_데이터관리" / "machines" / MAIN_METER, MAIN_METER, "parquet")


def replay_from():
    # 재생 시작일 (meta 없으면 None -> 전체 재생)
    meta_path = latest(ROOT / "03_전처리_모델학습" / "outputs" / "전처리", "meta", "json")
    if meta_path is None:
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("split", {}).get(REPLAY_SPLIT, {}).get("start")
