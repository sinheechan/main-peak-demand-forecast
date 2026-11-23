# 엣지 공통 설정 (simulator, controller, scenario 공용)
#   - 비밀번호는 .env, 나머지는 이 파일에서 관리
#   - 목표전력·배율·재생 파일·재생 시작일: main.py(train --deploy)가 만든 model/model_info.json
#     (학습에 쓴 전처리 결과와 같은 값으로 시뮬레이션)

import json
import os
import sys
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
REPLAY_DIR = ROOT / "data" / "machines" / "MainTerminal"
LOCAL_TZ = "Europe/Berlin"         # 공장 현지 시간대
MODEL_INFO_PATH = EDGE_DIR.parent / "model" / "model_info.json"


def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


# main.py 결과 (target_kw: 목표 최대수요전력, scale: 전처리 배율, source: 전처리에 쓴 parquet, replay_from: 재생 시작일)
def model_info():
    if not MODEL_INFO_PATH.exists():
        print(f"{MODEL_INFO_PATH} 이 없습니다. 먼저: python main.py")
        sys.exit(1)
    with open(MODEL_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)
