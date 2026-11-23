# 엣지 제어기: 계측값 읽기 -> 15분 수요 계산 -> ESS/부하 제어 -> MQTT 전송
#   - MQTT 끊김 시 buffer.db에 저장 후 재연결 시 재전송

import json
import sqlite3
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

from config import (ESS_KW, MODBUS_HOST, MODBUS_PORT, MQTT_HOST, MQTT_PASS, MQTT_PORT, MQTT_USER,
                    SHED_ORDER, SITE, model_info, now_kst)
from demand import Demand

TARGET_KW = model_info()["target_kw"]   # 목표 최대수요전력 (15분 평균, 학습과 같은 값)
TOPIC = f"site/{SITE}"
WARN = 0.9      # 목표의 90% 이상 -> ESS 방전
SAFE = 0.8      # 목표의 80% 미만 -> 복귀
SOC_MIN = 20    # ESS 최소 충전율 (%)
HOLD_MIN = 2    # 구간 초반 판단 보류 (분, 예상치 불안정)
COOLDOWN = 60   # 제어 간 최소 간격 (초)

# 미전송 메시지 버퍼 (재시작 후에도 유지)
db = sqlite3.connect(str(Path(__file__).resolve().parent / "buffer.db"), check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS buf (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, payload TEXT)")
db.commit()

# 제어 상태
shed = []            # 차단한 부하 번호 (차단 순서)
ess_on = False
last_action = None   # 마지막 제어 시각


def read_meter(mb):
    r = mb.read_holding_registers(0, 12, slave=1)
    if r.isError():
        raise IOError(r)

    h = r.registers
    return {
        "voltage": h[0] / 10,
        "current": h[1] / 10,
        "kw": h[2] / 10,
        "pf": h[3] / 1000,
        "hz": h[4] / 100,
        "kwh": ((h[5] << 16) | h[6]) / 10,
        "soc": h[10] / 10,
        "ess_kw": h[11] / 10,
    }


# 예상 수요 기준 제어 (1회 1동작, 제어 시 기록용 dict 반환)
#   목표 이상 + ESS 방전 중(또는 충전율 부족) -> 부하 차단
#   목표 90% 이상                             -> ESS 방전
#   목표 80% 미만                             -> 부하 복귀 후 ESS 정지
def control(mb, t, dem, soc):
    global ess_on, last_action

    if dem.elapsed_min < HOLD_MIN:
        return None
    if last_action is not None and (t - last_action).total_seconds() < COOLDOWN:
        return None

    p = dem.projected
    if p >= TARGET_KW and len(shed) < 3 and (ess_on or soc <= SOC_MIN):
        n = SHED_ORDER[len(shed)]
        mb.write_coil(n, False, slave=1)
        shed.append(n)
        action, target, reason = "shed", f"load{n + 1}", "예상 수요 ≥ 목표"
    elif p >= TARGET_KW * WARN and not ess_on and soc > SOC_MIN:
        mb.write_register(20, ESS_KW * 10, slave=1)
        ess_on = True
        action, target, reason = "ess_on", "ess", "예상 수요 ≥ 목표 90%"
    elif p < TARGET_KW * SAFE and shed:
        n = shed.pop()
        mb.write_coil(n, True, slave=1)
        action, target, reason = "restore", f"load{n + 1}", "예상 수요 < 목표 80%"
    elif ess_on and (p < TARGET_KW * SAFE or soc <= SOC_MIN):
        mb.write_register(20, 0, slave=1)
        ess_on = False
        action, target = "ess_off", "ess"
        reason = "여유 확보" if soc > SOC_MIN else "충전율 부족"
    else:
        return None

    last_action = t
    return {"ts": t, "action": action, "target": target, "reason": reason, "projected_kw": round(p, 1)}


def on_connect(client, userdata, flags, reason_code, properties):
    print("MQTT 연결:", reason_code)
    if not reason_code.is_failure:
        client.publish(f"{TOPIC}/status", json.dumps({"status": "online"}), qos=1, retain=True)


# MQTT 전송, 실패 시 buffer.db 저장 (default=str: 시각 -> 문자열)
def send(cli, sub, payload):
    topic = f"{TOPIC}/{sub}"
    if cli.is_connected():
        r = cli.publish(topic, json.dumps(payload, default=str), qos=1)
        if r.rc == mqtt.MQTT_ERR_SUCCESS:
            return

    payload["buffered"] = True
    db.execute("INSERT INTO buf (topic, payload) VALUES (?, ?)", (topic, json.dumps(payload, default=str)))
    db.commit()


# 버퍼 재전송 (오래된 순 200개씩, 전송 성공분만 삭제)
def flush(cli):
    if not cli.is_connected():
        return

    rows = db.execute("SELECT id, topic, payload FROM buf ORDER BY id LIMIT 200").fetchall()
    for i, topic, payload in rows:
        r = cli.publish(topic, payload, qos=1)
        if r.rc != mqtt.MQTT_ERR_SUCCESS:
            break
        db.execute("DELETE FROM buf WHERE id = ?", (i,))
    db.commit()


def main():
    mb = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-{SITE}")
    cli.username_pw_set(MQTT_USER, MQTT_PASS)
    # 비정상 종료 시 브로커가 offline 상태 전송 (LWT)
    cli.will_set(f"{TOPIC}/status", json.dumps({"status": "offline"}), qos=1, retain=True)
    cli.on_connect = on_connect
    cli.reconnect_delay_set(1, 10)
    cli.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)
    cli.loop_start()

    dem = Demand(TARGET_KW)
    seq = 0
    ready = False

    while True:
        t0 = time.time()
        t = now_kst()

        try:
            if not mb.connected:
                mb.connect()
            if not ready:
                # 시작 시 전체 부하 복귀 (이전 실행의 차단 상태 해제)
                for n in range(3):
                    mb.write_coil(n, True, slave=1)
                mb.write_register(20, 0, slave=1)
                ready = True
            meter = read_meter(mb)
        except Exception as e:
            print("Modbus 읽기 실패:", e)
            time.sleep(2)
            continue

        seq += 1
        data = {"ts": t, "seq": seq}
        data.update(meter)
        send(cli, "meter/data", data)

        closed = dem.update(t, meter["kwh"], meter["kw"])
        if closed:
            send(cli, "demand", closed)
        if seq % 5 == 0:
            send(cli, "demand", dem.summary())

        try:
            event = control(mb, t, dem, meter["soc"])
        except Exception as e:
            print("제어 명령 실패:", e)
            event = None
        if event:
            print("[제어]", event)
            send(cli, "control", event)

        flush(cli)

        if seq % 10 == 0:
            n_buf = db.execute("SELECT count(*) FROM buf").fetchone()[0]
            state = "연결" if cli.is_connected() else "끊김"
            print(f"{t:%H:%M:%S} | {meter['kw']:,.0f} kW | 15분 예상 {dem.projected:,.0f}/{TARGET_KW:,.0f} kW "
                  f"| ESS {meter['soc']:.0f}% | 버퍼 {n_buf} | MQTT {state}")

        time.sleep(max(0.0, 1 - (time.time() - t0)))


if __name__ == "__main__":
    main()
