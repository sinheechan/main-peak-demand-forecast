#!/bin/bash
# 시뮬레이션 전체 실행: 서버(docker) → 가상 계측기 → 제어기 → ML 서빙 (tmux 한 화면)
#   ./04_시뮬레이션/run.sh        시작 (이미 켜져 있으면 화면만 다시 붙음)
#   ./04_시뮬레이션/run.sh stop   전체 종료
SIM="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SIM/.." && pwd)"
VENV="source $ROOT/.venv/bin/activate"
WAIT_MODBUS='until (echo > /dev/tcp/localhost/5020) 2>/dev/null; do sleep 1; done'

if [ "$1" = "stop" ]; then
    tmux kill-session -t power 2>/dev/null
    cd "$SIM/server" && docker compose down
    exit
fi

tmux has-session -t power 2>/dev/null && exec tmux attach -t power

if [ ! -f "$SIM/model/model.pkl" ]; then
    echo "model/model.pkl 이 없습니다. 먼저: python 03_전처리_모델학습/train.py --deploy"
fi

if ! docker info >/dev/null 2>&1; then
    "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe" >/dev/null 2>&1 &
    echo "도커 켜는 중..."
    until docker info >/dev/null 2>&1; do sleep 3; done
fi
cd "$SIM/server" && docker compose up -d

tmux new-session -d -s power -c "$SIM/edge" "$VENV && python simulator.py; bash"
tmux split-window -h -t power -c "$SIM/edge" "echo 'simulator 준비 대기 중...'; $WAIT_MODBUS; sleep 3; $VENV && python controller.py; bash"
tmux split-window -v -t power:0.0 -c "$SIM/serving" "echo 'controller 대기 중...'; $WAIT_MODBUS; sleep 10; $VENV && python serve.py; bash"
tmux split-window -v -t power:0.1 -c "$ROOT"
tmux attach -t power
