# 01 ~ 03 단계를 순서대로 실행 (그다음 ./04_시뮬레이션/run.sh)
#   실행: python run_all.py
#   중간 단계부터: python run_all.py 3   (3번째 단계부터)

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ["01_데이터관리/load_data.py"],                  # 1. zip -> 기기별 parquet
    ["02_데이터분석/analyze_data.py"],               # 2. 분석 엑셀
    ["03_전처리_모델학습/preprocess.py"],            # 3. 특징 · 분할
    ["03_전처리_모델학습/train.py", "--deploy"],     # 4. 학습 · 평가 · 시뮬레이션 모델 반영
]

start = int(sys.argv[1]) if len(sys.argv) > 1 else 1

for i, step in enumerate(STEPS, start=1):
    if i < start:
        continue
    print(f"\n===== [{i}/{len(STEPS)}] {' '.join(step)} =====")
    result = subprocess.run([sys.executable, os.path.join(ROOT, step[0])] + step[1:], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[{i}] {step[0]} 실패 (종료 코드 {result.returncode}). 수정 후: python run_all.py {i}")
        sys.exit(result.returncode)

print("\n완료. 시뮬레이션: ./04_시뮬레이션/run.sh")
