# 전체 실행: 1 데이터 정리 → 2 데이터 분석 → 3 전처리 → 4 학습(--deploy)
#   실행: python main.py
#   - 각 단계는 따로 실행한 것과 똑같이 별도 프로세스로 돌림
#   - 입력 파일이 없거나 출력 폴더에 파일이 있으면 그 단계가 알려 주고 멈추고, 이후 단계는 실행 안 함

import os
import subprocess
import sys

import config

STEPS = [
    ("1. 데이터 정리", ["1_load_data.py"]),
    ("2. 데이터 분석", ["2_analyze_data.py"]),
    ("3. 전처리", ["3_preprocess.py"]),
    ("4. 학습", ["4_train.py", "--deploy"]),
]

logger = config.write_log_print("main")


def main():
    for name, command in STEPS:
        logger.info(f"===== {name} ({command[0]}) =====")
        script = os.path.join(config.ROOT, command[0])
        result = subprocess.run([sys.executable, script] + command[1:], cwd=config.ROOT)
        if result.returncode != 0:
            logger.error(f"[{name}] 실패 -> 이후 단계는 실행하지 않습니다.")
            sys.exit(result.returncode)
    logger.info("전체 완료")


if __name__ == "__main__":
    main()
