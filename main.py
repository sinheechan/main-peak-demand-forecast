# 전체 실행: 01 데이터 → 02 분석 → 03 전처리 → 03 학습(--deploy)
#   실행: python main.py
#   - 단계마다 시작 전에 입력 파일을 확인하고, 없으면 무엇이 없는지와 먼저 할 단계를 알려 주고 중단
#   - 각 단계는 따로 실행한 것과 똑같이 별도 프로세스로 돌림 (출력 폴더에 파일이 있으면 그 단계에서 중단)

import glob
import json
import os
import subprocess
import sys

import config

logger = config.log_write_print("main")


# 1단계 입력: 원본 zip
def check_load_data():
    if len(glob.glob(os.path.join(config.RAW_DIR, "hipe_cleaned_*.zip"))) == 0:
        return [f"원본 zip: {config.RAW_DIR}/hipe_cleaned_*.zip"], "README '데이터 준비'를 보고 zip을 data/raw/ 에 넣기"
    return [], ""


# 2단계 입력: 기기별 parquet
def check_analyze():
    found = glob.glob(os.path.join(config.MACHINE_DIR, "*", "*.parquet"))
    if len(found) == 0:
        return [f"기기별 parquet: {config.MACHINE_DIR}/{{기기}}/*.parquet"], "1단계 load_data.py"
    return [], ""


# 3단계 전처리 입력: 공장 전체(MainTerminal) parquet
def check_preprocess():
    folder = os.path.join(config.MACHINE_DIR, config.MAIN_METER)
    if config.latest_file(folder, config.MAIN_METER, "parquet") is None:
        return [f"{config.MAIN_METER} parquet: {folder}/{config.MAIN_METER}_*.parquet"], "1단계 load_data.py"
    return [], ""


# 3단계 학습 입력: 최신 meta와 같은 생성시간의 분할별 parquet, npy
def check_train():
    meta_path = config.latest_file(config.PREP_DIR, "meta", "json")
    if meta_path is None:
        return [f"전처리 meta: {config.PREP_DIR}/meta_*.json"], "3-1단계 preprocess.py"

    with open(meta_path, encoding="utf-8") as f:
        prep_stamp = json.load(f)["stamp"]

    missing = []
    for split in ["train", "valid", "test"]:
        folder = config.SPLIT_DIRS[split]
        for name in [f"{split}_{prep_stamp}.parquet", f"{split}_seq_{prep_stamp}.npy"]:
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                missing.append(path)
    return missing, "3-1단계 preprocess.py"


STAGES = [
    {"name": "1. 데이터 정리", "script": "load_data.py", "args": [], "check": check_load_data},
    {"name": "2. 데이터 분석", "script": "analyze_data.py", "args": [], "check": check_analyze},
    {"name": "3-1. 전처리", "script": "preprocess.py", "args": [], "check": check_preprocess},
    {"name": "3-2. 학습", "script": "train.py", "args": ["--deploy"], "check": check_train},
]


# 단계 1개: 입력 확인 -> 실행 -> 실패하면 중단
def run_stage(stage):
    logger.info(f"===== {stage['name']} ({stage['script']}) =====")

    missing, before = stage["check"]()
    if len(missing) > 0:
        logger.error(f"[{stage['name']}] 입력 파일이 없어서 시작하지 않습니다.")
        for item in missing:
            logger.error(f"  없음: {item}")
        logger.error(f"  먼저 할 것: {before}")
        sys.exit(1)

    command = [sys.executable, os.path.join(config.ROOT, stage["script"])] + stage["args"]
    result = subprocess.run(command, cwd=config.ROOT)
    if result.returncode != 0:
        logger.error(f"[{stage['name']}] 실패 (종료 코드 {result.returncode}) -> 이후 단계는 실행하지 않습니다.")
        sys.exit(result.returncode)


def main():
    for stage in STAGES:
        run_stage(stage)
    logger.info("전체 완료")


if __name__ == "__main__":
    main()
