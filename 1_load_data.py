# HIPE 원본 zip을 기기별 파일로 나눠 저장
#   입력: data/raw/hipe_cleaned_*.zip
#   출력: data/machines/{기기}/{기기}_{생성시간}.parquet, .csv
#   실행: python 1_load_data.py

import glob
import os
import sys
import zipfile

import pandas as pd
from tqdm import tqdm

import config

# 저장할 컬럼 (원본 362개 중 일부, 파일에 없는 컬럼은 건너뜀) / (outputs/reports/컬럼설명.md)
KEEP_COLS = ["SensorDateTime", "P_kW", "Q_kvar", "S_kVA",
             "I1_A", "I2_A", "I3_A", "V1_V", "V2_V", "V3_V", "F_Hz", "L_F"]

logger = config.write_log_print("load_data")


# read_csv에서 KEEP_COLS에 있는 컬럼만 읽게 하는 조건
def is_keep_column(col):
    return col in KEEP_COLS


def main():
    # 출력 폴더에 이미 파일이 있으면 중단 (덮어쓰지 않음)
    config.stop_if_output_exists([config.MACHINE_DIR], logger)

    # 원본 zip
    zip_list = glob.glob(os.path.join(config.RAW_DIR, "hipe_cleaned_*.zip"))
    if len(zip_list) == 0:
        logger.error(f"zip 파일이 없음: {config.RAW_DIR}")
        sys.exit(1)
    zip_path = zip_list[0]

    zfile = zipfile.ZipFile(zip_path)
    timenow = config.stamp()
    logger.info(f"zip: {os.path.basename(zip_path)}")

    # zip파일 파일명 목록
    file_list = []
    for name in zfile.namelist():
        if name.endswith(".csv"):
            file_list.append(name)
    logger.info(f"csv 파일 {len(file_list)}개")

    # 기기별 폴더 생성 후 저장
    for file_name in tqdm(file_list, desc="파일 저장중..."):
        machine = file_name.split("_")[0]

        df = pd.read_csv(zfile.open(file_name), usecols=is_keep_column)

        # 시각 정리: UTC로 저장하고 원래 시간대는 tmp_tzoffset에 남겨둠 (시각 컬럼을 맨 앞으로)
        time_text = df.pop("SensorDateTime")
        df.insert(0, "tmp_tzoffset", time_text.str[-3:])
        df.insert(0, "tmp_timestamp", pd.to_datetime(time_text, format="ISO8601", utc=True))

        # 정렬, 중복 시각 제거
        df = df.sort_values("tmp_timestamp").drop_duplicates("tmp_timestamp")

        # 저장 parquet, csv(검토용) / 파일명: {machine}_{timenow}.확장자
        machine_path = os.path.join(config.MACHINE_DIR, machine)
        os.makedirs(machine_path, exist_ok=True)
        parquet_path = os.path.join(machine_path, f"{machine}_{timenow}.parquet")
        csv_path = os.path.join(machine_path, f"{machine}_{timenow}.csv")

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"{machine} {df.shape}")

    logger.info(f"완료: {config.MACHINE_DIR}")


if __name__ == "__main__":
    main()
