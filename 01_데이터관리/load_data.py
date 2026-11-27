# HIPE 원본 zip을 기기별 파일로 나눠 저장
#   입력: 01_데이터관리/raw/hipe_cleaned_*.zip
#   출력: 01_데이터관리/machines/{기기}/{기기}_{생성시간}.parquet, .csv
#   실행: python 01_데이터관리/load_data.py

import glob
import os
import sys
import zipfile

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 1. 원본 zip 찾기
zip_list = glob.glob(os.path.join(config.RAW_DIR, "hipe_cleaned_*.zip"))
if len(zip_list) == 0:
    print("zip 파일이 없음:", config.RAW_DIR)
    sys.exit()
zip_path = zip_list[0]

zip_file = zipfile.ZipFile(zip_path)
run_stamp = config.stamp()
config.write_log_print("zip: " + os.path.basename(zip_path))

info = []  # manifest 기록용

# 2. 기기별로 읽어서 저장

zip_list = zip_file.namelist()
for file_name in tqdm(zip_list()):
    if not file_name.endswith(".csv"):
        continue
    machine = file_name.split("_")[0]  # 예: ChipPress_PhaseCount_3_... -> ChipPress

    # 필요한 컬럼만 읽기 (2상 기기는 일부 컬럼이 없음)
    header = pd.read_csv(zip_file.open(file_name), nrows=0).columns
    cols = []
    for col in config.KEEP_COLS:
        if col in header:
            cols.append(col)

    df = pd.read_csv(zip_file.open(file_name), usecols=cols)

    # 시각 정리
    df["tz_offset"] = df["SensorDateTime"].str[-3:]
    df["ts_utc"] = pd.to_datetime(df["SensorDateTime"], format="ISO8601", utc=True)
    df = df.drop(columns=["SensorDateTime"])

    # 정렬, 중복 시각 제거
    df_rows = len(df)

    df = df.sort_values("ts_utc")
    df = df.drop_duplicates("ts_utc").reset_index(drop=True)

    # 컬럼 순서: 시각 먼저
    other_cols = []
    for col in df.columns:
        if col != "ts_utc" and col != "tz_offset":
            other_cols.append(col)
    df = df[["ts_utc", "tz_offset"] + other_cols]

    # 저장 (parquet은 다음 단계 입력, csv는 확인용)
    machine_path = os.path.join(config.MACHINE_DIR, machine)
    os.makedirs(machine_path, exist_ok=True)

    parquet_path = os.path.join(machine_path, machine + "_" + run_stamp + ".parquet")
    csv_path = os.path.join(machine_path, machine + "_" + run_stamp + ".csv")
    
    if os.path.exists(parquet_path):
        print("파일이 이미 있습니다 :", parquet_path)
    else:
        df.to_parquet(parquet_path, index=False)

    if os.path.exists(csv_path):
        print("파일이 이미 있습니다 :", csv_path)
    else:
        df.to_csv(csv_path, index=False)

    info.append({
        "machine": machine,
        "source_file": file_name,
        "df_rows": df_rows,
        "duplicate_rows": df_rows - len(df),
        "saved_rows": len(df),
        "columns": ", ".join(other_cols),
        "start_utc": df["ts_utc"].min(),
        "end_utc": df["ts_utc"].max(),
        "parquet": parquet_path,
        "csv": csv_path,
    })
    config.write_log_print(machine + " " + str(df.shape))

# 3. 기기별 요약(manifest) 저장
manifest = pd.DataFrame(info)
manifest_path = os.path.join(config.MACHINE_DIR, "manifest_" + run_stamp + ".csv")
manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
config.write_log_print("완료: 기기 " + str(len(info)) + "개 -> " + config.MACHINE_DIR)
