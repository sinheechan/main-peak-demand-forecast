# HIPE 원본 zip을 기기별 파일로 나눠 저장
#   입력: 01_데이터관리/raw/hipe_cleaned_*.zip
#   출력: 01_데이터관리/machines/{기기}/{기기}_{생성시간}.parquet, .csv
#   실행: python 01_데이터관리/load_data.py

import glob
import os
import sys
import zipfile
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# 경로 (이 파일이 있는 01_데이터관리 폴더 기준)
BASE_DIR = r"/home/gmlck/projects/Edge-PDP/01_데이터관리"
RAW_DIR = os.path.join(BASE_DIR, "raw")             # input
MACHINE_DIR = os.path.join(BASE_DIR, "machines")    # output

# 저장할 컬럼 (원본 362개 중 일부, 2상 기기는 없는 컬럼 제외) / (컬럼설명.md)
KEEP_COLS = ["SensorDateTime", 
             "P_kW", 
             "Q_kvar", 
             "S_kVA", 
             "I1_A", "I2_A", "I3_A", 
             "V1_V", "V2_V", "V3_V", 
             "F_Hz", "L_F"]

# 원본 zip
zip_list = glob.glob(os.path.join(RAW_DIR, "hipe_cleaned_*.zip"))
if len(zip_list) == 0:
    print(f"zip 파일이 없음: {RAW_DIR}")
    sys.exit()
zip_path = zip_list[0]

zfile = zipfile.ZipFile(zip_path)
timenow = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"zip: {os.path.basename(zip_path)}")

# zip파일 파일명 목록
file_list = []
for name in zfile.namelist():
    if name.endswith(".csv"):
        file_list.append(name)
print(f"csv 파일 {len(file_list)}개")

# 기기별 폴더 생성 후 저장
for file_name in tqdm(file_list, desc = "파일 저장중..."):
    machine = file_name.split("_")[0]

    # KEEP_COLS에 있는 컬럼만 읽기 (컬럼설명.md, 파일에 없는 컬럼은 건너뜀)
    df = pd.read_csv(zfile.open(file_name), usecols=lambda col: col in KEEP_COLS)
    cols = [col for col in KEEP_COLS if col in df.columns]

    # 시각 정리
    df["tmp_tzoffset"] = df["SensorDateTime"].str[-3:]
    df["tmp_timestamp"] = pd.to_datetime(df["SensorDateTime"], format="ISO8601", utc=True)
    df = df.drop(columns=["SensorDateTime"])

    df = df.sort_values("tmp_timestamp").drop_duplicates("tmp_timestamp")
    df = df[["tmp_timestamp", "tmp_tzoffset"] + cols[1:]]

    # 저장 parquet, csv(검토용)
    machine_path = os.path.join(MACHINE_DIR, machine)
    os.makedirs(machine_path, exist_ok=True)

    # 파일명 규칙 : {machine}_{timenow}.확장자
    parquet_path = os.path.join(machine_path, f"{machine}_{timenow}.parquet")
    csv_path = os.path.join(machine_path, f"{machine}_{timenow}.csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"완료: {MACHINE_DIR}")
