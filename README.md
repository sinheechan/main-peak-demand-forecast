# peak-demand-forecast

공장 전체 전력(HIPE, 독일 전자부품 공장)의 **15분 평균 전력(최대수요전력)** 을 구간이 끝나기 전에 예측하고,
가상 계측기 · 제어기 · 서버로 실시간 운영을 시뮬레이션하는 프로젝트.

## 폴더 구성

```
Edge-PDP/
├── config.py                  모든 경로 · 숫자 설정 · 최신 파일 찾기 · 로그 (여기만 고치면 전체 반영)
├── common/                    여러 단계가 같이 쓰는 코드
│   ├── features.py            10초 평균 · 15분 구간 특징 (학습과 서빙이 같은 함수 사용)
│   └── models.py              모델 6종 · Optuna 탐색 범위 · 논문 근거
├── 01_데이터관리/
│   ├── load_data.py           HIPE zip → 기기별 폴더에 parquet + csv 저장
│   ├── raw/                   ← HIPE zip 을 여기에 (아래 '데이터 준비' 링크)
│   └── machines/<기기명>/<기기명>_<생성시간>.parquet (.csv)
├── 02_데이터분석/
│   ├── analyze_data.py        기기별 요약 · 통계 · 이상치 · 누락 · 패턴 → 엑셀 1개
│   ├── reports/데이터분석_<생성시간>.xlsx
│   └── notebooks/             탐색용 노트북
├── 03_전처리_모델학습/
│   ├── preprocess.py          특징 생성 · 날짜 기준 train/valid/test 분할
│   ├── train.py               모델 6개를 하나씩 Optuna 튜닝 · 학습 → valid/test 결과 · 앙상블
│   └── outputs/
│       ├── Train/ · Valid/ · Test/  <분할>_<생성시간>.parquet, <분할>_seq_<생성시간>.npy
│       ├── 전처리/meta_<생성시간>.json
│       └── 학습결과/<run_id>/학습결과_<run_id>.xlsx, models/, ensemble_<run_id>.pkl
├── 04_시뮬레이션/
│   ├── run.sh                 서버 → 계측기 → 제어기 → ML 서빙 한 번에 실행
│   ├── model/model.pkl        서빙이 쓰는 모델 (train.py --deploy 로 들어옴)
│   ├── edge/                  가상 계측기(Modbus) · HIPE 재생 · 15분 수요 제어 · MQTT 전송
│   ├── serving/serve.py       ML 그림자 운영 (예측 기록 · 구간 평가)
│   └── server/                docker-compose (MQTT 브로커 · PostgreSQL · 수신기 · 대시보드)
├── logs/peak_demand.log       모든 단계 실행 로그
└── requirements.txt
```

**파일명 규칙**: `<이름>_<생성시간>.<확장자>` (예: `MainTerminal_20261122_170616.parquet`).
다음 단계는 같은 이름 중 생성시간이 가장 늦은 파일을 자동으로 읽는다 (`config.latest_file`).

## 데이터 준비

원본 데이터(약 1.3 GB)는 저장소에 포함하지 않는다. 아래에서 받아 `01_데이터관리/raw/` 에 넣는다 (압축은 풀지 않음).

- 데이터셋: HIPE – High-resolution Industrial Production Energy (KIT) — https://www.energystatusdata.kit.edu/hipe.php
- 사용 파일 (3개월, 2017-10-01 ~ 2017-12-31): [hipe_cleaned_v1.0.1_geq_2017-10-01_lt_2018-01-01.zip](https://bwsyncandshare.kit.edu/s/JacGftzBH85dgGK/download?path=%2F&files=hipe_cleaned_v1.0.1_geq_2017-10-01_lt_2018-01-01.zip)
- 같은 페이지의 1주 파일(2017-10-23 ~ 10-30)은 위 기간에 포함된 샘플이라 사용하지 않는다

## 실행 순서

```bash
source .venv/bin/activate
python 01_데이터관리/load_data.py                 # 1. zip → 기기별 parquet
python 02_데이터분석/analyze_data.py              # 2. 분석 엑셀
python 03_전처리_모델학습/preprocess.py           # 3-1. 특징 · 분할
python 03_전처리_모델학습/train.py --deploy       # 3-2. 학습 · 평가 · 시뮬레이션 모델 반영
./04_시뮬레이션/run.sh                            # 4. 시뮬레이션 (stop 으로 종료)
```

빠르게 확인할 때: `config.py` 의 `MODELS`, `TRIALS` 를 줄여서 실행

## 단계별 산출물

| 단계 | 산출물 | 내용 |
|---|---|---|
| 1 | `machines/<기기>/…parquet`, `….csv`, `manifest_….csv` | 필요한 컬럼만, 시각은 UTC, 중복 시각 제거 |
| 2 | `데이터분석_….xlsx` | 요약 · 기술통계 · 이상치 · 누락구간 · 일별수집률 · 시간대패턴 · 장비관계 |
| 3 | `학습결과_….xlsx` | 요약(개선율) · 전체결과(valid/test) · 경과시간별 MAE · Optuna 기록 · 모델 근거 · 앙상블 전체 |
| 4 | DB `ml.forecast`, `ml.window_eval` | 같은 구간에서 기준 계산식 vs ML 오차 |

## 모델 (common/models.py)

학습 타깃은 **잔차 = 실제 15분 평균 − 기준 계산식**. 최종 예측 = 기준 계산식 + 예측 잔차.

| 모델 | 근거 논문 |
|---|---|
| Ridge | Hoerl & Kennard (1970), Technometrics |
| Random Forest | Breiman (2001), Machine Learning |
| XGBoost | Chen & Guestrin (2016), KDD |
| LightGBM | Ke et al. (2017), NeurIPS |
| GRU | Cho et al. (2014), EMNLP |
| 1D-CNN | Bai, Kolter & Koltun (2018), arXiv:1803.01271 |
| Optuna (TPE) | Akiba et al. (2019), KDD · Bergstra et al. (2011), NeurIPS |
| 가중 평균 앙상블 | Bates & Granger (1969), Operational Research Quarterly |

## 참고

- 분할은 현지 날짜 기준 70 / 15 / 15. HIPE 시험 기간(12월 말)은 크리스마스 연휴라 피크가 거의 없어,
  피크 지표는 valid 를 함께 본다. 시뮬레이션도 valid 기간부터 재생한다 (`config.REPLAY_SPLIT`).
- 비밀번호는 각 폴더의 `.env` (예시는 `.env.example`).
