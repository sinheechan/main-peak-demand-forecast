# 모델 튜닝·학습·평가 및 앙상블 선택
#   입력: 03_전처리_모델학습/outputs/전처리/ 최신 meta, 같은 생성시간의 Train/, Valid/, Test/ 데이터
#   출력: 03_전처리_모델학습/outputs/학습결과/{생성시간}/
#         학습결과_{생성시간}.xlsx, models/{모델}_{생성시간}.pkl, ensemble_{생성시간}.pkl
#   실행: python 03_전처리_모델학습/train.py
#         python 03_전처리_모델학습/train.py --deploy   (기존보다 좋으면 시뮬레이션 모델 교체)

import json
import os
import shutil
import sys
from itertools import combinations

import joblib
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from common.features import TAB_COLS
from common.models import PAPERS, SEQ_MODELS, make_model, predict_ensemble, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)
deploy = "--deploy" in sys.argv


# 평가 지표: MAE, 초반 MAE, 놓친 피크, 헛경보
#   - 놓친 피크: 목표 초과 구간인데 12분까지 경보 없음
#   - 헛경보: 실제가 목표의 90% 미만인데 경보
def evaluate(part, pred, target_kw):
    actual = part["actual_kw"].to_numpy()
    err = np.abs(pred - actual)
    early = part["elapsed_min"].to_numpy() <= EARLY_MIN

    frame = pd.DataFrame({
        "window_start": part["window_start"].to_numpy(),
        "elapsed_min": part["elapsed_min"].to_numpy(),
        "actual": actual,
        "pred": pred,
    })
    peaks = 0
    missed = 0
    false_alarms = 0
    for ws, group in frame.groupby("window_start"):
        window_actual = group["actual"].iloc[0]
        alarms = group[group["pred"] >= target_kw]
        if window_actual >= target_kw:
            peaks += 1
            if len(alarms[alarms["elapsed_min"] <= 12]) == 0:
                missed += 1
        elif window_actual < target_kw * 0.9 and len(alarms) > 0:
            false_alarms += 1

    return {
        "MAE": float(err.mean()),
        "초반 MAE": float(err[early].mean()),
        "피크 구간": peaks,
        "놓친 피크": missed,
        "헛경보": false_alarms,
    }


# 결과표 1행 생성
def make_row(kind, name, scores):
    row = {"구분": kind, "모델": name}
    for split in ["valid", "test"]:
        for key in scores[split]:
            row[split + " " + key] = scores[split][key]
    return row


# 1. 전처리 결과 로드 (최신 meta와 같은 생성시간의 Train, Valid, Test)
meta_path = latest_file(PREP_DIR, "meta", "json")
if meta_path is None:
    print("전처리 결과가 없음. 먼저 python 03_전처리_모델학습/preprocess.py")
    sys.exit()
with open(meta_path, encoding="utf-8") as f:
    meta = json.load(f)
prep_stamp = meta["stamp"]
target_kw = meta["target_kw"]

parts = {}   # 분할별 DataFrame
data = {}    # 분할별 (요약 특징, 시퀀스, 잔차)
for split in ["train", "valid", "test"]:
    folder = SPLIT_DIRS[split]
    parts[split] = pd.read_parquet(os.path.join(folder, split + "_" + prep_stamp + ".parquet"))
    data[split] = (
        parts[split][TAB_COLS].to_numpy(dtype="float32"),
        np.load(os.path.join(folder, split + "_seq_" + prep_stamp + ".npy")),
        parts[split]["residual"].to_numpy(dtype="float64"),
    )
    print(split, "로드:", len(parts[split]), "샘플")

run_id = stamp()
run_dir = os.path.join(TRAIN_DIR, run_id)
os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

write_log_print("=== 학습 시작 " + run_id + " (전처리 " + prep_stamp + ") ===")
write_log_print("목표전력 " + str(round(target_kw)) + " kW")

rows = []         # 결과표
preds = {}        # 모델별 valid, test 예측
history = []      # Optuna 시도 기록

# 2. 기준선 평가 (학습 없음)
for label, col in [("기준 계산식", "proj_base"), ("추세 계산식", "proj_trend")]:
    scores = {}
    preds[label] = {}
    for split in ["valid", "test"]:
        preds[label][split] = parts[split][col].to_numpy()
        scores[split] = evaluate(parts[split], preds[label][split], target_kw)
    rows.append(make_row("기준선", label, scores))
    write_log_print(label + " valid MAE " + str(round(scores["valid"]["MAE"], 1)) + " / test MAE " + str(round(scores["test"]["MAE"], 1)))

# 3. 모델별 튜닝 및 학습
X_tr, S_tr, y_tr = data["train"]
X_va, S_va, y_va = data["valid"]

# 튜닝용 샘플 (TUNE_ROWS개)
rng = np.random.default_rng(SEED)
pick = rng.choice(len(y_tr), min(TUNE_ROWS, len(y_tr)), replace=False)

models = {}
valid_mae = {}

for name in MODELS:
    n_trials = TRIALS
    if name in SEQ_MODELS:
        n_trials = max(3, TRIALS // 3)   # 딥러닝은 시도 횟수 1/3

    # Optuna: valid 잔차 MAE 최소화
    def objective(trial):
        model = make_model(name, suggest_params(name, trial))
        model.fit(X_tr[pick], S_tr[pick], y_tr[pick])
        pred = model.predict(X_va, S_va)
        return float(np.mean(np.abs(pred - y_va)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    pbar = tqdm(total=n_trials, desc=name)
    study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)])
    pbar.close()

    for trial in study.trials:
        history.append({"모델": name, "trial": trial.number, "valid 보정값 MAE": trial.value,
                        "파라미터": json.dumps(trial.params)})

    # 최적 파라미터로 train 전체 재학습
    best = study.best_params
    model = make_model(name, best)
    model.fit(X_tr, S_tr, y_tr)
    models[name] = model
    joblib.dump(model, os.path.join(run_dir, "models", name + "_" + run_id + ".pkl"))

    # 최종 예측 = 기준 계산식 + 예측 잔차
    scores = {}
    preds[name] = {}
    for split in ["valid", "test"]:
        X, S, y = data[split]
        preds[name][split] = parts[split]["proj_base"].to_numpy() + model.predict(X, S)
        scores[split] = evaluate(parts[split], preds[name][split], target_kw)
    valid_mae[name] = scores["valid"]["MAE"]

    row = make_row("단일", name, scores)
    row["파라미터"] = json.dumps(best)
    rows.append(row)
    write_log_print(name + " valid MAE " + str(round(scores["valid"]["MAE"], 1)) + " / test MAE " + str(round(scores["test"]["MAE"], 1)))

# 4. 앙상블 선택
#   - 2개 이상 모든 조합 x (단순 평균, 가중 평균) 중 valid MAE 최소
#   - 가중 평균 가중치 = valid MAE의 역수
ens_rows = []
best_ens = None
for n in range(2, len(MODELS) + 1):
    for combo in combinations(MODELS, n):
        for method in ["단순 평균", "가중 평균"]:
            weights = []
            for m in combo:
                if method == "단순 평균":
                    weights.append(1.0 / n)
                else:
                    weights.append(1.0 / valid_mae[m])
            total = sum(weights)
            weights = [w / total for w in weights]

            scores = {}
            for split in ["valid", "test"]:
                pred = np.zeros(len(parts[split]))
                for i in range(len(combo)):
                    pred = pred + weights[i] * preds[combo[i]][split]
                scores[split] = evaluate(parts[split], pred, target_kw)

            row = make_row("앙상블", "+".join(combo), scores)
            row["방법"] = method
            row["가중치"] = ", ".join([str(round(w, 2)) for w in weights])
            ens_rows.append(row)

            if best_ens is None or scores["valid"]["MAE"] < best_ens["valid MAE"]:
                best_ens = row
                best_ens["_members"] = list(combo)
                best_ens["_weights"] = weights

members = best_ens.pop("_members")
weights = best_ens.pop("_weights")
for row in ens_rows:
    row.pop("_members", None)
    row.pop("_weights", None)
write_log_print("앙상블 선택: " + best_ens["모델"] + " (" + best_ens["방법"] + ") valid MAE "
           + str(round(best_ens["valid MAE"], 1)) + " / test MAE " + str(round(best_ens["test MAE"], 1)))

# 서빙용 형식으로 저장
ens = {
    "run_id": run_id,
    "prep_stamp": prep_stamp,
    "members": members,
    "weights": weights,
    "models": {m: models[m] for m in members},
    "target_kw": target_kw,
    "scale": meta["scale"],
    "valid_mae": best_ens["valid MAE"],
    "test_mae": best_ens["test MAE"],
    "replay_from": meta["split"][REPLAY_SPLIT]["start"],
}
ens_path = os.path.join(run_dir, "ensemble_" + run_id + ".pkl")
joblib.dump(ens, ens_path)

# 5. 결과 엑셀 저장
# 경과 시간별 MAE
ens_pred = {}
for split in ["valid", "test"]:
    ens_pred[split] = np.zeros(len(parts[split]))
    for i in range(len(members)):
        ens_pred[split] = ens_pred[split] + weights[i] * preds[members[i]][split]
preds["앙상블"] = ens_pred

elapsed_rows = []
for split in ["valid", "test"]:
    part = parts[split]
    actual = part["actual_kw"].to_numpy()
    for minute in sorted(part["elapsed_min"].unique()):
        mask = (part["elapsed_min"] == minute).to_numpy()
        row = {"분할": split, "경과 (분)": minute}
        for label in preds:
            row[label] = float(np.mean(np.abs(preds[label][split][mask] - actual[mask])))
        elapsed_rows.append(row)

base_valid = rows[0]["valid MAE"]
base_test = rows[0]["test MAE"]
summary = pd.DataFrame([
    {"항목": "run_id", "값": run_id},
    {"항목": "전처리", "값": "meta_" + prep_stamp + ".json (Train / Valid / Test)"},
    {"항목": "목표전력 (kW)", "값": round(target_kw, 1)},
    {"항목": "선택 앙상블", "값": best_ens["모델"] + " (" + best_ens["방법"] + ")"},
    {"항목": "가중치", "값": best_ens["가중치"]},
    {"항목": "valid MAE 기준 계산식 -> 앙상블", "값": str(round(base_valid, 1)) + " -> " + str(round(best_ens["valid MAE"], 1))},
    {"항목": "valid 개선율", "값": str(round((1 - best_ens["valid MAE"] / base_valid) * 100, 1)) + "%"},
    {"항목": "test MAE 기준 계산식 -> 앙상블", "값": str(round(base_test, 1)) + " -> " + str(round(best_ens["test MAE"], 1))},
    {"항목": "test 개선율", "값": str(round((1 - best_ens["test MAE"] / base_test) * 100, 1)) + "%"},
])

ens_table = pd.DataFrame(ens_rows).sort_values("valid MAE")
excel_path = os.path.join(run_dir, "학습결과_" + run_id + ".xlsx")
with pd.ExcelWriter(excel_path) as writer:
    summary.to_excel(writer, sheet_name="0_요약", index=False)
    pd.DataFrame(rows + [best_ens]).to_excel(writer, sheet_name="1_전체결과", index=False)
    pd.DataFrame(elapsed_rows).to_excel(writer, sheet_name="2_경과시간별MAE", index=False)
    pd.DataFrame(history).to_excel(writer, sheet_name="3_Optuna기록", index=False)
    pd.DataFrame(list(PAPERS.items()), columns=["모델", "논문"]).to_excel(writer, sheet_name="4_모델근거", index=False)
    ens_table.to_excel(writer, sheet_name="5_앙상블전체", index=False)
write_log_print("저장: " + excel_path)

# 6. 시뮬레이션 모델 교체 (--deploy)
if deploy:
    # 기존 모델과 valid MAE 비교 후 개선 시에만 교체 (기존 모델은 백업)
    old_mae = None
    if os.path.exists(SIM_MODEL_PATH):
        try:
            old = joblib.load(SIM_MODEL_PATH)
            old_pred = predict_ensemble(old, parts["valid"], data["valid"][1])
            old_mae = float(np.mean(np.abs(old_pred - parts["valid"]["actual_kw"].to_numpy())))
        except Exception as e:
            print("기존 모델 비교 실패 -> 새 모델로 교체:", e)

    if old_mae is not None and ens["valid_mae"] > old_mae:
        write_log_print("배포 안 함: 새 모델 valid " + str(round(ens["valid_mae"], 1)) + " > 기존 " + str(round(old_mae, 1)))
    else:
        os.makedirs(os.path.dirname(SIM_MODEL_PATH), exist_ok=True)
        if os.path.exists(SIM_MODEL_PATH):
            backup = SIM_MODEL_PATH.replace("model.pkl", "model_backup_" + stamp() + ".pkl")
            shutil.copy(SIM_MODEL_PATH, backup)
        shutil.copy(ens_path, SIM_MODEL_PATH)
        write_log_print("배포 완료 -> " + SIM_MODEL_PATH)
else:
    print("시뮬레이션에 쓰려면 --deploy 붙여서 실행")
