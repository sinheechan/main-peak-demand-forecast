# 모델 튜닝·학습·평가 및 앙상블 선택
#   입력: outputs/전처리/ 최신 meta, 같은 생성시간의 outputs/Train/, Valid/, Test/ 데이터
#   출력: outputs/학습결과/{생성시간}/
#         학습결과_{생성시간}.xlsx (시트 1개, 모델별 1행), models/{모델}_{생성시간}.pkl, ensemble_{생성시간}.pkl
#   실행: python train.py
#         python train.py --deploy      (기존보다 좋으면 시뮬레이션 모델 교체)
#         python train.py --workers 4   (ML 모델 병렬 수, ML 모델 수보다 크면 모델 수로 맞춤)

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import joblib
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm
import config
from models import MODEL_SPECS, PAPERS, SEED, predict_ensemble   # 같은 폴더의 models.py
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 학습 설정 (SEED는 models.py와 같은 값)
MODELS = ["ridge", "rf", "xgb", "lgbm", "gru", "cnn"]   # 학습할 모델
TRIALS = 20                # 모델별 Optuna 시도 횟수 (딥러닝은 1/3)
TUNE_ROWS = 30000          # 튜닝용 샘플 수
EARLY_MIN = 8              # 초반 MAE 기준 (분)
REPLAY_SPLIT = "valid"     # 시뮬레이션 재생 기간 (test는 연휴라 피크가 거의 없음)
SPLITS = ["train", "valid", "test"]

logger = config.log_write_print("train")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


# 전처리 결과 로드 (최신 meta와 같은 생성시간의 Train, Valid, Test)
def load_data():
    meta_path = config.latest_file(config.PREP_DIR, "meta", "json")
    if meta_path is None:
        logger.error(f"전처리 결과(meta) 없음: {config.PREP_DIR} -> preprocess.py 먼저 실행")
        sys.exit(1)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    prep_stamp = meta["stamp"]

    parts = {}   # 분할별 DataFrame
    data = {}    # 분할별 (요약 특징, 시퀀스, 잔차)
    for split in SPLITS:
        folder = config.SPLIT_DIRS[split]
        df = pd.read_parquet(os.path.join(folder, f"{split}_{prep_stamp}.parquet"))
        X_seq = np.load(os.path.join(folder, f"{split}_seq_{prep_stamp}.npy"))

        # 로드 확인: 시퀀스 개수(X_seq 첫 번째 숫자)와 parquet 행 수가 같아야 정상
        logger.info(f"{split}: parquet {len(df)}행, seq {X_seq.shape}")
        if X_seq.shape[0] != len(df):
            logger.error(f"  -> {split} 개수가 안 맞음! 전처리 다시 확인")
            sys.exit(1)

        parts[split] = df
        data[split] = (
            df[config.TAB_COLS].to_numpy(dtype="float32"),
            X_seq,
            df["residual"].to_numpy(dtype="float64"),
        )
    return meta, parts, data


# 평가 지표 (15분 구간 단위로 피크 판정)
#   - 놓친 피크: 목표 초과 구간인데 12분까지 경보 없음
#   - 헛경보: 실제가 목표의 90% 미만인데 경보
#   - 정확도·정밀도·재현율·F1: 정답 = 실제 평균 >= 목표, 예측 = 12분까지 한 번이라도 예측 >= 목표
def evaluate(part, pred, target_kw):
    actual = part["actual_kw"].to_numpy()
    elapsed = part["elapsed_min"].to_numpy()
    err = np.abs(pred - actual)
    early = elapsed <= EARLY_MIN

    frame = pd.DataFrame({
        "window_start": part["window_start"].to_numpy(),
        "actual": actual,
        "alarm": pred >= target_kw,
        "alarm_12": (pred >= target_kw) & (elapsed <= 12),
    })
    windows = frame.groupby("window_start").agg(
        actual=("actual", "first"),
        alarm=("alarm", "any"),
        alarm_12=("alarm_12", "any"),
    )
    is_peak = windows["actual"] >= target_kw
    caught = windows["alarm_12"]

    tp = int((is_peak & caught).sum())
    fp = int((~is_peak & caught).sum())
    fn = int((is_peak & ~caught).sum())
    tn = int((~is_peak & ~caught).sum())

    # 분모가 0이면 0으로 기록
    precision = 0.0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    recall = 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "MAE": float(err.mean()),
        "초반 MAE": float(err[early].mean()),
        "피크 구간": int(is_peak.sum()),
        "놓친 피크": fn,
        "헛경보": int(((windows["actual"] < target_kw * 0.9) & windows["alarm"]).sum()),
        "정확도": (tp + tn) / len(windows),
        "정밀도": precision,
        "재현율": recall,
        "F1": f1,
    }


# train/valid/test 전체 평가
def evaluate_all(parts, preds, target_kw):
    scores = {}
    for split in SPLITS:
        scores[split] = evaluate(parts[split], preds[split], target_kw)
    return scores


# 과적합 점검
#   - 학습 과적합: 모델의 valid/train 비율 ÷ 기준 계산식의 valid/train 비율 > limit
#     (기준 계산식은 학습을 안 하므로 그 비율 = 기간 자체의 난이도 차이 → 이걸 빼고 판정)
#   - valid 과적합: test/valid MAE 비율 > limit
def overfit_check(scores, base_scores, limit=1.3):
    valid_train = scores["valid"]["MAE"] / scores["train"]["MAE"]
    base_ratio = base_scores["valid"]["MAE"] / base_scores["train"]["MAE"]
    relative = valid_train / base_ratio
    test_valid = scores["test"]["MAE"] / scores["valid"]["MAE"]

    flags = []
    if relative > limit:
        flags.append("학습 과적합")
    if test_valid > limit:
        flags.append("valid 과적합")
    return {
        "valid/train MAE": valid_train,
        "기준선 대비 비율": relative,
        "test/valid MAE": test_valid,
        "과적합": ", ".join(flags),
    }


# 결과표 1행: 모델 이름, 유형, 성과지표, 에포크, 파라미터, 과적합
#   - 분류 지표는 valid 위주 (test는 연휴라 피크 구간이 거의 없어 F1이 크게 흔들림 → 피크 구간 수를 같이 표시)
def make_row(kind, name, scores, base_scores, info=None, weights=""):
    train = scores["train"]
    valid = scores["valid"]
    test = scores["test"]
    if info is None:
        info = {}

    row = {
        "모델": name,
        "유형": kind,
        "선택": "",
        # 회귀 지표
        "train MAE": train["MAE"],
        "valid MAE": valid["MAE"],
        "test MAE": test["MAE"],
        "valid 개선율(%)": (1 - valid["MAE"] / base_scores["valid"]["MAE"]) * 100,
        "test 개선율(%)": (1 - test["MAE"] / base_scores["test"]["MAE"]) * 100,
        "valid 초반 MAE": valid["초반 MAE"],
        "test 초반 MAE": test["초반 MAE"],
        # 분류 지표 (15분 구간 단위)
        "valid 정확도": valid["정확도"],
        "valid 정밀도": valid["정밀도"],
        "valid 재현율": valid["재현율"],
        "valid F1": valid["F1"],
        "valid 피크 구간": valid["피크 구간"],
        "valid 놓친 피크": valid["놓친 피크"],
        "valid 헛경보": valid["헛경보"],
        "test F1": test["F1"],
        "test 피크 구간": test["피크 구간"],
        "test 놓친 피크": test["놓친 피크"],
        "test 헛경보": test["헛경보"],
        # 학습 정보
        "trial 수": info.get("trial 수"),
        "실행 에포크": info.get("실행 에포크"),
        "최적 에포크": info.get("최적 에포크"),
        "학습시간(초)": info.get("학습시간(초)"),
        "파라미터": info.get("파라미터", ""),
        "가중치": weights,
    }
    row.update(overfit_check(scores, base_scores))
    row["논문"] = PAPERS.get(name, "")
    return row


# 모델 1개: Optuna 튜닝 -> train 전체 재학습 -> pkl 저장 -> 평가
#   멀티프로세싱 자식에서도 돌아가므로 로그는 파일에 쓰지 않고 logs로 돌려줌
def run_model(name, spec, parts, data, target_kw, pick, n_trials, n_jobs, run_dir, run_id, position=0):
    X_tr, S_tr, y_tr = data["train"]
    X_va, S_va, y_va = data["valid"]

    # Optuna: valid 잔차 MAE 최소화
    def objective(trial):
        model = spec["build"](spec["suggest"](trial), n_jobs)
        model.fit(X_tr[pick], S_tr[pick], y_tr[pick])
        pred = model.predict(X_va, S_va)
        return float(np.mean(np.abs(pred - y_va)))

    pbar = tqdm(total=n_trials, desc=name, position=position)

    def update_bar(study, trial):
        pbar.update(1)

    start = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, callbacks=[update_bar])
    pbar.close()

    # 최적 파라미터로 train 전체 재학습 (DL은 에포크별 valid loss 기록)
    best = study.best_params
    model = spec["build"](best, n_jobs)
    if spec["kind"] == "DL":
        model.fit(X_tr, S_tr, y_tr, valid=(X_va, S_va, y_va))
    else:
        model.fit(X_tr, S_tr, y_tr)
    total_sec = time.time() - start
    joblib.dump(model, os.path.join(run_dir, "models", f"{name}_{run_id}.pkl"))

    # 최종 예측 = 기준 계산식 + 예측 잔차
    preds = {}
    for split in SPLITS:
        X, S, y = data[split]
        preds[split] = parts[split]["proj_base"].to_numpy() + model.predict(X, S)
    scores = evaluate_all(parts, preds, target_kw)

    # 실행 에포크: Optuna가 고른 최대값(best["epochs"])이 아니라 실제로 돈 횟수
    #   → early stopping으로 일찍 멈추면 둘이 다름
    history = getattr(model, "history", [])
    run_epochs = None
    if len(history) > 0:
        run_epochs = len(history)
    info = {
        "trial 수": n_trials,
        "실행 에포크": run_epochs,
        "최적 에포크": getattr(model, "best_epoch", None),
        "학습시간(초)": round(total_sec, 1),
        "파라미터": json.dumps(best, ensure_ascii=False),
    }

    logs = [f"{name} valid MAE {scores['valid']['MAE']:.1f} / test MAE {scores['test']['MAE']:.1f} "
            f"({total_sec:.0f}초)"]

    return {"name": name, "preds": preds, "scores": scores, "info": info, "logs": logs}


# 전체 모델 학습
#   - ML: ProcessPoolExecutor로 병렬, 프로세스 수는 ML 모델 수를 넘지 않게
#         (모델 4개에 workers 10이면 6개는 놀고, 코어만 10등분되어 모델당 코어가 줄어듦)
#   - DL: ML이 끝난 뒤 메인 프로세스에서 순차, 혼자 돌므로 코어 전부 사용
#   - 로그는 결과가 돌아온 뒤 메인에서 기록
def run_all_models(parts, data, target_kw, run_dir, run_id, workers):
    X_tr, S_tr, y_tr = data["train"]

    # 튜닝용 샘플 (TUNE_ROWS개)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(y_tr), min(TUNE_ROWS, len(y_tr)), replace=False)

    ml_names = []
    dl_names = []
    for name in MODELS:
        if MODEL_SPECS[name]["kind"] == "ML":
            ml_names.append(name)
        else:
            dl_names.append(name)
    dl_trials = max(3, TRIALS // 3)   # 딥러닝은 시도 횟수 1/3

    workers = max(1, min(workers, len(ml_names)))
    n_jobs = max(1, os.cpu_count() // workers)
    logger.info(f"ML 병렬 {workers}개 x 모델당 코어 {n_jobs}개")

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = []
        for i, name in enumerate(ml_names):
            jobs.append(pool.submit(run_model, name, MODEL_SPECS[name], parts, data, target_kw, pick,
                                    TRIALS, n_jobs, run_dir, run_id, i))
        for job in as_completed(jobs):
            result = job.result()
            for line in result["logs"]:
                logger.info(line)
            results[result["name"]] = result

    for name in dl_names:
        result = run_model(name, MODEL_SPECS[name], parts, data, target_kw, pick,
                           dl_trials, os.cpu_count(), run_dir, run_id)
        for line in result["logs"]:
            logger.info(line)
        results[name] = result
    return results


# 앙상블: 모델 3개 조합 x (단순 평균, 가중 평균) 중 valid MAE 최소
#   - 가중 평균 가중치 = valid MAE의 역수
def search_ensembles(preds, valid_mae, parts, target_kw, base_scores):
    rows = []
    best = None
    for combo in combinations(MODELS, 3):
        for method in ["단순 평균", "가중 평균"]:
            raw = []
            for m in combo:
                if method == "단순 평균":
                    raw.append(1.0)
                else:
                    raw.append(1.0 / valid_mae[m])
            weights = []
            for w in raw:
                weights.append(w / sum(raw))

            ens_pred = {}
            for split in SPLITS:
                ens_pred[split] = np.zeros(len(parts[split]))
                for m, w in zip(combo, weights):
                    ens_pred[split] = ens_pred[split] + w * preds[m][split]
            scores = evaluate_all(parts, ens_pred, target_kw)

            weight_parts = []
            for m, w in zip(combo, weights):
                weight_parts.append(f"{m} {w:.2f}")
            weight_text = ", ".join(weight_parts)
            row = make_row("앙상블", f"{'+'.join(combo)} ({method})", scores, base_scores, weights=weight_text)
            rows.append(row)

            if best is None or scores["valid"]["MAE"] < best["row"]["valid MAE"]:
                best = {"row": row, "members": list(combo), "weights": weights}

    best["row"]["선택"] = "✔"
    return rows, best


# 시뮬레이션 모델 교체: 기존 모델과 valid MAE 비교 후 개선 시에만 교체 (기존 모델은 백업)
def deploy_model(ens, ens_path, parts, data):
    old_mae = None
    if os.path.exists(config.SIM_MODEL_PATH):
        try:
            old = joblib.load(config.SIM_MODEL_PATH)
            old_pred = predict_ensemble(old, parts["valid"], data["valid"][1])
            old_mae = float(np.mean(np.abs(old_pred - parts["valid"]["actual_kw"].to_numpy())))
        except Exception as e:
            logger.error(f"기존 모델 비교 실패 -> 새 모델로 교체: {e}")

    if old_mae is not None and ens["valid_mae"] > old_mae:
        logger.info(f"배포 안 함: 새 모델 valid {ens['valid_mae']:.1f} > 기존 {old_mae:.1f}")
        return

    os.makedirs(os.path.dirname(config.SIM_MODEL_PATH), exist_ok=True)
    if os.path.exists(config.SIM_MODEL_PATH):
        backup = config.SIM_MODEL_PATH.replace("model.pkl", f"model_backup_{config.stamp()}.pkl")
        shutil.copy(config.SIM_MODEL_PATH, backup)
    shutil.copy(ens_path, config.SIM_MODEL_PATH)

    # 엣지(simulator, controller)용 정보: 모델 로드 없이 목표전력·배율·재생 기간만 읽게 따로 저장
    info = {
        "run_id": ens["run_id"],
        "prep_stamp": ens["prep_stamp"],
        "members": ens["members"],
        "valid_mae": ens["valid_mae"],
        "target_kw": ens["target_kw"],
        "scale": ens["scale"],
        "source": ens["source"],
        "replay_from": ens["replay_from"],
    }
    with open(config.SIM_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logger.info(f"배포 완료 -> {config.SIM_MODEL_PATH}, {config.SIM_INFO_PATH}")


def main():
    args = parse_args()

    # 출력 폴더에 이미 파일이 있으면 중단 (덮어쓰지 않음)
    config.stop_if_output_exists([config.TRAIN_DIR], logger)

    # 1. 데이터 로드
    meta, parts, data = load_data()
    prep_stamp = meta["stamp"]
    target_kw = meta["target_kw"]

    run_id = config.stamp()
    run_dir = os.path.join(config.TRAIN_DIR, run_id)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

    logger.info(f"=== 학습 시작 {run_id} (전처리 {prep_stamp}) ===")
    logger.info(f"목표전력 {target_kw:.0f} kW")

    preds = {}   # 이름별 {분할: 최종 예측}
    rows = []    # 결과표 (모델별 1행)

    # 2. 기준선 평가 (학습 없음), 개선율·과적합 판정의 기준
    base_preds = {}
    for split in SPLITS:
        base_preds[split] = parts[split]["proj_base"].to_numpy()
    base_scores = evaluate_all(parts, base_preds, target_kw)

    for label, col in [("기준 계산식", "proj_base"), ("추세 계산식", "proj_trend")]:
        preds[label] = {}
        for split in SPLITS:
            preds[label][split] = parts[split][col].to_numpy()
        scores = evaluate_all(parts, preds[label], target_kw)
        rows.append(make_row("기준선", label, scores, base_scores))
        logger.info(f"{label} valid MAE {scores['valid']['MAE']:.1f} / test MAE {scores['test']['MAE']:.1f}")

    # 3. 모델별 튜닝 및 학습
    results = run_all_models(parts, data, target_kw, run_dir, run_id, args.workers)

    valid_mae = {}
    for name in MODELS:
        result = results[name]
        preds[name] = result["preds"]
        valid_mae[name] = result["scores"]["valid"]["MAE"]
        rows.append(make_row("단일", name, result["scores"], base_scores, info=result["info"]))

    # 4. 앙상블 선택 (valid MAE 기준)
    ens_rows, best = search_ensembles(preds, valid_mae, parts, target_kw, base_scores)
    best_row = best["row"]
    logger.info(f"앙상블 선택: {best_row['모델']} "
                f"valid MAE {best_row['valid MAE']:.1f} / test MAE {best_row['test MAE']:.1f}")

    # 서빙용 형식으로 저장 (모델은 저장해 둔 pkl에서 로드)
    models = {}
    for m in best["members"]:
        models[m] = joblib.load(os.path.join(run_dir, "models", f"{m}_{run_id}.pkl"))
    ens = {
        "run_id": run_id,
        "prep_stamp": prep_stamp,
        "members": best["members"],
        "weights": best["weights"],
        "models": models,
        "target_kw": target_kw,
        "scale": meta["scale"],
        "source": meta["source"],
        "valid_mae": best_row["valid MAE"],
        "test_mae": best_row["test MAE"],
        "replay_from": meta["split"][REPLAY_SPLIT]["start"],
    }
    ens_path = os.path.join(run_dir, f"ensemble_{run_id}.pkl")
    joblib.dump(ens, ens_path)

    # 5. 결과 출력 + 엑셀 저장 (시트 1개)
    #   순서: 기준선 → 단일 → 앙상블(valid MAE 좋은 순)
    ens_table = pd.DataFrame(ens_rows).sort_values("valid MAE")
    table = pd.concat([pd.DataFrame(rows), ens_table], ignore_index=True).round(3)

    # 콘솔에는 기준선, 단일, 선택된 앙상블만 (앙상블 40행은 엑셀에서 확인)
    show = table[(table["유형"] != "앙상블") | (table["선택"] == "✔")]
    columns = ["모델", "유형", "train MAE", "valid MAE", "test MAE", "valid 개선율(%)",
               "valid F1", "실행 에포크", "기준선 대비 비율", "과적합"]
    logger.info("결과표\n" + show[columns].to_string(index=False))
    logger.info(f"valid 개선율 {best_row['valid 개선율(%)']:.1f}% / test 개선율 {best_row['test 개선율(%)']:.1f}%")

    excel_path = os.path.join(run_dir, f"학습결과_{run_id}.xlsx")
    table.to_excel(excel_path, sheet_name="결과", index=False, freeze_panes=(1, 2))
    logger.info(f"저장: {excel_path}")

    # 6. 시뮬레이션 모델 교체 (--deploy)
    if args.deploy:
        deploy_model(ens, ens_path, parts, data)
    else:
        logger.info("시뮬레이션에 쓰려면 --deploy 붙여서 실행")


if __name__ == "__main__":
    main()