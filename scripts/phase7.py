# =========================================================
# PHASE 7 → CONFIDENCE-GATED DUAL-STAGE RANSOMWARE DETECTION
#
# Experiment 1A:
#   Pure zero-shot evaluation
#   - no dual-use used for training
#   - no dual-use used for threshold selection
#
# Experiment 1B:
#   Zero-shot model + dual-use threshold calibration
#   - no dual-use used for model training
#   - full dual-use set used only for threshold selection
#
# Experiment 2:
#   Dual-use calibrated mitigation with holdout
#   - dual-use split into calibration + final holdout test
#   - calibration split used for threshold selection
#   - final split untouched until final evaluation
#
# Saves structured outputs under:
#   results/phase7_confidence_gating/
# =========================================================

import numpy as np
import pandas as pd

from os import path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import (
    PHASE7_DIR,
    PHASE7_METRICS_DIR,
    PHASE7_THRESHOLD_DIR,
    PHASE7_PREDICTIONS_DIR,
    PHASE7_CURVES_DIR,
    PHASE7_ROC_DIR,
    PHASE7_PR_DIR,
    PHASE7_SUMMARIES_DIR
)
from data_loader import (
    build_training_features,
    build_dualuse_features,
    prepare_xy
)
from utils import make_dir, safe_name


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42

phase7_dir = PHASE7_DIR
metrics_dir = PHASE7_METRICS_DIR
threshold_dir = PHASE7_THRESHOLD_DIR
predictions_dir = PHASE7_PREDICTIONS_DIR
curves_dir = PHASE7_CURVES_DIR
roc_dir = PHASE7_ROC_DIR
pr_dir = PHASE7_PR_DIR
summaries_dir = PHASE7_SUMMARIES_DIR

for d in [
    phase7_dir,
    metrics_dir,
    threshold_dir,
    predictions_dir,
    curves_dir,
    roc_dir,
    pr_dir,
    summaries_dir
]:
    make_dir(d)

HIGH_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
LOW_THRESHOLDS  = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60]

DUAL_CALIBRATION_SIZE = 0.30
MIN_RECALL = 0.97

print("\n")
print("=" * 80)
print("PHASE 7 -> CONFIDENCE-GATED DUAL-STAGE RANSOMWARE DETECTION")
print("=" * 80)

print("Output directory:", phase7_dir)


# =========================================================
# LOAD AND ALIGN DATA
# =========================================================

training_datasets, training_merged_df = (
    build_training_features()
)

dual_datasets, dual_merged_X = (
    build_dualuse_features()
)

X_merged, y_merged = prepare_xy(training_merged_df)

assert isinstance(X_merged, pd.DataFrame), "X_merged must be a pandas DataFrame"
assert isinstance(dual_merged_X, pd.DataFrame), "dual_merged_X must be a pandas DataFrame"

missing_cols = set(X_merged.columns) - set(dual_merged_X.columns)

if missing_cols:
    raise ValueError(
        "Dual-use merged dataset missing columns: "
        f"{sorted(missing_cols)}"
    )

dual_merged_X = dual_merged_X[X_merged.columns]

assert list(X_merged.columns) == list(dual_merged_X.columns), \
    "Feature mismatch between X_merged and dual_merged_X"

print("\nFeature alignment verified.")
print("Training shape :", X_merged.shape)
print("Dual-use shape :", dual_merged_X.shape)


# =========================================================
# HELPER: SAVE ROC / PR CURVE DATA
# =========================================================

def save_curve_data(y_true, y_prob, label):
    """
    Save ROC and PR curve points for later plotting.
    """

    label_safe = safe_name(label)

    # ROC
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true, y_prob)

    roc_df = pd.DataFrame({
        "Evaluation": label,
        "fpr": fpr_arr,
        "tpr": tpr_arr,
        "threshold": roc_thresholds
    })

    roc_path = path.join(
        roc_dir,
        f"{label_safe}_roc_curve.csv"
    )

    roc_df.to_csv(roc_path, index=False)

    # PR
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
        y_true,
        y_prob
    )

    pr_df = pd.DataFrame({
        "Evaluation": label,
        "precision": precision_arr,
        "recall": recall_arr,
        "threshold": list(pr_thresholds) + [np.nan]
    })

    pr_path = path.join(
        pr_dir,
        f"{label_safe}_pr_curve.csv"
    )

    pr_df.to_csv(pr_path, index=False)

    return roc_path, pr_path


# =========================================================
# TRAIN STAGE MODELS
# =========================================================

def train_stage_models_uncertainty(
    X_train,
    y_train,
    initial_low=0.25,
    initial_high=0.75,
    min_samples=None,
    max_iter=8
):
    """
    Train Stage 1 XGBoost and Stage 2 SVM.

    Stage 2 is trained on samples falling inside the low-confidence
    region of Stage 1. If the low-confidence region is too small or
    contains only one class, the region is expanded iteratively.
    """

    print("\n" + "=" * 60)
    print("TRAINING DUAL-STAGE MODEL")
    print("=" * 60)

    if min_samples is None:
        min_samples = max(50, int(0.10 * len(X_train)))

    print("Minimum Stage 2 training samples:", min_samples)

    # -------------------------
    # Stage 1: XGBoost
    # -------------------------

    stage1 = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9
    )

    stage1.fit(X_train, y_train)

    train_probs = stage1.predict_proba(X_train)[:, 1]

    low = initial_low
    high = initial_high

    final_mask = None

    # -------------------------
    # Find valid low-confidence region
    # -------------------------

    for i in range(max_iter):

        low = max(0.0, low)
        high = min(1.0, high)

        uncertainty_mask = (
            (train_probs > low) &
            (train_probs < high)
        )

        uncertain_y = y_train[uncertainty_mask]

        print(f"\nIteration {i + 1}")
        print(f"Bounds: {low:.2f} - {high:.2f}")
        print("Uncertain samples:", len(uncertain_y))

        if len(uncertain_y) == 0:
            print("Empty uncertainty region. Expanding.")
            low -= 0.05
            high += 0.05
            continue

        class_counts = pd.Series(uncertain_y).value_counts()
        print("Stage 2 class distribution:")
        print(class_counts)

        if len(class_counts) < 2:
            print("Only one class found. Expanding.")
            low -= 0.05
            high += 0.05
            continue

        if len(uncertain_y) < min_samples:
            print("Too few samples. Expanding.")
            low -= 0.05
            high += 0.05
            continue

        final_mask = uncertainty_mask
        print("Valid low-confidence region found.")
        break

    if final_mask is None:
        raise RuntimeError(
            "Could not find valid low-confidence region for Stage 2 training."
        )

    print("\nFinal Stage 2 training bounds:")
    print("Low :", low)
    print("High:", high)
    print("Final uncertain samples:", int(final_mask.sum()))
    print("Training routing ratio:", round(final_mask.mean(), 4))
    print("Final Stage 2 class distribution:")
    print(pd.Series(y_train[final_mask]).value_counts())

    # -------------------------
    # Stage 2: SVM verifier
    # -------------------------

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    stage2 = SVC(
        kernel="rbf",
        probability=False,
        random_state=RANDOM_STATE
    )

    stage2.fit(
        X_train_scaled[final_mask],
        y_train[final_mask]
    )

    stage2_training_info = {
        "Stage2 Train Low Bound": low,
        "Stage2 Train High Bound": high,
        "Stage2 Train Samples": int(final_mask.sum()),
        "Stage2 Train Routing Rate": float(final_mask.mean()),
        "Stage2 Train Class Counts": pd.Series(
            y_train[final_mask]
        ).value_counts().to_dict()
    }

    return stage1, stage2, scaler, stage2_training_info


# =========================================================
# DUAL-STAGE PREDICTION
# =========================================================

def dual_stage_predict(
    X,
    stage1_model,
    stage2_model,
    scaler,
    low_threshold,
    high_threshold
):
    """
    Confidence-gated prediction.

    p <= low_threshold  → benign
    p >= high_threshold → ransomware
    otherwise           → SVM verification
    """

    probs = stage1_model.predict_proba(X)[:, 1]
    X_scaled = scaler.transform(X)

    final_preds = np.zeros(len(X), dtype=int)

    stage2_mask = (
        (probs > low_threshold) &
        (probs < high_threshold)
    )

    final_preds[probs >= high_threshold] = 1
    final_preds[probs <= low_threshold] = 0

    if stage2_mask.sum() > 0:
        final_preds[stage2_mask] = stage2_model.predict(
            X_scaled[stage2_mask]
        )

    return final_preds, probs, stage2_mask


# =========================================================
# EVALUATION HELPERS
# =========================================================

def evaluate_predictions(y_true, y_pred, label, y_prob=None):
    """
    Evaluate baseline test performance with full metrics.
    """

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    roc_auc = None
    pr_auc = None
    roc_path = None
    pr_path = None

    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        roc_path, pr_path = save_curve_data(
            y_true=y_true,
            y_prob=y_prob,
            label=label
        )

    return {
        "Evaluation": label,

        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        "Baseline FPR": fpr,
        "Baseline FNR": fnr,

        "ROC Curve File": roc_path,
        "PR Curve File": pr_path
    }


def evaluate_dual_use(preds, label):
    """
    Evaluate dual-use benign samples.

    Since all dual-use samples are benign:
    prediction 1 = false positive
    prediction 0 = true negative
    """

    total = len(preds)

    fp = int(np.sum(preds == 1))
    tn = int(np.sum(preds == 0))

    fp_rate = fp / total
    accuracy = tn / total

    return {
        "Evaluation": label,
        "Dual-use Samples": total,
        "Dual-use FP": fp,
        "Dual-use TN": tn,
        "Dual-use FP Rate": fp_rate,
        "Dual-use Accuracy": accuracy
    }


# =========================================================
# THRESHOLD SEARCH - BASELINE ONLY
# =========================================================

def threshold_grid_search_baseline_only(
    X_val,
    y_val,
    stage1_model,
    stage2_model,
    scaler,
    min_recall=0.97
):
    """
    Select thresholds using only baseline validation data.
    Used for pure zero-shot setting.
    """

    rows = []

    for low in LOW_THRESHOLDS:
        for high in HIGH_THRESHOLDS:

            if low >= high:
                continue

            preds, probs, stage2_mask = dual_stage_predict(
                X_val,
                stage1_model,
                stage2_model,
                scaler,
                low_threshold=low,
                high_threshold=high
            )

            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

            rows.append({
                "Low Threshold": low,
                "High Threshold": high,

                "Validation Accuracy": accuracy_score(y_val, preds),
                "Validation Precision": precision_score(y_val, preds, zero_division=0),
                "Validation Recall": recall_score(y_val, preds, zero_division=0),
                "Validation F1": f1_score(y_val, preds, zero_division=0),
                "Validation ROC-AUC": roc_auc_score(y_val, probs),
                "Validation PR-AUC": average_precision_score(y_val, probs),

                "Validation TN": int(tn),
                "Validation FP": int(fp),
                "Validation FN": int(fn),
                "Validation TP": int(tp),

                "Validation Stage2 Routed Count": int(stage2_mask.sum()),
                "Validation Stage2 Routed Rate": stage2_mask.mean()
            })

    result_df = pd.DataFrame(rows)

    valid_df = result_df[
        result_df["Validation Recall"] >= min_recall
    ].copy()

    if valid_df.empty:
        best_row = result_df.sort_values(
            by=["Validation Recall", "Validation F1", "Validation Precision"],
            ascending=False
        ).iloc[0]
    else:
        best_row = valid_df.sort_values(
            by=["Validation Precision", "Validation F1", "Validation Accuracy"],
            ascending=False
        ).iloc[0]

    return best_row, result_df


# =========================================================
# THRESHOLD SEARCH - DUAL-AWARE CALIBRATION
# =========================================================

def threshold_grid_search_dual_aware(
    X_val,
    y_val,
    X_dual_calib,
    stage1_model,
    stage2_model,
    scaler,
    min_recall=0.97
):
    """
    Select thresholds using baseline validation recall constraint
    and dual-use false-positive minimization.
    """

    rows = []

    for low in LOW_THRESHOLDS:
        for high in HIGH_THRESHOLDS:

            if low >= high:
                continue

            val_preds, val_probs, val_stage2_mask = dual_stage_predict(
                X_val,
                stage1_model,
                stage2_model,
                scaler,
                low_threshold=low,
                high_threshold=high
            )

            dual_preds, dual_probs, dual_stage2_mask = dual_stage_predict(
                X_dual_calib,
                stage1_model,
                stage2_model,
                scaler,
                low_threshold=low,
                high_threshold=high
            )

            tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()

            val_recall = recall_score(y_val, val_preds, zero_division=0)
            val_precision = precision_score(y_val, val_preds, zero_division=0)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)

            dual_fp = int(np.sum(dual_preds == 1))
            dual_tn = int(np.sum(dual_preds == 0))
            dual_fp_rate = dual_fp / len(dual_preds)

            rows.append({
                "Low Threshold": low,
                "High Threshold": high,

                "Validation Accuracy": accuracy_score(y_val, val_preds),
                "Validation Precision": val_precision,
                "Validation Recall": val_recall,
                "Validation F1": val_f1,
                "Validation ROC-AUC": roc_auc_score(y_val, val_probs),
                "Validation PR-AUC": average_precision_score(y_val, val_probs),

                "Validation TN": int(tn),
                "Validation FP": int(fp),
                "Validation FN": int(fn),
                "Validation TP": int(tp),

                "Dual Calibration TN": dual_tn,
                "Dual Calibration FP": dual_fp,
                "Dual Calibration FP Rate": dual_fp_rate,

                "Validation Stage2 Routed Count": int(val_stage2_mask.sum()),
                "Validation Stage2 Routed Rate": val_stage2_mask.mean(),

                "Dual Stage2 Routed Count": int(dual_stage2_mask.sum()),
                "Dual Stage2 Routed Rate": dual_stage2_mask.mean()
            })

    result_df = pd.DataFrame(rows)

    valid_df = result_df[
        result_df["Validation Recall"] >= min_recall
    ].copy()

    if valid_df.empty:
        best_row = result_df.sort_values(
            by=[
                "Validation Recall",
                "Dual Calibration FP Rate",
                "Validation F1"
            ],
            ascending=[False, True, False]
        ).iloc[0]
    else:
        best_row = valid_df.sort_values(
            by=[
                "Dual Calibration FP Rate",
                "Validation F1",
                "Validation Precision"
            ],
            ascending=[True, False, False]
        ).iloc[0]

    return best_row, result_df


# =========================================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_merged,
    y_merged,
    test_size=0.20,
    stratify=y_merged,
    random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

print("\nBaseline split:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# =========================================================
# TRAIN SHARED ZERO-SHOT MODEL
# Model never sees dual-use samples
# =========================================================

stage1_model, stage2_model, scaler_model, stage2_info = train_stage_models_uncertainty(
    X_train,
    y_train,
    initial_low=0.25,
    initial_high=0.75,
    min_samples=None
)


# =========================================================
# EXPERIMENT 1A - PURE ZERO-SHOT
# =========================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1A → PURE ZERO-SHOT")
print("=" * 80)

best_1a, grid_1a = threshold_grid_search_baseline_only(
    X_val,
    y_val,
    stage1_model,
    stage2_model,
    scaler_model,
    min_recall=MIN_RECALL
)

exp1a_low = float(best_1a["Low Threshold"])
exp1a_high = float(best_1a["High Threshold"])

print("\nBest Experiment 1A thresholds:")
print(best_1a)

grid_1a_path = path.join(
    threshold_dir,
    "experiment1a_pure_zero_shot_threshold_grid.csv"
)

grid_1a.to_csv(grid_1a_path, index=False)
print("Saved threshold grid →", grid_1a_path)

test_preds_1a, test_probs_1a, test_stage2_mask_1a = dual_stage_predict(
    X_test,
    stage1_model,
    stage2_model,
    scaler_model,
    low_threshold=exp1a_low,
    high_threshold=exp1a_high
)

dual_preds_1a, dual_probs_1a, dual_stage2_mask_1a = dual_stage_predict(
    dual_merged_X,
    stage1_model,
    stage2_model,
    scaler_model,
    low_threshold=exp1a_low,
    high_threshold=exp1a_high
)

exp1a_test_result = evaluate_predictions(
    y_test,
    test_preds_1a,
    "Experiment 1A - Baseline Test",
    y_prob=test_probs_1a
)

exp1a_dual_result = evaluate_dual_use(
    dual_preds_1a,
    "Experiment 1A - Pure Zero-shot Dual-use"
)


# =========================================================
# EXPERIMENT 1B - ZERO-SHOT MODEL + DUAL-USE THRESHOLD CALIBRATION
# =========================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1B → ZERO-SHOT MODEL + DUAL-USE THRESHOLD CALIBRATION")
print("=" * 80)

best_1b, grid_1b = threshold_grid_search_dual_aware(
    X_val,
    y_val,
    dual_merged_X,
    stage1_model,
    stage2_model,
    scaler_model,
    min_recall=MIN_RECALL
)

exp1b_low = float(best_1b["Low Threshold"])
exp1b_high = float(best_1b["High Threshold"])

print("\nBest Experiment 1B thresholds:")
print(best_1b)

grid_1b_path = path.join(
    threshold_dir,
    "experiment1b_dual_aware_threshold_grid.csv"
)

grid_1b.to_csv(grid_1b_path, index=False)
print("Saved threshold grid →", grid_1b_path)

test_preds_1b, test_probs_1b, test_stage2_mask_1b = dual_stage_predict(
    X_test,
    stage1_model,
    stage2_model,
    scaler_model,
    low_threshold=exp1b_low,
    high_threshold=exp1b_high
)

dual_preds_1b, dual_probs_1b, dual_stage2_mask_1b = dual_stage_predict(
    dual_merged_X,
    stage1_model,
    stage2_model,
    scaler_model,
    low_threshold=exp1b_low,
    high_threshold=exp1b_high
)

exp1b_test_result = evaluate_predictions(
    y_test,
    test_preds_1b,
    "Experiment 1B - Baseline Test",
    y_prob=test_probs_1b
)

exp1b_dual_result = evaluate_dual_use(
    dual_preds_1b,
    "Experiment 1B - Dual-aware Calibrated Dual-use"
)


# =========================================================
# EXPERIMENT 2 - DUAL-USE CALIBRATION WITH HOLDOUT
# =========================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2 → DUAL-USE CALIBRATION WITH HOLDOUT")
print("=" * 80)

dual_calib_X, dual_final_X = train_test_split(
    dual_merged_X,
    test_size=1.0 - DUAL_CALIBRATION_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\nDual-use split:")
print("Calibration:", dual_calib_X.shape)
print("Final test :", dual_final_X.shape)

stage1_holdout, stage2_holdout, scaler_holdout, stage2_info_holdout = (
    train_stage_models_uncertainty(
        X_train,
        y_train,
        initial_low=0.25,
        initial_high=0.75,
        min_samples=None
    )
)

best_2, grid_2 = threshold_grid_search_dual_aware(
    X_val,
    y_val,
    dual_calib_X,
    stage1_holdout,
    stage2_holdout,
    scaler_holdout,
    min_recall=MIN_RECALL
)

exp2_low = float(best_2["Low Threshold"])
exp2_high = float(best_2["High Threshold"])

print("\nBest Experiment 2 thresholds:")
print(best_2)

grid_2_path = path.join(
    threshold_dir,
    "experiment2_dual_holdout_threshold_grid.csv"
)

grid_2.to_csv(grid_2_path, index=False)
print("Saved threshold grid →", grid_2_path)

test_preds_2, test_probs_2, test_stage2_mask_2 = dual_stage_predict(
    X_test,
    stage1_holdout,
    stage2_holdout,
    scaler_holdout,
    low_threshold=exp2_low,
    high_threshold=exp2_high
)

dual_preds_2, dual_probs_2, dual_stage2_mask_2 = dual_stage_predict(
    dual_final_X,
    stage1_holdout,
    stage2_holdout,
    scaler_holdout,
    low_threshold=exp2_low,
    high_threshold=exp2_high
)

exp2_test_result = evaluate_predictions(
    y_test,
    test_preds_2,
    "Experiment 2 - Baseline Test",
    y_prob=test_probs_2
)

exp2_dual_result = evaluate_dual_use(
    dual_preds_2,
    "Experiment 2 - Dual-use Holdout Test"
)


# =========================================================
# SINGLE-MODEL REFERENCES
# =========================================================

print("\n" + "=" * 80)
print("SINGLE-MODEL BASELINE REFERENCES")
print("=" * 80)

# -------------------------
# XGBoost reference
# -------------------------

xgb_ref = XGBClassifier(
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9
)

xgb_ref.fit(X_train, y_train)

xgb_test_preds = xgb_ref.predict(X_test)
xgb_test_probs = xgb_ref.predict_proba(X_test)[:, 1]

xgb_dual_preds = xgb_ref.predict(dual_merged_X)
xgb_dual_probs = xgb_ref.predict_proba(dual_merged_X)[:, 1]

xgb_test_result = evaluate_predictions(
    y_test,
    xgb_test_preds,
    "Reference - XGBoost Test",
    y_prob=xgb_test_probs
)

xgb_dual_result = evaluate_dual_use(
    xgb_dual_preds,
    "Reference - XGBoost Dual-use"
)

xgb_reference_fp_rate = xgb_dual_result["Dual-use FP Rate"]


# -------------------------
# SVM reference
# -------------------------

svm_scaler_ref = StandardScaler()

X_train_scaled_ref = svm_scaler_ref.fit_transform(X_train)
X_test_scaled_ref = svm_scaler_ref.transform(X_test)
dual_scaled_ref = svm_scaler_ref.transform(dual_merged_X)

svm_ref = SVC(
    kernel="rbf",
    probability=False,
    random_state=RANDOM_STATE
)

svm_ref.fit(X_train_scaled_ref, y_train)

svm_test_preds = svm_ref.predict(X_test_scaled_ref)
svm_test_scores = svm_ref.decision_function(X_test_scaled_ref)

svm_dual_preds = svm_ref.predict(dual_scaled_ref)
svm_dual_scores = svm_ref.decision_function(dual_scaled_ref)

svm_test_result = evaluate_predictions(
    y_test,
    svm_test_preds,
    "Reference - SVM Test",
    y_prob=svm_test_scores
)

svm_dual_result = evaluate_dual_use(
    svm_dual_preds,
    "Reference - SVM Dual-use"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

summary_rows = []


def add_summary_row(
    experiment,
    setting,
    stage2_info_row,
    low_threshold,
    high_threshold,
    test_result,
    dual_result,
    test_stage2_mask,
    dual_stage2_mask,
    dual_used_for_threshold,
    dual_holdout_used,
    reference_fp_rate=None
):
    current_fp_rate = dual_result["Dual-use FP Rate"]

    relative_fp_reduction = None

    if reference_fp_rate is not None and reference_fp_rate > 0:
        relative_fp_reduction = (
            (reference_fp_rate - current_fp_rate) / reference_fp_rate
        )

    summary_rows.append({
        "Experiment": experiment,
        "Setting": setting,

        "Stage2 Train Low Bound": stage2_info_row.get("Stage2 Train Low Bound"),
        "Stage2 Train High Bound": stage2_info_row.get("Stage2 Train High Bound"),
        "Stage2 Train Samples": stage2_info_row.get("Stage2 Train Samples"),
        "Stage2 Train Routing Rate": stage2_info_row.get("Stage2 Train Routing Rate"),
        "Stage2 Train Class Counts": str(stage2_info_row.get("Stage2 Train Class Counts")),

        "Inference Low Threshold": low_threshold,
        "Inference High Threshold": high_threshold,

        "Baseline Accuracy": test_result["Accuracy"],
        "Baseline Precision": test_result["Precision"],
        "Baseline Recall": test_result["Recall"],
        "Baseline F1": test_result["F1 Score"],
        "Baseline ROC-AUC": test_result["ROC-AUC"],
        "Baseline PR-AUC": test_result["PR-AUC"],
        "Baseline FPR": test_result["Baseline FPR"],
        "Baseline FNR": test_result["Baseline FNR"],

        "TN": test_result["TN"],
        "FP": test_result["FP"],
        "FN": test_result["FN"],
        "TP": test_result["TP"],

        "Dual-use Samples": dual_result["Dual-use Samples"],
        "Dual-use FP": dual_result["Dual-use FP"],
        "Dual-use TN": dual_result["Dual-use TN"],
        "Dual-use FP Rate": dual_result["Dual-use FP Rate"],
        "Dual-use Accuracy": dual_result["Dual-use Accuracy"],

        "Relative FP Reduction vs XGBoost": relative_fp_reduction,

        "Baseline Stage2 Routed Count": int(test_stage2_mask.sum()) if test_stage2_mask is not None else None,
        "Baseline Stage2 Routed Rate": float(test_stage2_mask.mean()) if test_stage2_mask is not None else None,

        "Dual-use Stage2 Routed Count": int(dual_stage2_mask.sum()) if dual_stage2_mask is not None else None,
        "Dual-use Stage2 Routed Rate": float(dual_stage2_mask.mean()) if dual_stage2_mask is not None else None,

        "Dual-use Used For Threshold Selection": dual_used_for_threshold,
        "Dual-use Holdout Used": dual_holdout_used,

        "ROC Curve File": test_result.get("ROC Curve File"),
        "PR Curve File": test_result.get("PR Curve File")
    })


add_summary_row(
    "Experiment 1A",
    "Pure zero-shot confidence-gated dual-stage",
    stage2_info,
    exp1a_low,
    exp1a_high,
    exp1a_test_result,
    exp1a_dual_result,
    test_stage2_mask_1a,
    dual_stage2_mask_1a,
    dual_used_for_threshold=False,
    dual_holdout_used=False,
    reference_fp_rate=xgb_reference_fp_rate
)

add_summary_row(
    "Experiment 1B",
    "Dual-aware threshold calibrated confidence-gated dual-stage",
    stage2_info,
    exp1b_low,
    exp1b_high,
    exp1b_test_result,
    exp1b_dual_result,
    test_stage2_mask_1b,
    dual_stage2_mask_1b,
    dual_used_for_threshold=True,
    dual_holdout_used=False,
    reference_fp_rate=xgb_reference_fp_rate
)

add_summary_row(
    "Experiment 2",
    "Dual-use holdout calibrated confidence-gated dual-stage",
    stage2_info_holdout,
    exp2_low,
    exp2_high,
    exp2_test_result,
    exp2_dual_result,
    test_stage2_mask_2,
    dual_stage2_mask_2,
    dual_used_for_threshold=True,
    dual_holdout_used=True,
    reference_fp_rate=xgb_reference_fp_rate
)


# -------------------------
# Reference row: XGBoost
# -------------------------

summary_rows.append({
    "Experiment": "Reference",
    "Setting": "XGBoost only",

    "Stage2 Train Low Bound": None,
    "Stage2 Train High Bound": None,
    "Stage2 Train Samples": None,
    "Stage2 Train Routing Rate": None,
    "Stage2 Train Class Counts": None,

    "Inference Low Threshold": None,
    "Inference High Threshold": None,

    "Baseline Accuracy": xgb_test_result["Accuracy"],
    "Baseline Precision": xgb_test_result["Precision"],
    "Baseline Recall": xgb_test_result["Recall"],
    "Baseline F1": xgb_test_result["F1 Score"],
    "Baseline ROC-AUC": xgb_test_result["ROC-AUC"],
    "Baseline PR-AUC": xgb_test_result["PR-AUC"],
    "Baseline FPR": xgb_test_result["Baseline FPR"],
    "Baseline FNR": xgb_test_result["Baseline FNR"],

    "TN": xgb_test_result["TN"],
    "FP": xgb_test_result["FP"],
    "FN": xgb_test_result["FN"],
    "TP": xgb_test_result["TP"],

    "Dual-use Samples": xgb_dual_result["Dual-use Samples"],
    "Dual-use FP": xgb_dual_result["Dual-use FP"],
    "Dual-use TN": xgb_dual_result["Dual-use TN"],
    "Dual-use FP Rate": xgb_dual_result["Dual-use FP Rate"],
    "Dual-use Accuracy": xgb_dual_result["Dual-use Accuracy"],

    "Relative FP Reduction vs XGBoost": 0.0,

    "Baseline Stage2 Routed Count": None,
    "Baseline Stage2 Routed Rate": None,
    "Dual-use Stage2 Routed Count": None,
    "Dual-use Stage2 Routed Rate": None,

    "Dual-use Used For Threshold Selection": False,
    "Dual-use Holdout Used": False,

    "ROC Curve File": xgb_test_result.get("ROC Curve File"),
    "PR Curve File": xgb_test_result.get("PR Curve File")
})


# -------------------------
# Reference row: SVM
# -------------------------

summary_rows.append({
    "Experiment": "Reference",
    "Setting": "SVM only",

    "Stage2 Train Low Bound": None,
    "Stage2 Train High Bound": None,
    "Stage2 Train Samples": None,
    "Stage2 Train Routing Rate": None,
    "Stage2 Train Class Counts": None,

    "Inference Low Threshold": None,
    "Inference High Threshold": None,

    "Baseline Accuracy": svm_test_result["Accuracy"],
    "Baseline Precision": svm_test_result["Precision"],
    "Baseline Recall": svm_test_result["Recall"],
    "Baseline F1": svm_test_result["F1 Score"],
    "Baseline ROC-AUC": svm_test_result["ROC-AUC"],
    "Baseline PR-AUC": svm_test_result["PR-AUC"],
    "Baseline FPR": svm_test_result["Baseline FPR"],
    "Baseline FNR": svm_test_result["Baseline FNR"],

    "TN": svm_test_result["TN"],
    "FP": svm_test_result["FP"],
    "FN": svm_test_result["FN"],
    "TP": svm_test_result["TP"],

    "Dual-use Samples": svm_dual_result["Dual-use Samples"],
    "Dual-use FP": svm_dual_result["Dual-use FP"],
    "Dual-use TN": svm_dual_result["Dual-use TN"],
    "Dual-use FP Rate": svm_dual_result["Dual-use FP Rate"],
    "Dual-use Accuracy": svm_dual_result["Dual-use Accuracy"],

    "Relative FP Reduction vs XGBoost": (
        (xgb_reference_fp_rate - svm_dual_result["Dual-use FP Rate"])
        / xgb_reference_fp_rate
    ),

    "Baseline Stage2 Routed Count": None,
    "Baseline Stage2 Routed Rate": None,
    "Dual-use Stage2 Routed Count": None,
    "Dual-use Stage2 Routed Rate": None,

    "Dual-use Used For Threshold Selection": False,
    "Dual-use Holdout Used": False,

    "ROC Curve File": svm_test_result.get("ROC Curve File"),
    "PR Curve File": svm_test_result.get("PR Curve File")
})


# =========================================================
# SAVE FINAL SUMMARY
# =========================================================

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(summary_df)

summary_path = path.join(
    summaries_dir,
    "phase7_confidence_gating_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\nSaved final summary →", summary_path)


# =========================================================
# SAVE DETAILED PREDICTIONS
# =========================================================

# -------------------------
# Test predictions
# -------------------------

test_predictions_df = pd.DataFrame({
    "y_true": y_test.values,

    "Experiment1A_Pred": test_preds_1a,
    "Experiment1A_Prob": test_probs_1a,
    "Experiment1A_Stage2_Routed": test_stage2_mask_1a.astype(int),

    "Experiment1B_Pred": test_preds_1b,
    "Experiment1B_Prob": test_probs_1b,
    "Experiment1B_Stage2_Routed": test_stage2_mask_1b.astype(int),

    "Experiment2_Pred": test_preds_2,
    "Experiment2_Prob": test_probs_2,
    "Experiment2_Stage2_Routed": test_stage2_mask_2.astype(int),

    "XGBoost_Reference_Pred": xgb_test_preds,
    "XGBoost_Reference_Prob": xgb_test_probs,

    "SVM_Reference_Pred": svm_test_preds,
    "SVM_Reference_Score": svm_test_scores
})

test_predictions_path = path.join(
    predictions_dir,
    "phase7_baseline_test_predictions.csv"
)

test_predictions_df.to_csv(
    test_predictions_path,
    index=False
)

print("Saved baseline test detailed predictions →", test_predictions_path)


# -------------------------
# Dual-use predictions for full dual-use set
# Experiment 1A, 1B, XGBoost, SVM
# -------------------------

detailed_dual_predictions = pd.DataFrame({
    "Experiment1A_Pred": dual_preds_1a,
    "Experiment1A_Prob": dual_probs_1a,
    "Experiment1A_Stage2_Routed": dual_stage2_mask_1a.astype(int),

    "Experiment1B_Pred": dual_preds_1b,
    "Experiment1B_Prob": dual_probs_1b,
    "Experiment1B_Stage2_Routed": dual_stage2_mask_1b.astype(int),

    "XGBoost_Reference_Pred": xgb_dual_preds,
    "XGBoost_Reference_Prob": xgb_dual_probs,

    "SVM_Reference_Pred": svm_dual_preds,
    "SVM_Reference_Score": svm_dual_scores
})

detailed_predictions_path = path.join(
    predictions_dir,
    "phase7_dualuse_predictions.csv"
)

detailed_dual_predictions.to_csv(
    detailed_predictions_path,
    index=False
)

print("Saved dual-use detailed predictions →", detailed_predictions_path)


# -------------------------
# Experiment 2 holdout predictions
# Holdout has fewer rows, so save separately
# -------------------------

experiment2_holdout_predictions = pd.DataFrame({
    "Experiment2_Pred": dual_preds_2,
    "Experiment2_Prob": dual_probs_2,
    "Experiment2_Stage2_Routed": dual_stage2_mask_2.astype(int)
})

experiment2_holdout_path = path.join(
    predictions_dir,
    "phase7_experiment2_dualuse_holdout_predictions.csv"
)

experiment2_holdout_predictions.to_csv(
    experiment2_holdout_path,
    index=False
)

print("Saved Experiment 2 holdout predictions →", experiment2_holdout_path)


# =========================================================
# DONE
# =========================================================

print("\n")
print("=" * 80)
print("PHASE 7 CONFIDENCE-GATING COMPLETE")
print("=" * 80)

print("Summary saved at:")
print(summary_path)

print("\nAll outputs saved under:")
print(phase7_dir)
