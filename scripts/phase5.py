# =========================================================
# PHASE 5 → STATISTICAL VALIDATION
# Includes:
#   1. Wilson confidence intervals
#   2. McNemar's test
#   3. Stratified k-fold cross-validation
#   4. Paired t-test on fold-wise CV scores
# =========================================================

import os
from os import path
import numpy as np
import pandas as pd

from google.colab import drive

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

from scipy.stats import ttest_rel

phase5_summary_dir = make_dir(path.join(phase5_dir, "summaries"))
phase5_tests_dir = make_dir(path.join(phase5_dir, "tests"))
phase5_cv_dir = make_dir(path.join(phase5_dir, "kfold_cv"))

phase2_pred_dir = path.join(RESULTS_ROOT, "phase2_merged", "predictions")
phase4_pred_dir = path.join(RESULTS_ROOT, "phase4_cross_dataset", "predictions")

print("Materials:", materials)
print("Results root:", RESULTS_ROOT)
print("Phase 5 output:", phase5_dir)
print("Phase 2 prediction dir:", phase2_pred_dir)
print("Phase 4 prediction dir:", phase4_pred_dir)

assert path.exists(phase2_pred_dir), f"Missing: {phase2_pred_dir}"
assert path.exists(phase4_pred_dir), f"Missing: {phase4_pred_dir}"


model_file_tokens = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Logistic Regression": "logistic_regression",
    "SVM": "svm"
}

# =========================================================
# BUILD MERGED FEATURE DATASET
# Same structure as Phase 2
# =========================================================

drop_feature_cols = ["ID", "RG", "filename", "family"]

meta_cols = [c for c in ["ID", "RG", "filename"] if c in header_df.columns]

_, merged_df = build_training_features()

X_merged, y_merged = prepare_xy(
    merged_df
)	
	

print("\nMerged dataset:")
print("X_merged:", X_merged.shape)
print("y_merged:", y_merged.shape)
print("Class distribution:")
print(y_merged.value_counts())



# =========================================================
# PART 1 → WILSON 95% CONFIDENCE INTERVALS
# =========================================================


wilson_rows = []



for model_name, token in model_file_tokens.items():

    # Internal baseline recall
    try:
        internal_df, internal_path = load_prediction_file("internal", token)

        tp = int(((internal_df["y_true"] == 1) & (internal_df["y_pred"] == 1)).sum())
        fn = int(((internal_df["y_true"] == 1) & (internal_df["y_pred"] == 0)).sum())

        add_wilson_row(
            dataset="Internal Baseline Test",
            model=model_name,
            metric="Ransomware Recall",
            successes=tp,
            total=tp + fn,
            file_path=internal_path
        )

    except Exception as e:
        print(f"Skipped internal CI for {model_name}: {e}")

    # Dual-use FPR
    try:
        dual_df, dual_path = load_prediction_file("dualuse", token)

        fp = int(((dual_df["y_true"] == 0) & (dual_df["y_pred"] == 1)).sum())
        tn = int(((dual_df["y_true"] == 0) & (dual_df["y_pred"] == 0)).sum())

        add_wilson_row(
            dataset="Dual-use Benign",
            model=model_name,
            metric="Dual-use FPR",
            successes=fp,
            total=fp + tn,
            file_path=dual_path
        )

    except Exception as e:
        print(f"Skipped dual-use CI for {model_name}: {e}")

    # VERA external recall
    try:
        vera_df, vera_path = load_prediction_file("vera", token)

        tp = int(((vera_df["y_true"] == 1) & (vera_df["y_pred"] == 1)).sum())
        fn = int(((vera_df["y_true"] == 1) & (vera_df["y_pred"] == 0)).sum())

        add_wilson_row(
            dataset="VERA Active",
            model=model_name,
            metric="External Detection Rate / Recall",
            successes=tp,
            total=tp + fn,
            file_path=vera_path
        )

    except Exception as e:
        print(f"Skipped VERA CI for {model_name}: {e}")

    # Dike external benign FPR
    try:
        dike_df, dike_path = load_prediction_file("dike", token)

        fp = int(((dike_df["y_true"] == 0) & (dike_df["y_pred"] == 1)).sum())
        tn = int(((dike_df["y_true"] == 0) & (dike_df["y_pred"] == 0)).sum())

        add_wilson_row(
            dataset="DikeDataset Benign PE",
            model=model_name,
            metric="External Benign FPR",
            successes=fp,
            total=fp + tn,
            file_path=dike_path
        )

    except Exception as e:
        print(f"Skipped Dike CI for {model_name}: {e}")


wilson_df = pd.DataFrame(wilson_rows)

wilson_path = path.join(
    phase5_summary_dir,
    "phase5_wilson_confidence_intervals.csv"
)

wilson_df.to_csv(wilson_path, index=False)

print("\nWilson confidence intervals saved:")
print(wilson_path)

display(wilson_df)


# =========================================================
# PART 2 → MCNEMAR TEST
# =========================================================


comparisons = [
    ("internal", "Internal Baseline Test", "XGBoost", "xgboost", "LightGBM", "lightgbm"),
    ("internal", "Internal Baseline Test", "XGBoost", "xgboost", "SVM", "svm"),

    ("dualuse", "Dual-use Benign", "XGBoost", "xgboost", "SVM", "svm"),
    ("dualuse", "Dual-use Benign", "XGBoost", "xgboost", "Logistic Regression", "logistic_regression"),
    ("dualuse", "Dual-use Benign", "SVM", "svm", "Logistic Regression", "logistic_regression"),

    ("vera", "VERA Active", "Logistic Regression", "logistic_regression", "XGBoost", "xgboost"),
    ("vera", "VERA Active", "Logistic Regression", "logistic_regression", "LightGBM", "lightgbm"),
    ("vera", "VERA Active", "Logistic Regression", "logistic_regression", "SVM", "svm"),
    ("vera", "VERA Active", "XGBoost", "xgboost", "SVM", "svm"),

    ("dike", "DikeDataset Benign PE", "XGBoost", "xgboost", "Logistic Regression", "logistic_regression"),
    ("dike", "DikeDataset Benign PE", "XGBoost", "xgboost", "Random Forest", "random_forest"),
    ("dike", "DikeDataset Benign PE", "XGBoost", "xgboost", "SVM", "svm"),
    ("dike", "DikeDataset Benign PE", "LightGBM", "lightgbm", "SVM", "svm"),
]


mcnemar_rows = []

for dataset_type, dataset_name, model_a, token_a, model_b, token_b in comparisons:
    try:
        df_a, path_a = load_prediction_file(dataset_type, token_a)
        df_b, path_b = load_prediction_file(dataset_type, token_b)

        row = run_mcnemar_test(
            df_a=df_a,
            df_b=df_b,
            model_a=model_a,
            model_b=model_b,
            dataset_name=dataset_name
        )

        row["Model A File"] = path_a
        row["Model B File"] = path_b

        mcnemar_rows.append(row)

    except Exception as e:
        print("\nSkipped McNemar comparison:")
        print(dataset_name, "|", model_a, "vs", model_b)
        print("Reason:", e)


mcnemar_df = pd.DataFrame(mcnemar_rows)

mcnemar_path = path.join(
    phase5_tests_dir,
    "phase5_mcnemar_tests.csv"
)

mcnemar_df.to_csv(mcnemar_path, index=False)

print("\nMcNemar tests saved:")
print(mcnemar_path)

display(mcnemar_df)


# =========================================================
# PART 3 → STRATIFIED K-FOLD CROSS-VALIDATION
# =========================================================


cv_results_df = run_kfold_cv(
    X=X_merged,
    y=y_merged,
    n_splits=5,
    random_state=42
)

cv_results_path = path.join(
    phase5_cv_dir,
    "phase5_kfold_cv_foldwise_results.csv"
)

cv_results_df.to_csv(cv_results_path, index=False)

print("\nK-fold CV fold-wise results saved:")
print(cv_results_path)

display(cv_results_df)


# =========================================================
# PART 4 → K-FOLD CV SUMMARY
# Mean ± std across folds
# =========================================================

cv_summary_df = (
    cv_results_df
    .groupby("Model")
    .agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        Precision_Mean=("Precision", "mean"),
        Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"),
        Recall_Std=("Recall", "std"),
        F1_Mean=("F1 Score", "mean"),
        F1_Std=("F1 Score", "std"),
        ROC_AUC_Mean=("ROC-AUC", "mean"),
        ROC_AUC_Std=("ROC-AUC", "std"),
        PR_AUC_Mean=("PR-AUC", "mean"),
        PR_AUC_Std=("PR-AUC", "std")
    )
    .reset_index()
)

# Percent columns for paper readability
for col in cv_summary_df.columns:
    if col != "Model":
        cv_summary_df[col] = cv_summary_df[col].round(4)

cv_summary_path = path.join(
    phase5_cv_dir,
    "phase5_kfold_cv_summary.csv"
)

cv_summary_df.to_csv(cv_summary_path, index=False)

print("\nK-fold CV summary saved:")
print(cv_summary_path)

display(cv_summary_df)


# =========================================================
# PART 5 → PAIRED T-TEST ON K-FOLD SCORES
# =========================================================


ttest_comparisons = [
    ("XGBoost", "LightGBM"),
    ("XGBoost", "SVM"),
    ("LightGBM", "SVM"),
    ("XGBoost", "Logistic Regression"),
    ("LightGBM", "Logistic Regression"),
]

ttest_metrics = [
    "Accuracy",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC"
]

ttest_rows = []

for model_a, model_b in ttest_comparisons:
    for metric in ttest_metrics:
        try:
            row = paired_ttest_cv(
                cv_df=cv_results_df,
                model_a=model_a,
                model_b=model_b,
                metric=metric
            )
            ttest_rows.append(row)

        except Exception as e:
            print("Skipped paired t-test:", model_a, "vs", model_b, metric)
            print("Reason:", e)

ttest_df = pd.DataFrame(ttest_rows)

ttest_path = path.join(
    phase5_tests_dir,
    "phase5_paired_ttests_kfold.csv"
)

ttest_df.to_csv(ttest_path, index=False)

print("\nPaired t-tests saved:")
print(ttest_path)

display(ttest_df)


# =========================================================
# PART 6 → PAPER-READY SUMMARY TABLES
# =========================================================

key_ci_df = wilson_df[
    wilson_df["Dataset"].isin([
        "Dual-use Benign",
        "VERA Active",
        "DikeDataset Benign PE"
    ])
].copy()

key_ci_df = key_ci_df[
    [
        "Dataset",
        "Model",
        "Metric",
        "Value (%)",
        "CI Lower (%)",
        "CI Upper (%)",
        "Successes",
        "Total"
    ]
]

key_ci_path = path.join(
    phase5_summary_dir,
    "phase5_key_metric_confidence_intervals_for_paper.csv"
)

key_ci_df.to_csv(key_ci_path, index=False)

print("\nPaper-ready key CI table saved:")
print(key_ci_path)

display(key_ci_df)


key_mcnemar_df = mcnemar_df[
    [
        "Dataset",
        "Model A",
        "Model B",
        "A Correct B Wrong",
        "A Wrong B Correct",
        "Discordant Pairs",
        "Test Type",
        "p-value",
        "Significant at 0.05"
    ]
].copy()

key_mcnemar_path = path.join(
    phase5_tests_dir,
    "phase5_key_mcnemar_tests_for_paper.csv"
)

key_mcnemar_df.to_csv(key_mcnemar_path, index=False)

print("\nPaper-ready McNemar table saved:")
print(key_mcnemar_path)

display(key_mcnemar_df)


key_ttest_df = ttest_df[
    [
        "Model A",
        "Model B",
        "Metric",
        "Model A Mean",
        "Model B Mean",
        "Mean Difference A-B",
        "t-statistic",
        "p-value",
        "Significant at 0.05"
    ]
].copy()

key_ttest_path = path.join(
    phase5_tests_dir,
    "phase5_key_paired_ttests_for_paper.csv"
)

key_ttest_df.to_csv(key_ttest_path, index=False)

print("\nPaper-ready paired t-test table saved:")
print(key_ttest_path)

display(key_ttest_df)


print("\n" + "=" * 80)
print("PHASE 5 STATISTICAL VALIDATION COMPLETED")
print("=" * 80)
print("Outputs:")
print("1.", wilson_path)
print("2.", mcnemar_path)
print("3.", cv_results_path)
print("4.", cv_summary_path)
print("5.", ttest_path)
print("6.", key_ci_path)
print("7.", key_mcnemar_path)
print("8.", key_ttest_path)