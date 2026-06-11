# =========================================================
# PHASE 3 → FEATURE SELECTION
# RF + XGBoost Combined Importance
# =========================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE2B_DIR,
    PHASE3_DIR,
    FINAL_DIR
)

from utils import make_dir

from data_loader import (
    build_training_features,
    build_dualuse_features,
    prepare_xy
)

from evaluation import (
    evaluate_models_modular
)

# =========================================================
# LOAD MERGED DATASETS
# =========================================================

_, training_merged_df = build_training_features()

_, dual_merged_X = build_dualuse_features()

# =========================================================
# PHASE 3 DIRECTORIES
# =========================================================

phase3_dir = PHASE3_DIR

feature_rankings_dir = make_dir(
    phase3_dir / "feature_rankings"
)

top_n_root_dir = make_dir(
    phase3_dir / "top_n_results"
)

summary_dir = make_dir(
    phase3_dir / "summaries"
)

print("\n")
print("=" * 80)
print("PHASE 3 → FEATURE SELECTION")
print("=" * 80)

# =========================================================
# BUILD TRAINING FEATURES
# =========================================================

X_merged, y_merged = prepare_xy(
    training_merged_df
)

missing_cols = (
    set(X_merged.columns)
    - set(dual_merged_X.columns)
)

if missing_cols:

    raise ValueError(
        f"Dual-use merged dataset missing columns: "
        f"{sorted(missing_cols)}"
    )

dual_merged_X = dual_merged_X[
    X_merged.columns
]

assert (
    list(X_merged.columns)
    == list(dual_merged_X.columns)
)

print(
    f"Original merged feature count: "
    f"{X_merged.shape[1]}"
)

print(
    f"Dual-use merged feature count: "
    f"{dual_merged_X.shape[1]}"
)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_merged,
    y_merged,
    test_size=0.2,
    stratify=y_merged,
    random_state=42
)

# =========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

print("\nTraining Random Forest for feature importance...")

rf_selector = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_selector.fit(
    X_train,
    y_train
)

rf_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "rf_importance": rf_selector.feature_importances_
}).sort_values(
    by="rf_importance",
    ascending=False
)

rf_importance_path = (
    feature_rankings_dir
    / "rf_feature_importance.csv"
)

rf_importance_df.to_csv(
    rf_importance_path,
    index=False
)

# =========================================================
# XGBOOST FEATURE IMPORTANCE
# =========================================================

print("\nTraining XGBoost for feature importance...")

xgb_selector = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb_selector.fit(
    X_train,
    y_train
)

xgb_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "xgb_importance": xgb_selector.feature_importances_
}).sort_values(
    by="xgb_importance",
    ascending=False
)

xgb_importance_path = (
    feature_rankings_dir
    / "xgb_feature_importance.csv"
)

xgb_importance_df.to_csv(
    xgb_importance_path,
    index=False
)

# =========================================================
# COMBINED IMPORTANCE
# =========================================================

importance_df = pd.merge(
    rf_importance_df,
    xgb_importance_df,
    on="feature",
    how="inner"
)

importance_df["combined_score"] = (
    importance_df["rf_importance"]
    + importance_df["xgb_importance"]
)

importance_df = importance_df.sort_values(
    by="combined_score",
    ascending=False
)

combined_importance_path = (
    feature_rankings_dir
    / "combined_rf_xgb_feature_importance.csv"
)

importance_df.to_csv(
    combined_importance_path,
    index=False
)

print("\nSaved feature rankings:")
print(rf_importance_path)
print(xgb_importance_path)
print(combined_importance_path)

# =========================================================
# TOP-N FEATURE EVALUATION
# =========================================================

candidate_top_n = [
    100,
    250,
    500,
    1000
]

top_n_list = [
    n
    for n in candidate_top_n
    if n <= X_merged.shape[1]
]

print(
    f"\nRunning feature selection for: "
    f"{top_n_list}"
)

phase3_results = []

for top_n in top_n_list:

    print("\n")
    print("=" * 80)
    print(f"PHASE 3 RUNNING → TOP {top_n} FEATURES")
    print("=" * 80)

    top_n_dir = make_dir(
        top_n_root_dir
        / f"top_{top_n}"
    )

    make_dir(top_n_dir / "metrics")
    make_dir(top_n_dir / "predictions")

    make_dir(top_n_dir / "curves")
    make_dir(top_n_dir / "curves" / "roc")
    make_dir(top_n_dir / "curves" / "pr")

    make_dir(top_n_dir / "selected_features")

    # -----------------------------------------------------
    # SELECT TOP FEATURES
    # -----------------------------------------------------

    selected_df = (
        importance_df
        .head(top_n)
        .copy()
    )

    selected_features = (
        selected_df["feature"]
        .tolist()
    )

    selected_features_path = (
        top_n_dir
        / "selected_features"
        / f"top_{top_n}_selected_features.csv"
    )

    selected_df.to_csv(
        selected_features_path,
        index=False
    )

    # -----------------------------------------------------
    # BUILD FEATURE-SELECTED DATASETS
    # -----------------------------------------------------

    X_fs = X_merged[
        selected_features
    ]

    dual_fs = dual_merged_X[
        selected_features
    ]

    assert (
        list(X_fs.columns)
        == list(dual_fs.columns)
    )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    result_df = evaluate_models_modular(
        X=X_fs,
        y=y_merged,
        dataset_name=f"Top {top_n} Features",
        phase_dir=top_n_dir,
        dualuse_X=dual_fs,
        extra_metadata={
            "Phase": "Phase 3",
            "Experiment Type": "Feature Selection",
            "Feature Selection Method":
                "RF + XGBoost Combined Importance",
            "Top Features": top_n,
            "Selected Features File":
                str(selected_features_path)
        }
    )

    phase3_results.append(
        result_df
    )

# =========================================================
# PHASE 3 SUMMARY
# =========================================================

phase3_summary_df = pd.concat(
    phase3_results,
    ignore_index=True
)

phase3_summary_path = (
    summary_dir
    / "phase3_feature_selection_summary.csv"
)

phase3_summary_df.to_csv(
    phase3_summary_path,
    index=False
)

print("\nPhase 3 summary saved:")
print(phase3_summary_path)

# =========================================================
# FINAL COMBINED SUMMARY
# =========================================================

final_summary_dir = make_dir(
    FINAL_DIR
)

summary_files = [
    PHASE1_DIR
    / "summaries"
    / "phase1_individual_feature_summary.csv",

    PHASE2_DIR
    / "summaries"
    / "phase2_merged_features_summary.csv",

    PHASE2B_DIR
    / "summaries"
    / "phase2b_ablation_summary.csv",

    PHASE3_DIR
    / "summaries"
    / "phase3_feature_selection_summary.csv"
]

summary_dfs = []

for summary_file in summary_files:

    if summary_file.exists():

        summary_dfs.append(
            pd.read_csv(summary_file)
        )

    else:

        print(
            "Missing summary:",
            summary_file
        )

if not summary_dfs:
    raise ValueError(
        "No phase summaries found."
    )

final_results_df = pd.concat(
    summary_dfs,
    ignore_index=True
)

final_results_path = (
    final_summary_dir
    / "all_phases_combined_summary.csv"
)

final_results_df.to_csv(
    final_results_path,
    index=False
)

print("\nFinal combined summary saved:")
print(final_results_path)

print(
    "\nFinal results shape:",
    final_results_df.shape
)