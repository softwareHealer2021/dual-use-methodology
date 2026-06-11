import pandas as pd

from config import PHASE2_DIR

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

training_datasets, training_merged_df = (
    build_training_features()
)

dual_datasets, dual_merged_X = (
    build_dualuse_features()
)

# =========================================================
# PHASE 2 → FULL FEATURE FUSION
# =========================================================

phase2_dir = PHASE2_DIR

make_dir(phase2_dir)

print("\n")
print("=" * 80)
print("PHASE 2 → FULL MERGED FEATURES")
print("=" * 80)

# =========================================================
# BUILD TRAINING X/y
# =========================================================

X_merged, y_merged = prepare_xy(
    training_merged_df
)

# =========================================================
# ALIGN DUAL-USE FEATURES
# =========================================================

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
    list(dual_merged_X.columns)
    == list(X_merged.columns)
)

# =========================================================
# RUN EVALUATION
# =========================================================

phase2_result_df = evaluate_models_modular(
    X=X_merged,
    y=y_merged,
    dataset_name="Merged Features",
    phase_dir=phase2_dir,
    dualuse_X=dual_merged_X,
    extra_metadata={
        "Phase": "Phase 2",
        "Experiment Type": "Full Feature Fusion",
        "Feature Family Count": 4
    }
)

# =========================================================
# SAVE SUMMARY
# =========================================================

summary_dir = make_dir(
    phase2_dir / "summaries"
)

phase2_summary_path = (
    summary_dir
    / "phase2_merged_features_summary.csv"
)

phase2_result_df.to_csv(
    phase2_summary_path,
    index=False
)

print("\nPhase 2 summary saved:")
print(phase2_summary_path)