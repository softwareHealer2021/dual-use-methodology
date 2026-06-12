# =========================================================
# PHASE 4 → CROSS-DATASET VALIDATION
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    PHASE4_DIR
)

from utils import (
    make_dir
)

from data_loader import (
    load_file,
    build_training_features,
    build_external_feature_matrix,
    prepare_xy
)

from evaluation import (
    evaluate_models_phase4_cross_dataset
)

# =========================================================
# DIRECTORIES
# =========================================================

phase4_dir = PHASE4_DIR

make_dir(phase4_dir)

print("\n")
print("=" * 80)
print("PHASE 4 → CROSS-DATASET VALIDATION")
print("=" * 80)

# =========================================================
# LOAD TRAINING DATA
# =========================================================

_, merged_df = build_training_features()

X_merged, y_merged = prepare_xy(
    merged_df
)

print("\nTraining dataset:")
print("X_merged:", X_merged.shape)
print("y_merged:", y_merged.shape)

# =========================================================
# CROSS-DATASET FEATURE DIRECTORY
# =========================================================

cross_dataset_dir = Path(
    "data/cross_validation"
)

print("\nCross-dataset folder:")
print(cross_dataset_dir)

# =========================================================
# LOAD VERA DATASETS
# =========================================================

vera_header_df = load_file(
    cross_dataset_dir
    / "vera_active_header_features.csv"
)

vera_dll_df = load_file(
    cross_dataset_dir
    / "vera_active_dll_features.csv"
)

vera_function_df = load_file(
    cross_dataset_dir
    / "vera_active_function_features.csv"
)

vera_entropy_df = load_file(
    cross_dataset_dir
    / "vera_active_entropy_features.csv"
)

# =========================================================
# LOAD DIKE DATASETS
# =========================================================

dike_header_df = load_file(
    cross_dataset_dir
    / "dike_header_features.csv"
)

dike_dll_df = load_file(
    cross_dataset_dir
    / "dike_dll_features.csv"
)

dike_function_df = load_file(
    cross_dataset_dir
    / "dike_function_features.csv"
)

dike_entropy_df = load_file(
    cross_dataset_dir
    / "dike_entropy_features.csv"
)

# =========================================================
# METADATA
# =========================================================

if "filename" in vera_header_df.columns:

    vera_meta = (
        vera_header_df[["filename"]]
        .reset_index(drop=True)
    )

else:

    vera_meta = pd.DataFrame({
        "filename": [
            f"vera_sample_{i}"
            for i in range(len(vera_header_df))
        ]
    })

if "filename" in dike_header_df.columns:

    dike_meta = (
        dike_header_df[["filename"]]
        .reset_index(drop=True)
    )

else:

    dike_meta = pd.DataFrame({
        "filename": [
            f"dike_sample_{i}"
            for i in range(len(dike_header_df))
        ]
    })

# =========================================================
# BUILD EXTERNAL FEATURE MATRICES
# =========================================================

vera_merged_X = build_external_feature_matrix(
    header_df=vera_header_df,
    dll_df=vera_dll_df,
    function_df=vera_function_df,
    entropy_df=vera_entropy_df,
    training_columns=X_merged.columns
)

dike_merged_X = build_external_feature_matrix(
    header_df=dike_header_df,
    dll_df=dike_dll_df,
    function_df=dike_function_df,
    entropy_df=dike_entropy_df,
    training_columns=X_merged.columns
)

y_vera = np.ones(
    len(vera_merged_X),
    dtype=int
)

y_dike = np.zeros(
    len(dike_merged_X),
    dtype=int
)

print("\nExternal merged shapes:")
print("VERA :", vera_merged_X.shape)
print("DIKE :", dike_merged_X.shape)

assert list(
    vera_merged_X.columns
) == list(
    X_merged.columns
)

assert list(
    dike_merged_X.columns
) == list(
    X_merged.columns
)

# =========================================================
# RUN PHASE 4
# =========================================================

phase4_result_df = (
    evaluate_models_phase4_cross_dataset(
        X=X_merged,
        y=y_merged,
        dataset_name="Merged Features Cross Dataset",
        phase_dir=phase4_dir,

        vera_X=vera_merged_X,
        vera_y=y_vera,
        vera_meta=vera_meta,

        dike_X=dike_merged_X,
        dike_y=y_dike,
        dike_meta=dike_meta,

        extra_metadata={
            "Phase": "Phase 4",
            "Experiment Type":
                "Cross-Dataset Validation",
            "Feature Setting":
                "Merged Features",
            "Training Dataset":
                "Original Baseline Dataset",
            "External Ransomware Dataset":
                "VERA Active",
            "External Benign Dataset":
                "DikeDataset Benign PE"
        }
    )
)

# =========================================================
# SAVE SUMMARY
# =========================================================

summary_dir = make_dir(
    phase4_dir / "summaries"
)

phase4_summary_path = (
    summary_dir
    / "phase4_cross_dataset_summary.csv"
)

phase4_result_df.to_csv(
    phase4_summary_path,
    index=False
)

print("\nPhase 4 completed.")
print("Summary saved:")
print(phase4_summary_path)