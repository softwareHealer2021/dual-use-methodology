import itertools
import pandas as pd

from config import PHASE2B_DIR

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
# LOAD DATA
# =========================================================

training_datasets, _ = (
    build_training_features()
)

dual_datasets, _ = (
    build_dualuse_features()
)

# =========================================================
# PHASE 2B → ABLATION STUDY
# =========================================================

phase2b_dir = PHASE2B_DIR

make_dir(phase2b_dir)

print("\n")
print("=" * 80)
print("PHASE 2B → ABLATION STUDY")
print("=" * 80)

feature_family_names = [
    "header",
    "dll",
    "function",
    "entropy"
]

# ---------------------------------------------------------
# Generate all 2-family combinations
# ---------------------------------------------------------

ablation_combinations = list(
    itertools.combinations(
        feature_family_names,
        2
    )
)

# ---------------------------------------------------------
# Generate all 3-family combinations
# ---------------------------------------------------------

ablation_combinations.extend(
    itertools.combinations(
        feature_family_names,
        3
    )
)

phase2b_results = []

# =========================================================
# RUN ALL COMBINATIONS
# =========================================================

for selected_families in ablation_combinations:

    dataset_name = " + ".join(
        family.title()
        for family in selected_families
    )

    print("\n")
    print("=" * 80)
    print(f"PHASE 2B RUNNING → {dataset_name}")
    print("=" * 80)

    # -----------------------------------------------------
    # Build training combination
    # -----------------------------------------------------

    combo_df = pd.concat(
        [
            training_datasets[family].reset_index(drop=True)
            for family in selected_families
        ],
        axis=1
    )

    # Add metadata back so prepare_xy works

    training_merged_df = pd.concat(
        [
            pd.read_csv(
                "data/training/header_constant_removed.csv"
            )[["ID", "RG", "filename"]]
            .reset_index(drop=True),

            combo_df
        ],
        axis=1
    )

    X_combo, y_combo = prepare_xy(
        training_merged_df
    )

    # -----------------------------------------------------
    # Build dual-use combination
    # -----------------------------------------------------

    dual_combo_X = pd.concat(
        [
            dual_datasets[family].reset_index(drop=True)
            for family in selected_families
        ],
        axis=1
    )

    missing_cols = (
        set(X_combo.columns)
        - set(dual_combo_X.columns)
    )

    if missing_cols:

        raise ValueError(
            f"Missing dual-use columns: "
            f"{sorted(missing_cols)}"
        )

    dual_combo_X = dual_combo_X[
        X_combo.columns
    ]

    assert (
        list(dual_combo_X.columns)
        == list(X_combo.columns)
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    result_df = evaluate_models_modular(
        X=X_combo,
        y=y_combo,
        dataset_name=dataset_name,
        phase_dir=phase2b_dir,
        dualuse_X=dual_combo_X,
        extra_metadata={
            "Phase": "Phase 2B",
            "Experiment Type": "Feature Family Ablation",
            "Feature Families": dataset_name,
            "Feature Family Count": len(
                selected_families
            )
        }
    )

    phase2b_results.append(
        result_df
    )

# =========================================================
# SAVE SUMMARY
# =========================================================

phase2b_summary_df = pd.concat(
    phase2b_results,
    ignore_index=True
)

summary_dir = make_dir(
    phase2b_dir / "summaries"
)

phase2b_summary_path = (
    summary_dir
    / "phase2b_ablation_summary.csv"
)

phase2b_summary_df.to_csv(
    phase2b_summary_path,
    index=False
)

print("\nPhase 2B summary saved:")
print(phase2b_summary_path)