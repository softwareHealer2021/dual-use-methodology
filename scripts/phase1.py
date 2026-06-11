import pandas as pd

from config import PHASE1_DIR

from data_loader import (
    load_training_datasets,
    build_dualuse_features,
    prepare_xy
)

from evaluation import (
    evaluate_models_modular
)

# =========================================================
# LOAD DATA
# =========================================================

datasets = load_training_datasets()

dual_datasets, dual_merged_X = build_dualuse_features()

# =========================================================
# PHASE 1
# =========================================================

phase1_results = []

phase1_datasets = {
    "Header Only": (
        datasets["header"],
        dual_datasets["header"]
    ),

    "DLL Only": (
        datasets["dll"],
        dual_datasets["dll"]
    ),

    "Function Only": (
        datasets["function"],
        dual_datasets["function"]
    ),

    "Entropy Only": (
        datasets["entropy"],
        dual_datasets["entropy"]
    )
}

for dataset_name, (main_df, dualuse_df) in phase1_datasets.items():

    print("\n")
    print("=" * 80)
    print(f"PHASE 1 RUNNING → {dataset_name}")
    print("=" * 80)

    X, y = prepare_xy(main_df)

    dualuse_X = dualuse_df[X.columns]

    result_df = evaluate_models_modular(
        X=X,
        y=y,
        dataset_name=dataset_name,
        phase_dir=PHASE1_DIR,
        dualuse_X=dualuse_X,
        extra_metadata={
            "Phase": "Phase 1",
            "Experiment Type": "Individual Feature Family",
            "Feature Family Count": 1
        }
    )

    phase1_results.append(result_df)

# =========================================================
# SAVE SUMMARY
# =========================================================

phase1_summary_df = pd.concat(
    phase1_results,
    ignore_index=True
)

summary_path = (
    PHASE1_DIR
    / "summaries"
    / "phase1_individual_feature_summary.csv"
)

phase1_summary_df.to_csv(
    summary_path,
    index=False
)

print("\nPhase 1 summary saved:")
print(summary_path)