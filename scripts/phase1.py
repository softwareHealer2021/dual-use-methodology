import pandas as pd

from config import PHASE1_DIR

from utils import make_dir

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

phase1_dir = PHASE1_DIR

make_dir(phase1_dir)

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

    print("\n" + "=" * 80)
    print(f"PHASE 1 RUNNING → {dataset_name}")
    print("=" * 80)

    X, y = prepare_xy(main_df)

    missing = set(X.columns) - set(dualuse_df.columns)

    if missing:
        raise ValueError(
            f"Missing dual-use columns: {missing}"
        )

    dualuse_X = dualuse_df[X.columns]

    result_df = evaluate_models_modular(
        X=X,
        y=y,
        dataset_name=dataset_name,
        phase_dir=phase1_dir,
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

summary_dir = make_dir(
    phase1_dir / "summaries"
)

phase1_summary_path = (
    summary_dir
    / "phase1_individual_feature_summary.csv"
)

phase1_summary_df.to_csv(
    phase1_summary_path,
    index=False
)

print("\nPhase 1 summary saved:")
print(phase1_summary_path)