from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================================================
# DATA
# =========================================================

DATA_DIR = PROJECT_ROOT / "data"

TRAINING_DIR = DATA_DIR / "training"
DUALUSE_DIR = DATA_DIR / "dualuse"

# Training datasets
HEADER_DATASET = TRAINING_DIR / "header_constant_removed.csv"
DLL_DATASET = TRAINING_DIR / "dll_constant_removed.csv"
FUNCTION_DATASET = TRAINING_DIR / "function_constant_removed.csv"
ENTROPY_DATASET = TRAINING_DIR / "entropy_constant_removed.csv"

# Dual-use datasets
DUAL_HEADER_DATASET = DUALUSE_DIR / "dualuse_header_features.csv"
DUAL_DLL_DATASET = DUALUSE_DIR / "dualuse_dll_features.csv"
DUAL_FUNCTION_DATASET = DUALUSE_DIR / "dualuse_function_features.csv"
DUAL_ENTROPY_DATASET = DUALUSE_DIR / "dualuse_entropy_features.csv"

# =========================================================
# RESULTS
# =========================================================

RESULTS_DIR = PROJECT_ROOT / "results"

PHASE1_DIR = RESULTS_DIR / "phase1_individual"
PHASE2_DIR = RESULTS_DIR / "phase2_merged"
PHASE2B_DIR = RESULTS_DIR / "phase2b_ablation"
PHASE2_PREDICTIONS_DIR = PHASE2_DIR / "predictions"
PHASE3_DIR = RESULTS_DIR / "phase3_feature_selection"
PHASE4_DIR = RESULTS_DIR / "phase4_cross_dataset"
CROSS_DATASET_DIR = (
    DATA_DIR / "cross_validation"
)
PHASE4_PREDICTIONS_DIR = PHASE4_DIR / "predictions"
PHASE5_DIR = RESULTS_DIR / "phase5_statistical_validation"
PHASE5_SUMMARIES_DIR = (
    PHASE5_DIR / "summaries"
)

PHASE5_TESTS_DIR = (
    PHASE5_DIR / "tests"
)

PHASE5_CV_DIR = (
    PHASE5_DIR / "kfold_cv"
)
PHASE6_DIR = RESULTS_DIR / "phase6_shap_analysis"
PHASE6_MERGED_XGBOOST_DIR = (
    PHASE6_DIR / "merged_xgboost"
)
PHASE6_NO_ENTROPY_XGBOOST_DIR = (
    PHASE6_DIR / "all_minus_entropy_xgboost"
)
PHASE7_DIR = RESULTS_DIR / "phase7_confidence_gating"
PHASE7_METRICS_DIR = (
    PHASE7_DIR / "metrics"
)
PHASE7_THRESHOLD_DIR = (
    PHASE7_DIR / "threshold_grids"
)
PHASE7_PREDICTIONS_DIR = (
    PHASE7_DIR / "predictions"
)
PHASE7_CURVES_DIR = (
    PHASE7_DIR / "curves"
)
PHASE7_ROC_DIR = (
    PHASE7_CURVES_DIR / "roc"
)
PHASE7_PR_DIR = (
    PHASE7_CURVES_DIR / "pr"
)
PHASE7_SUMMARIES_DIR = (
    PHASE7_DIR / "summaries"
)
FINAL_DIR = RESULTS_DIR / "final_summary"
