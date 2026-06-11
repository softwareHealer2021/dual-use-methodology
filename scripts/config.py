from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path.cwd()

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
PHASE3_DIR = RESULTS_DIR / "phase3_feature_selection"
FINAL_DIR = RESULTS_DIR / "final_summary"