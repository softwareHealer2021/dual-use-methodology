# Dual-Use Methodology for Ransomware Detection

This repository implements a multi-phase machine learning pipeline designed to evaluate feature families, perform feature selection, run cross-dataset validation, analyze models using SHAP (SHapley Additive exPlanations), and evaluate confidence-gated dual-stage detection on ransomware vs benign and dual-use datasets.

---

## Data Sources & Links

The project utilizes datasets for training, dual-use validation, and external cross-dataset evaluation. Please download the datasets from the following links:

*   **Preprocessed Features Dataset**: [Google Drive Folder](https://drive.google.com/drive/folders/1OiwYOi4FEkJsMHHBoqwuxEmBQkqWCbVF)  
    *Contains the preprocessed CSV feature matrices extracted for training, dual-use, and external validation.*
*   **Baseline PE Dataset**: [Mendeley Data](https://data.mendeley.com/datasets/yzhcvn7sj5/2)  
    *The original baseline dataset of benign and ransomware PE files used for initial feature extraction.*
*   **VERA Active Binary Dataset**: [Zenodo](https://zenodo.org/records/17904705)  
    *External active ransomware PE samples used for cross-dataset validation.*
*   **Dike Binary Dataset (Benign)**: [GitHub Repository](https://github.com/iosifache/DikeDataset/tree/main/files/benign)  
    *External benign PE samples used for cross-dataset validation.*

---

## Directory Structure

To run the pipeline successfully, ensure your local project structure matches the following:

```text
dual-use-methodology/
├── data/                                 # Raw and preprocessed datasets (create if missing)
│   ├── training/                         # Baseline training CSV files
│   │   ├── header_constant_removed.csv
│   │   ├── dll_constant_removed.csv
│   │   ├── function_constant_removed.csv
│   │   └── entropy_constant_removed.csv
│   ├── dualuse/                          # Dual-use benign encryption CSV files
│   │   ├── dualuse_header_features.csv
│   │   ├── dualuse_dll_features.csv
│   │   ├── dualuse_function_features.csv
│   │   └── dualuse_entropy_features.csv
│   └── cross_validation/                 # External evaluation CSV files
│       ├── vera_active_header_features.csv
│       ├── vera_active_dll_features.csv
│       ├── vera_active_function_features.csv
│       ├── vera_active_entropy_features.csv
│       ├── dike_header_features.csv
│       ├── dike_dll_features.csv
│       ├── dike_function_features.csv
│       └── dike_entropy_features.csv
├── results/                              # Pipeline outputs (auto-generated)
│   ├── phase1_individual/                # Individual feature family evaluations
│   ├── phase2_merged/                    # Merged feature family evaluations
│   ├── phase2b_ablation/                 # Feature family ablation studies
│   ├── phase3_feature_selection/         # Feature selection metrics and evaluation
│   ├── phase4_cross_dataset/             # Cross-dataset evaluation metrics and predictions
│   ├── phase5_statistical_validation/    # K-fold cross-validation and statistical tests
│   ├── phase6_shap_analysis/             # SHAP local/global importance plots and tables
│   └── phase7_confidence_gating/         # Confidence gating results, ROC/PR curves
├── scripts/                              # Pipeline source files
│   ├── config.py                         # Global path configurations and variables
│   ├── data_loader.py                    # File loading and feature matrix alignment functions
│   ├── evaluation.py                     # Evaluation functions, metric computations, and plotting
│   ├── stats_utils.py                    # Statistical testing helper functions
│   ├── utils.py                          # General file system and utility tools
│   ├── phase1_individual.py              # Evaluates individual feature families
│   ├── phase2_merged.py                  # Evaluates merged feature sets
│   ├── phase2_ablation.py                # Evaluates feature family ablation sets
│   ├── phase3_feature_selection.py       # Executes feature selection analysis
│   ├── phase4_cross_validation.py        # Runs cross-dataset model evaluations
│   ├── phase5_statistical_validation.py  # Performs cross-validation and significance testing
│   ├── phase6_shap.py                    # Generates SHAP explanation tables and plots
│   └── phase7_confidence_gated.py        # Simulates confidence-gated dual-stage detection
├── requirements.txt                      # Project package requirements
└── README.md                             # Project documentation
```

---

## Setup & Installation

### Prerequisites
*   Python **`3.12.1`** (recommended)

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd dual-use-methodology
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **Windows (Command Prompt)**:
        ```cmd
        venv\Scripts\activate.bat
        ```
    *   **Windows (PowerShell)**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **Linux / macOS**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## Data Preparation

1.  Download the preprocessed feature CSVs from the **Preprocessed Features Dataset** link.
2.  Place the files in their corresponding directories under `data/` as shown in the **Directory Structure** section:
    *   Place baseline datasets under `data/training/`
    *   Place dual-use datasets under `data/dualuse/`
    *   Place VERA active and Dike datasets under `data/cross_validation/`

---

## Running the Pipeline

The pipeline is split into individual phases. You can run them sequentially to produce the outputs under `results/`:

*   **Phase 1: Individual Feature Family Evaluation**
    Evaluates individual feature families (Header, DLL, Function, Entropy) using multiple machine learning classifiers.
    ```bash
    python scripts/phase1_individual.py
    ```

*   **Phase 2: Merged Dataset Evaluation**
    Evaluates classifier performance when all feature families are combined.
    ```bash
    python scripts/phase2_merged.py
    ```

*   **Phase 2 Ablation: Feature Ablation Study**
    Systematically ablates single feature families to evaluate their contribution to overall model performance.
    ```bash
    python scripts/phase2_ablation.py
    ```

*   **Phase 3: Feature Selection Analysis**
    Performs feature selection and evaluates the impact of feature subset sizes on classification performance.
    ```bash
    python scripts/phase3_feature_selection.py
    ```

*   **Phase 4: Cross-Dataset Validation**
    Trains models on the baseline dataset and evaluates generalization on external datasets (VERA Active and DikeDataset Benign).
    ```bash
    python scripts/phase4_cross_validation.py
    ```

*   **Phase 5: Statistical Validation**
    Performs K-fold cross-validation and statistical significance testing (e.g., McNemar's test) on models.
    ```bash
    python scripts/phase5_statistical_validation.py
    ```

*   **Phase 6: SHAP Explanation Analysis**
    Computes global feature importances, local decision waterfalls, and class-wise explanations. (Requires the `shap` library).
    ```bash
    python scripts/phase6_shap.py
    ```

*   **Phase 7: Confidence-Gated Dual-Stage Detection**
    Evaluates zero-shot, calibrated, and holdout-tested confidence gating thresholds to reduce false positives on dual-use tools while maintaining high detection sensitivity.
    ```bash
    python scripts/phase7_confidence_gated.py
    ```
