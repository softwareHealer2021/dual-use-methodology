from pathlib import Path
import pandas as pd
import numpy as np

from config import TRAINING_DIR, DUALUSE_DIR


def load_file(file_path):
    file_path = Path(file_path)

    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)

    elif file_path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file format: {file_path}")


def prepare_xy(df):
    df["family"] = (df["ID"] >= 20000).astype(int)

    drop_cols = [
        c for c in ["ID", "family", "RG", "filename"]
        if c in df.columns
    ]

    X = df.drop(columns=drop_cols)
    y = df["family"]

    return X, y


def load_training_datasets():

    return {
        "header": load_file(
            TRAINING_DIR / "header_constant_removed.csv"
        ),

        "dll": load_file(
            TRAINING_DIR / "dll_constant_removed.csv"
        ),

        "function": load_file(
            TRAINING_DIR / "function_constant_removed.csv"
        ),

        "entropy": load_file(
            TRAINING_DIR / "entropy_constant_removed.csv"
        )
    }


def load_dualuse_datasets():

    return {
        "header": load_file(
            DUALUSE_DIR / "dualuse_header_features.csv"
        ),

        "dll": load_file(
            DUALUSE_DIR / "dualuse_dll_features.csv"
        ),

        "function": load_file(
            DUALUSE_DIR / "dualuse_function_features.csv"
        ),

        "entropy": load_file(
            DUALUSE_DIR / "dualuse_entropy_features.csv"
        )
    }


def build_dualuse_features():

    datasets = load_dualuse_datasets()

    drop_cols = ["filename", "RG", "family", "ID"]

    cleaned = {}

    for name, df in datasets.items():

        cleaned[name] = df.drop(
            columns=[
                c for c in drop_cols
                if c in df.columns
            ],
            errors="ignore"
        )

    merged = pd.concat(
        [
            cleaned["header"],
            cleaned["dll"],
            cleaned["function"],
            cleaned["entropy"]
        ],
        axis=1
    )

    return cleaned, merged


def build_training_features():
    """
    Returns:
        cleaned_datasets -> dict of individual feature families
        merged_df -> all feature families concatenated
    """

    datasets = load_training_datasets()

    drop_cols = [
        "filename",
        "RG",
        "family",
        "ID"
    ]

    cleaned = {}

    for name, df in datasets.items():

        cleaned[name] = df.drop(
            columns=[
                c for c in drop_cols
                if c in df.columns
            ],
            errors="ignore"
        )

    meta_cols = [
        c
        for c in ["ID", "RG", "filename"]
        if c in datasets["header"].columns
    ]

    merged_df = pd.concat(
        [
            datasets["header"][meta_cols].reset_index(drop=True),

            cleaned["header"].reset_index(drop=True),
            cleaned["dll"].reset_index(drop=True),
            cleaned["function"].reset_index(drop=True),
            cleaned["entropy"].reset_index(drop=True)
        ],
        axis=1
    )

    return cleaned, merged_df

def build_external_feature_matrix(
    header_df,
    dll_df,
    function_df,
    entropy_df,
    training_columns
):
    drop_cols = [
        "ID",
        "RG",
        "filename",
        "family",
        "label"
    ]

    header_X = header_df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    dll_X = dll_df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    function_X = function_df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    entropy_X = entropy_df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    merged_X = pd.concat(
        [
            header_X.reset_index(drop=True),
            dll_X.reset_index(drop=True),
            function_X.reset_index(drop=True),
            entropy_X.reset_index(drop=True)
        ],
        axis=1
    )

    merged_X = merged_X.reindex(
        columns=training_columns,
        fill_value=0
    )

    return merged_X