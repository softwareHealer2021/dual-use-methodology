# =========================================================
# PHASE 6 -> SHAP ANALYSIS FOR MERGED XGBOOST MODELS
#
# Saves quantitative tables and publication-ready plots under:
#   results/phase6_shap_analysis/
# =========================================================

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from os import path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    PHASE6_DIR,
    PHASE6_MERGED_XGBOOST_DIR,
    PHASE6_NO_ENTROPY_XGBOOST_DIR
)
from data_loader import (
    build_training_features,
    build_dualuse_features,
    prepare_xy
)
from utils import make_dir


RANDOM_STATE = 42
IEEE_DOUBLE_COL_WIDTH = 7.16
TOP_FEATURES = 20


def ensure_phase_dirs():
    for directory in [
        PHASE6_DIR,
        PHASE6_MERGED_XGBOOST_DIR,
        PHASE6_NO_ENTROPY_XGBOOST_DIR,
        PHASE6_MERGED_XGBOOST_DIR / "plots",
        PHASE6_MERGED_XGBOOST_DIR / "tables",
        PHASE6_NO_ENTROPY_XGBOOST_DIR / "plots",
        PHASE6_NO_ENTROPY_XGBOOST_DIR / "tables",
    ]:
        make_dir(directory)


def align_dualuse_features(X, dual_X):
    missing_cols = set(X.columns) - set(dual_X.columns)

    if missing_cols:
        raise ValueError(
            "Dual-use dataset missing columns: "
            f"{sorted(missing_cols)}"
        )

    return dual_X[X.columns]


def build_ablation_training_df(training_datasets, training_merged_df, families):
    meta_cols = [
        c
        for c in ["ID", "RG", "filename"]
        if c in training_merged_df.columns
    ]

    feature_df = pd.concat(
        [
            training_datasets[family].reset_index(drop=True)
            for family in families
        ],
        axis=1
    )

    return pd.concat(
        [
            training_merged_df[meta_cols].reset_index(drop=True),
            feature_df
        ],
        axis=1
    )


def build_ablation_dualuse_df(dual_datasets, families):
    return pd.concat(
        [
            dual_datasets[family].reset_index(drop=True)
            for family in families
        ],
        axis=1
    )


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def shap_values_to_array(shap_values):
    values = shap_values.values

    if values.ndim == 3:
        return values[:, :, 1]

    return values


def save_global_importance(shap_array, X, output_dir, file_prefix):
    global_importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_array).mean(axis=0),
        "mean_shap": shap_array.mean(axis=0)
    }).sort_values(
        by="mean_abs_shap",
        ascending=False
    )

    global_importance_df["rank"] = range(
        1,
        len(global_importance_df) + 1
    )

    output_path = path.join(
        output_dir,
        "tables",
        f"{file_prefix}_global_shap_importance.csv"
    )

    global_importance_df.to_csv(output_path, index=False)

    print("Saved global SHAP importance:")
    print(output_path)

    print("\nTop 20 global SHAP features:")
    print(global_importance_df.head(TOP_FEATURES))

    return global_importance_df


def save_summary_plot(shap_values, X, output_dir, file_prefix):
    plt.figure(figsize=(IEEE_DOUBLE_COL_WIDTH, 5.5))

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        max_display=TOP_FEATURES,
        plot_size=None
    )

    ax = plt.gca()
    ax.set_title("")
    ax.set_xlabel(
        "SHAP Value Impact on Model Output",
        fontsize=15,
        fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    base_path = path.join(
        output_dir,
        "plots",
        f"{file_prefix}_global_shap_summary_publication"
    )

    save_current_figure(base_path)

    print("Saved publication-quality SHAP summary plot:")
    print(base_path + ".png")
    print(base_path + ".pdf")
    print(base_path + ".svg")


def save_bar_plot(shap_values, X, output_dir, file_prefix):
    plt.figure(figsize=(IEEE_DOUBLE_COL_WIDTH, 5.2))

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        max_display=TOP_FEATURES,
        plot_type="bar",
        plot_size=None
    )

    ax = plt.gca()
    ax.set_title("")
    ax.set_xlabel(
        "Mean Absolute SHAP Value",
        fontsize=15,
        fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    base_path = path.join(
        output_dir,
        "plots",
        f"{file_prefix}_global_shap_bar_publication"
    )

    save_current_figure(base_path)

    print("Saved publication-quality SHAP bar plot:")
    print(base_path + ".png")
    print(base_path + ".pdf")
    print(base_path + ".svg")


def save_current_figure(base_path):
    plt.savefig(
        base_path + ".png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.savefig(
        base_path + ".pdf",
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.savefig(
        base_path + ".svg",
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.close()


def save_classwise_comparison(shap_array, X, y_test, output_dir, file_prefix):
    y_test_array = y_test.values

    benign_mask = y_test_array == 0
    ransomware_mask = y_test_array == 1

    benign_shap = shap_array[benign_mask]
    ransomware_shap = shap_array[ransomware_mask]

    classwise_df = pd.DataFrame({
        "feature": X.columns,
        "benign_mean_abs_shap": np.abs(benign_shap).mean(axis=0),
        "ransomware_mean_abs_shap": np.abs(ransomware_shap).mean(axis=0),
        "benign_mean_shap": benign_shap.mean(axis=0),
        "ransomware_mean_shap": ransomware_shap.mean(axis=0),
    })

    classwise_df["abs_difference"] = (
        classwise_df["ransomware_mean_abs_shap"]
        - classwise_df["benign_mean_abs_shap"]
    )

    classwise_df["signed_difference"] = (
        classwise_df["ransomware_mean_shap"]
        - classwise_df["benign_mean_shap"]
    )

    classwise_df = classwise_df.sort_values(
        by="abs_difference",
        ascending=False
    )

    output_path = path.join(
        output_dir,
        "tables",
        f"{file_prefix}_benign_vs_ransomware_shap_comparison.csv"
    )

    classwise_df.to_csv(output_path, index=False)

    print("Saved benign vs ransomware SHAP comparison:")
    print(output_path)

    print("\nTop 20 features contributing more to ransomware:")
    print(classwise_df.head(TOP_FEATURES))


def save_dualuse_fp_analysis(
    shap_array,
    dual_X,
    dual_preds,
    output_dir,
    file_prefix
):
    dual_fp_mask = dual_preds == 1
    dual_tn_mask = dual_preds == 0

    print("Dual-use FP count:", int(dual_fp_mask.sum()))
    print("Dual-use TN count:", int(dual_tn_mask.sum()))

    if dual_fp_mask.sum() == 0:
        print("No dual-use false positives found.")
        return

    dual_fp_shap = shap_array[dual_fp_mask]
    dual_tn_shap = shap_array[dual_tn_mask]

    dual_fp_df = pd.DataFrame({
        "feature": dual_X.columns,
        "dual_fp_mean_abs_shap": np.abs(dual_fp_shap).mean(axis=0),
        "dual_fp_mean_shap": dual_fp_shap.mean(axis=0),
        "dual_tn_mean_abs_shap": (
            np.abs(dual_tn_shap).mean(axis=0)
            if dual_tn_mask.sum() > 0
            else np.nan
        ),
        "dual_tn_mean_shap": (
            dual_tn_shap.mean(axis=0)
            if dual_tn_mask.sum() > 0
            else np.nan
        ),
    })

    dual_fp_df["fp_minus_tn_abs_difference"] = (
        dual_fp_df["dual_fp_mean_abs_shap"]
        - dual_fp_df["dual_tn_mean_abs_shap"]
    )

    dual_fp_df["fp_minus_tn_signed_difference"] = (
        dual_fp_df["dual_fp_mean_shap"]
        - dual_fp_df["dual_tn_mean_shap"]
    )

    dual_fp_df = dual_fp_df.sort_values(
        by="dual_fp_mean_abs_shap",
        ascending=False
    )

    output_path = path.join(
        output_dir,
        "tables",
        f"{file_prefix}_dualuse_false_positive_shap_importance.csv"
    )

    dual_fp_df.to_csv(output_path, index=False)

    print("Saved dual-use FP SHAP table:")
    print(output_path)

    print("\nTop 20 SHAP features for dual-use false positives:")
    print(dual_fp_df.head(TOP_FEATURES))


def save_waterfall(
    explainer,
    X,
    selected_idx,
    metadata,
    output_dir,
    metadata_filename,
    plot_filename
):
    metadata_path = path.join(
        output_dir,
        "tables",
        metadata_filename
    )

    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)

    shap_sample = explainer(X.iloc[[selected_idx]])

    plt.figure(figsize=(IEEE_DOUBLE_COL_WIDTH, 5.5))

    shap.plots.waterfall(
        shap_sample[0],
        show=False,
        max_display=TOP_FEATURES
    )

    ax = plt.gca()
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    base_path = path.join(
        output_dir,
        "plots",
        plot_filename
    )

    save_current_figure(base_path)

    print("Saved waterfall plot:")
    print(base_path + ".png")
    print(base_path + ".pdf")
    print(base_path + ".svg")
    print("Saved selected sample metadata:")
    print(metadata_path)


def save_dualuse_fp_waterfall(
    explainer,
    dual_X,
    dual_preds,
    dual_probs,
    output_dir
):
    fp_indices = np.where(dual_preds == 1)[0]

    if len(fp_indices) == 0:
        print("No dual-use false positive found.")
        return

    selected_pos = int(np.argmax(dual_probs[fp_indices]))
    fp_idx = int(fp_indices[selected_pos])

    print(f"Using dual-use FP index: {fp_idx}")
    print(f"Dual-use ransomware probability: {dual_probs[fp_idx]:.4f}")

    save_waterfall(
        explainer=explainer,
        X=dual_X,
        selected_idx=fp_idx,
        metadata={
            "dualuse_index": fp_idx,
            "ransomware_probability": float(dual_probs[fp_idx]),
            "prediction": int(dual_preds[fp_idx])
        },
        output_dir=output_dir,
        metadata_filename="dualuse_false_positive_waterfall_selected_sample.csv",
        plot_filename="dualuse_false_positive_waterfall_publication"
    )


def save_true_ransomware_waterfall(
    explainer,
    X_test,
    y_test,
    test_preds,
    test_probs,
    output_dir
):
    tp_mask = (
        (test_preds == 1)
        & (y_test.values == 1)
    )

    tp_indices = np.where(tp_mask)[0]

    if len(tp_indices) == 0:
        print("No true positive found.")
        return

    selected_pos = int(np.argmax(test_probs[tp_indices]))
    tp_idx = int(tp_indices[selected_pos])

    print(f"Using true positive index: {tp_idx}")
    print(f"Ransomware probability: {test_probs[tp_idx]:.4f}")

    save_waterfall(
        explainer=explainer,
        X=X_test,
        selected_idx=tp_idx,
        metadata={
            "test_index": tp_idx,
            "true_label": int(y_test.values[tp_idx]),
            "prediction": int(test_preds[tp_idx]),
            "ransomware_probability": float(test_probs[tp_idx])
        },
        output_dir=output_dir,
        metadata_filename="true_ransomware_waterfall_selected_sample.csv",
        plot_filename="true_ransomware_waterfall_publication"
    )


def run_shap_pipeline(
    X,
    y,
    dual_X,
    output_dir,
    file_prefix,
    include_detailed_outputs=False
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    xgb_model = train_xgboost(X_train, y_train)

    test_preds = xgb_model.predict(X_test)
    test_probs = xgb_model.predict_proba(X_test)[:, 1]

    dual_preds = xgb_model.predict(dual_X)
    dual_probs = xgb_model.predict_proba(dual_X)[:, 1]

    explainer = shap.TreeExplainer(xgb_model)

    shap_values_test = explainer(X_test)
    shap_test_array = shap_values_to_array(shap_values_test)

    shap_values_dual = explainer(dual_X)
    shap_dual_array = shap_values_to_array(shap_values_dual)

    global_importance_df = save_global_importance(
        shap_array=shap_test_array,
        X=X_test,
        output_dir=output_dir,
        file_prefix=file_prefix
    )

    save_summary_plot(
        shap_values=shap_values_test,
        X=X_test,
        output_dir=output_dir,
        file_prefix=file_prefix
    )

    save_bar_plot(
        shap_values=shap_values_test,
        X=X_test,
        output_dir=output_dir,
        file_prefix=file_prefix
    )

    if include_detailed_outputs:
        save_classwise_comparison(
            shap_array=shap_test_array,
            X=X_test,
            y_test=y_test,
            output_dir=output_dir,
            file_prefix=file_prefix
        )

        save_dualuse_fp_analysis(
            shap_array=shap_dual_array,
            dual_X=dual_X,
            dual_preds=dual_preds,
            output_dir=output_dir,
            file_prefix=file_prefix
        )

        save_dualuse_fp_waterfall(
            explainer=explainer,
            dual_X=dual_X,
            dual_preds=dual_preds,
            dual_probs=dual_probs,
            output_dir=output_dir
        )

        save_true_ransomware_waterfall(
            explainer=explainer,
            X_test=X_test,
            y_test=y_test,
            test_preds=test_preds,
            test_probs=test_probs,
            output_dir=output_dir
        )

    return global_importance_df


def save_merged_vs_no_entropy_comparison(
    merged_importance_df,
    no_entropy_importance_df
):
    merged_comp = merged_importance_df[[
        "feature",
        "mean_abs_shap"
    ]].rename(
        columns={"mean_abs_shap": "merged_mean_abs_shap"}
    )

    no_entropy_comp = no_entropy_importance_df[[
        "feature",
        "mean_abs_shap"
    ]].rename(
        columns={"mean_abs_shap": "no_entropy_mean_abs_shap"}
    )

    shap_comparison_df = pd.merge(
        merged_comp,
        no_entropy_comp,
        on="feature",
        how="outer"
    ).fillna(0)

    shap_comparison_df["difference_merged_minus_no_entropy"] = (
        shap_comparison_df["merged_mean_abs_shap"]
        - shap_comparison_df["no_entropy_mean_abs_shap"]
    )

    shap_comparison_df = shap_comparison_df.sort_values(
        by="difference_merged_minus_no_entropy",
        ascending=False
    )

    comparison_path = path.join(
        PHASE6_DIR,
        "merged_vs_no_entropy_shap_comparison.csv"
    )

    shap_comparison_df.to_csv(comparison_path, index=False)

    print("Saved merged vs no-entropy SHAP comparison:")
    print(comparison_path)

    print("\nTop 20 features with higher contribution in merged model:")
    print(shap_comparison_df.head(TOP_FEATURES))


def main():
    print("\n")
    print("=" * 80)
    print("PHASE 6 -> SHAP ANALYSIS: MERGED + XGBOOST")
    print("=" * 80)

    ensure_phase_dirs()

    training_datasets, training_merged_df = build_training_features()
    dual_datasets, dual_merged_X = build_dualuse_features()

    X_merged, y_merged = prepare_xy(training_merged_df.copy())
    dual_merged_X = align_dualuse_features(X_merged, dual_merged_X)

    no_entropy_families = [
        "header",
        "dll",
        "function"
    ]

    no_entropy_df = build_ablation_training_df(
        training_datasets=training_datasets,
        training_merged_df=training_merged_df,
        families=no_entropy_families
    )

    X_no_entropy, y_no_entropy = prepare_xy(no_entropy_df)

    dual_no_entropy_X = build_ablation_dualuse_df(
        dual_datasets=dual_datasets,
        families=no_entropy_families
    )

    dual_no_entropy_X = align_dualuse_features(
        X_no_entropy,
        dual_no_entropy_X
    )

    print("\nFeature alignment verified.")
    print("Merged training shape        :", X_merged.shape)
    print("Merged dual-use shape        :", dual_merged_X.shape)
    print("No-entropy training shape    :", X_no_entropy.shape)
    print("No-entropy dual-use shape    :", dual_no_entropy_X.shape)

    print("\n" + "=" * 80)
    print("MERGED XGBOOST SHAP")
    print("=" * 80)

    merged_importance_df = run_shap_pipeline(
        X=X_merged,
        y=y_merged,
        dual_X=dual_merged_X,
        output_dir=PHASE6_MERGED_XGBOOST_DIR,
        file_prefix="merged_xgboost",
        include_detailed_outputs=True
    )

    print("\n" + "=" * 80)
    print("ALL-MINUS-ENTROPY XGBOOST SHAP")
    print("=" * 80)

    no_entropy_importance_df = run_shap_pipeline(
        X=X_no_entropy,
        y=y_no_entropy,
        dual_X=dual_no_entropy_X,
        output_dir=PHASE6_NO_ENTROPY_XGBOOST_DIR,
        file_prefix="all_minus_entropy_xgboost",
        include_detailed_outputs=False
    )

    save_merged_vs_no_entropy_comparison(
        merged_importance_df=merged_importance_df,
        no_entropy_importance_df=no_entropy_importance_df
    )

    print("\n")
    print("=" * 80)
    print("PHASE 6 SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print("All outputs saved under:")
    print(PHASE6_DIR)


if __name__ == "__main__":
    main()
