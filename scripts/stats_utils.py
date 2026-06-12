# =========================================================
# LOAD SAVED PREDICTION FILES
# Used for Wilson CI and McNemar tests
# =========================================================

def load_prediction_file(
    dataset_type,
    model_token,
    phase2_pred_dir,
    phase4_pred_dir
):
    """
    dataset_type:
        - internal
        - dualuse
        - vera
        - dike
    """

    if dataset_type == "internal":
        file_path = path.join(
            phase2_pred_dir,
            f"merged_features__{model_token}__test_predictions.csv"
        )

    elif dataset_type == "dualuse":
        file_path = path.join(
            phase2_pred_dir,
            f"merged_features__{model_token}__dualuse_predictions.csv"
        )

    elif dataset_type == "vera":
        file_path = path.join(
            phase4_pred_dir,
            f"merged_features_cross_dataset__{model_token}__vera_predictions.csv"
        )

    elif dataset_type == "dike":
        file_path = path.join(
            phase4_pred_dir,
            f"merged_features_cross_dataset__{model_token}__dike_predictions.csv"
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if not path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.duplicated()]

    required_cols = {"y_true", "y_pred"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")

    return df, file_path


# =========================================================
# PART 1 → WILSON 95% CONFIDENCE INTERVALS
# =========================================================


def wilson_ci(successes, total, alpha=0.05):
    low, high = proportion_confint(
        count=successes,
        nobs=total,
        alpha=alpha,
        method="wilson"
    )
    return low, high


def build_wilson_row(dataset, model, metric, successes, total, file_path):
    value = successes / total if total > 0 else np.nan
    ci_low, ci_high = wilson_ci(successes, total)

    return {
        "Dataset": dataset,
        "Model": model,
        "Metric": metric,
        "Successes": int(successes),
        "Total": int(total),
        "Value": round(value, 4),
        "CI Lower": round(ci_low, 4),
        "CI Upper": round(ci_high, 4),
        "Value (%)": round(value * 100, 2),
        "CI Lower (%)": round(ci_low * 100, 2),
        "CI Upper (%)": round(ci_high * 100, 2),
        "Source File": file_path
    }



# =========================================================
# PART 2 → MCNEMAR TEST
# =========================================================

def run_mcnemar_test(df_a, df_b, model_a, model_b, dataset_name):
    """
    McNemar's test for paired predictions.
    """

    assert len(df_a) == len(df_b), "Prediction files have different lengths"

    y_true_a = df_a["y_true"].values
    y_true_b = df_b["y_true"].values

    assert np.array_equal(y_true_a, y_true_b), "y_true mismatch between files"

    correct_a = df_a["y_pred"].values == y_true_a
    correct_b = df_b["y_pred"].values == y_true_a

    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))

    table = [[n00, n01], [n10, n11]]
    discordant = n01 + n10

    if discordant < 25:
        result = mcnemar(table, exact=True)
        test_type = "Exact McNemar"
    else:
        result = mcnemar(table, exact=False, correction=True)
        test_type = "Chi-square McNemar with continuity correction"

    return {
        "Dataset": dataset_name,
        "Model A": model_a,
        "Model B": model_b,
        "Both Correct": n00,
        "A Correct B Wrong": n01,
        "A Wrong B Correct": n10,
        "Both Wrong": n11,
        "Discordant Pairs": discordant,
        "Test Type": test_type,
        "Statistic": result.statistic,
        "p-value": result.pvalue,
        "Significant at 0.05": result.pvalue < 0.05
    }



# =========================================================
# PART 3 → STRATIFIED K-FOLD CROSS-VALIDATION
# =========================================================


def run_kfold_cv(X, y, n_splits=5, random_state=42):
    """
    Stratified k-fold CV on merged feature set.
    Scaling is applied inside each fold only for LR and SVM.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    cv_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{n_splits}")
        print("=" * 80)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = get_models(random_state=random_state)

        for model_name, model in models.items():

            print(f"Training {model_name}...")

            use_scaled = model_name in ["Logistic Regression", "SVM"]

            if use_scaled:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)

            cv_rows.append({
                "Fold": fold_idx,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc
            })

            print(
                f"{model_name} | "
                f"Acc: {acc:.4f} | "
                f"Recall: {rec:.4f} | "
                f"F1: {f1:.4f} | "
                f"ROC-AUC: {roc_auc:.4f}"
            )

    return pd.DataFrame(cv_rows)



# =========================================================
# PART 5 → PAIRED T-TEST ON K-FOLD SCORES
# =========================================================

def paired_ttest_cv(cv_df, model_a, model_b, metric):
    """
    Paired t-test on fold-wise metric values.
    """

    a = (
        cv_df[cv_df["Model"] == model_a]
        .sort_values("Fold")[metric]
        .values
    )

    b = (
        cv_df[cv_df["Model"] == model_b]
        .sort_values("Fold")[metric]
        .values
    )

    assert len(a) == len(b), "Fold count mismatch"

    t_stat, p_value = ttest_rel(a, b)

    return {
        "Model A": model_a,
        "Model B": model_b,
        "Metric": metric,
        "Model A Mean": round(float(np.mean(a)), 4),
        "Model B Mean": round(float(np.mean(b)), 4),
        "Mean Difference A-B": round(float(np.mean(a - b)), 4),
        "t-statistic": round(float(t_stat), 6),
        "p-value": float(p_value),
        "Significant at 0.05": p_value < 0.05
    }

