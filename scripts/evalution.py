def evaluate_models_modular(
    X,
    y,
    dataset_name,
    phase_dir,
    dualuse_X=None,
    extra_metadata=None,
    random_state=42
):
    """
    Modular evaluator.

    Saves:
    - metrics per dataset
    - test predictions per model
    - dual-use predictions per model
    - ROC curve data per model
    - PR curve data per model
    """

    print("\n")
    print("=" * 80)
    print(f"DATASET → {dataset_name}")
    print("=" * 80)

    dataset_safe = safe_name(dataset_name)

    metrics_dir = make_dir(path.join(phase_dir, "metrics"))
    predictions_dir = make_dir(path.join(phase_dir, "predictions"))
    roc_dir = make_dir(path.join(phase_dir, "curves", "roc"))
    pr_dir = make_dir(path.join(phase_dir, "curves", "pr"))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if dualuse_X is not None:
        dualuse_X = dualuse_X[X.columns]
        dualuse_scaled = scaler.transform(dualuse_X)
        y_dual = np.zeros(len(dualuse_X), dtype=int)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            random_state=random_state,
            verbose=-1
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=random_state
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state
        )
    }

    results = []

    for model_name, model in models.items():

        print("\n------------------------------")
        print(f"Model → {model_name}")
        print("------------------------------")

        model_safe = safe_name(model_name)

        use_scaled = model_name in ["Logistic Regression", "SVM"]

        if use_scaled:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        baseline_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        baseline_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"FPR       : {baseline_fpr:.4f}")
        print(f"FNR       : {baseline_fnr:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")

        # -------------------------------------------------
        # Save baseline test predictions
        # -------------------------------------------------

        test_pred_df = pd.DataFrame({
            "Dataset": dataset_name,
            "Model": model_name,
            "y_true": y_test.values,
            "y_pred": preds,
            "ransomware_probability": probs
        })

        test_pred_path = path.join(
            predictions_dir,
            f"{dataset_safe}__{model_safe}__test_predictions.csv"
        )

        test_pred_df.to_csv(test_pred_path, index=False)

        # -------------------------------------------------
        # Save ROC curve data
        # -------------------------------------------------

        fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_test, probs)

        roc_df = pd.DataFrame({
            "Dataset": dataset_name,
            "Model": model_name,
            "fpr": fpr_arr,
            "tpr": tpr_arr,
            "threshold": roc_thresholds
        })

        roc_path = path.join(
            roc_dir,
            f"{dataset_safe}__{model_safe}__roc_curve.csv"
        )

        roc_df.to_csv(roc_path, index=False)

        # -------------------------------------------------
        # Save PR curve data
        # -------------------------------------------------

        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
            y_test,
            probs
        )

        # precision_recall_curve returns thresholds with length n-1
        pr_df = pd.DataFrame({
            "Dataset": dataset_name,
            "Model": model_name,
            "precision": precision_arr,
            "recall": recall_arr,
            "threshold": list(pr_thresholds) + [np.nan]
        })

        pr_path = path.join(
            pr_dir,
            f"{dataset_safe}__{model_safe}__pr_curve.csv"
        )

        pr_df.to_csv(pr_path, index=False)

        # -------------------------------------------------
        # Dual-use benign encryption test
        # -------------------------------------------------

        dual_fp_count = None
        dual_tn_count = None
        dual_fp_rate = None
        dual_accuracy = None
        dual_pred_path = None

        if dualuse_X is not None:

            if use_scaled:
                dual_preds = model.predict(dualuse_scaled)
                dual_probs = model.predict_proba(dualuse_scaled)[:, 1]
            else:
                dual_preds = model.predict(dualuse_X)
                dual_probs = model.predict_proba(dualuse_X)[:, 1]

            dual_fp_count = int(np.sum(dual_preds == 1))
            dual_tn_count = int(np.sum(dual_preds == 0))

            dual_fp_rate = dual_fp_count / len(dual_preds)
            dual_accuracy = dual_tn_count / len(dual_preds)

            print("\nDual-use Encryption Test:")
            print(f"Dual-use TN      : {dual_tn_count}")
            print(f"Dual-use FP      : {dual_fp_count}")
            print(f"Dual-use FP Rate : {dual_fp_rate:.4f}")
            print(f"Dual-use Accuracy: {dual_accuracy:.4f}")

            dual_pred_df = pd.DataFrame({
                "Dataset": dataset_name,
                "Model": model_name,
                "y_true": y_dual,
                "y_pred": dual_preds,
                "ransomware_probability": dual_probs
            })

            dual_pred_path = path.join(
                predictions_dir,
                f"{dataset_safe}__{model_safe}__dualuse_predictions.csv"
            )

            dual_pred_df.to_csv(dual_pred_path, index=False)

        row = {
            "Dataset": dataset_name,
            "Model": model_name,

            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),

            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),

            "Baseline FPR": round(baseline_fpr, 4),
            "Baseline FNR": round(baseline_fnr, 4),

            "Dual-use TN": dual_tn_count,
            "Dual-use FP": dual_fp_count,
            "Dual-use FP Rate": None if dual_fp_rate is None else round(dual_fp_rate, 4),
            "Dual-use Accuracy": None if dual_accuracy is None else round(dual_accuracy, 4),

            "Test Prediction File": test_pred_path,
            "Dual-use Prediction File": dual_pred_path,
            "ROC Curve File": roc_path,
            "PR Curve File": pr_path
        }

        if extra_metadata is not None:
            row.update(extra_metadata)

        results.append(row)

    result_df = pd.DataFrame(results)

    metrics_path = path.join(
        metrics_dir,
        f"{dataset_safe}__metrics.csv"
    )

    result_df.to_csv(metrics_path, index=False)

    print(f"\nSaved metrics → {metrics_path}")

    return result_df