import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from src import config, evaluate, features, model


def load_modelling_data() -> pd.DataFrame:
    df = pd.read_pickle(config.LABELLED_DATA_PATH)
    df = df.sort_values(["Student ID", "Created At"]).reset_index(drop=True)
    df = features.add_features(df)
    model_df = df[df["target"].notna() & df["label_mature"]].copy()
    model_df["target"] = model_df["target"].astype(int)
    print(f"Modelling set: {len(model_df):,} applications, {model_df['Student ID'].nunique():,} students, "
          f"{model_df['target'].mean():.1%} enrolled")
    return model_df


def grouped_split(X, y, groups, cv):
    # Hold out one grouped fold (~20%) as the test set. A student shouldn't appear in both sets.
    train_idx, test_idx = next(cv.split(X, y, groups))
    assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    return train_idx, test_idx


def tune_gradient_boosting(X_train, y_train, groups_train, cv):
    search = RandomizedSearchCV(model.make_gradient_boosting(), config.PARAM_DISTRIBUTIONS,
                                n_iter=config.TUNING_ITERATIONS, scoring="average_precision", cv=cv,
                                n_jobs=-1, random_state=config.RANDOM_STATE, refit=False)
    search.fit(X_train, y_train, groups=groups_train)

    results = pd.DataFrame(search.cv_results_)
    best = results.loc[results["rank_test_score"] == 1].iloc[0]
    within_one_std = results[results["mean_test_score"] >= best["mean_test_score"] - best["std_test_score"]]
    simplest = within_one_std.sort_values(["param_model__max_iter", "param_model__max_leaf_nodes", "mean_test_score"],
                                          ascending=[True, True, False]).iloc[0]
    print(f"Tuning: best CV PR-AUC {best['mean_test_score']:.4f}; chosen (simplest) "
          f"{simplest['mean_test_score']:.4f} ± {simplest['std_test_score']:.4f}")

    chosen_params = simplest["params"]
    tuned = model.make_gradient_boosting().set_params(**chosen_params).fit(X_train, y_train)
    return tuned, chosen_params


def choose_final_model(tuned_gb, X_train, y_train, groups_train, cv):
    # Compare tuned gradient boosting with logistic regression on out-of-fold predictions.
    logistic = model.make_logistic_regression()
    oof_gb = cross_val_predict(tuned_gb, X_train, y_train, groups=groups_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_lr = cross_val_predict(logistic, X_train, y_train, groups=groups_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    comparison = {
        name: {"roc_auc": roc_auc_score(y_train, p), "pr_auc": average_precision_score(y_train, p),
               "brier": brier_score_loss(y_train, p)}
        for name, p in [("gradient_boosting", oof_gb), ("logistic_regression", oof_lr)]
    }
    gb, lr = comparison["gradient_boosting"], comparison["logistic_regression"]
    logistic_wins = lr["pr_auc"] >= gb["pr_auc"] - config.PR_AUC_TOLERANCE and lr["brier"] <= gb["brier"]

    if logistic_wins:
        return "Logistic regression", logistic.fit(X_train, y_train), oof_lr, comparison
    return "Tuned gradient boosting", tuned_gb, oof_gb, comparison


def choose_threshold_and_bands(y_train, oof_proba) -> dict:
    # F1-optimal threshold and priority-band cut-offs, learned from out-of-fold TRAINING predictions.
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
    f1 = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    return {
        "threshold": float(thresholds[np.argmax(f1)]),
        "high_cutoff": float(np.quantile(oof_proba, config.HIGH_BAND_QUANTILE)),
        "medium_cutoff": float(np.quantile(oof_proba, config.MEDIUM_BAND_QUANTILE)),
    }


def form_options(model_df: pd.DataFrame) -> dict:
    # Values the web form offers in its drop-down lists (taken from the training data).
    def sorted_values(column):
        return sorted(model_df[column].dropna().astype(str).unique().tolist())
    return {
        "lead_sources": sorted_values("Lead Source Main Category"),
        "deal_owners": sorted_values("Deal Owner"),
        "countries": sorted_values("Student Country Location"),
        "levels": ["Undergraduate", "Postgraduate", "Diploma/Certificate", "Other/Undecided"],
        "destinations": sorted_values("destination"),
    }


def main() -> None:
    model_df = load_modelling_data()
    X = features.get_feature_matrix(model_df)
    y = model_df["target"]
    groups = model_df["Student ID"]
    cv = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    # Grouped train/test split
    train_idx, test_idx = grouped_split(X, y, groups, cv)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # Tune gradient boosting, and choose the final algorithm
    tuned_gb, chosen_params = tune_gradient_boosting(X_train, y_train, groups_train, cv)
    algorithm, selected_model, oof_proba, comparison = choose_final_model(tuned_gb, X_train, y_train, groups_train, cv)
    print(f"Selected model: {algorithm}")

    # Threshold and priority bands (training data only)
    cutoffs = choose_threshold_and_bands(y_train, oof_proba)

    # Evaluate once on the untouched test set
    test_proba = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate.compute_metrics(y_test, test_proba, cutoffs["threshold"])
    lift = evaluate.lift_table(y_test, test_proba)
    bands = evaluate.band_table(y_test, evaluate.assign_band(test_proba, cutoffs["high_cutoff"], cutoffs["medium_cutoff"]))
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    # Refit on all labelled data for deployment
    final_model = clone(selected_model).fit(X, y)

    # Save the artifacts
    for folder in [config.MODELS_DIR, config.REPORTS_DIR, config.OUTPUTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, config.MODEL_PATH)
    X_train.sample(min(config.SHAP_BACKGROUND_SIZE, len(X_train)), random_state=config.RANDOM_STATE) \
        .to_pickle(config.SHAP_BACKGROUND_PATH)

    snapshot_date = json.loads(config.DATA_INFO_PATH.read_text())["snapshot_date"]
    metadata = {
        "algorithm": algorithm,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "chosen_params": {k: v if v is None or isinstance(v, (int, float, str)) else str(v) for k, v in chosen_params.items()},
        **cutoffs,
        "training_base_rate": float(y.mean()),
        "training_rows": int(len(model_df)),
        "data_snapshot_date": snapshot_date,
        "test_metrics": test_metrics,
        "form_options": form_options(model_df),
        "library_versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "scikit-learn": sklearn.__version__, "numpy": np.__version__},
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    reports = {"model_comparison_oof": comparison, "test_metrics": test_metrics,
               "priority_bands_test": bands.reset_index().to_dict(orient="records")}
    config.METRICS_PATH.write_text(json.dumps(reports, indent=2, default=float))
    lift.to_csv(config.LIFT_TABLE_PATH)

    split = pd.DataFrame({"Opportunity ID": model_df["Opportunity ID"],
                          "split": np.where(np.isin(np.arange(len(model_df)), test_idx), "test", "train")})
    split.to_csv(config.SPLIT_PATH, index=False)

    print(f"Saved model to {config.MODEL_PATH} and reports to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
