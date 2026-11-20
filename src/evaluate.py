# Evaluation helpers: metrics, decile lift table and priority bands.
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    base_rate = float(np.mean(y_true))
    brier = brier_score_loss(y_true, y_proba)
    base_brier = brier_score_loss(y_true, np.full(len(y_true), base_rate))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "base_rate": base_rate,
        "brier": float(brier),
        "brier_base_rate": float(base_brier),
        "brier_skill": float(1 - brier / base_brier),
        "log_loss": float(log_loss(y_true, y_proba)),
        "f1_at_threshold": float(f1_score(y_true, (np.asarray(y_proba) >= threshold).astype(int))),
    }


def lift_table(y_true, y_proba) -> pd.DataFrame:
    # Sort by score, cut into 10 equal groups, and count enrolments per group.
    table = pd.DataFrame({"score": y_proba, "enrolled": np.asarray(y_true)})
    table["decile"] = pd.qcut(table["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    lift = table.groupby("decile", observed=True).agg(applications=("enrolled", "size"), enrolments=("enrolled", "sum"), mean_score=("score", "mean"))
    lift["enrolment_rate_%"] = (lift["enrolments"] / lift["applications"] * 100).round(1)
    lift["lift_vs_average"] = (lift["enrolment_rate_%"] / (table["enrolled"].mean() * 100)).round(2)
    lift["cumulative_%_of_enrolments"] = (lift["enrolments"].cumsum() / lift["enrolments"].sum() * 100).round(1)
    return lift


def assign_band(probabilities, high_cutoff: float, medium_cutoff: float) -> np.ndarray:
    # High / Medium / Low priority based on the cut-offs learned on training data.
    probabilities = np.asarray(probabilities)
    return np.where(probabilities >= high_cutoff, "High", np.where(probabilities >= medium_cutoff, "Medium", "Low"))


def band_table(y_true, bands) -> pd.DataFrame:
    table = pd.DataFrame({"band": bands, "enrolled": np.asarray(y_true)})
    return table.groupby("band")["enrolled"].agg(applications="size", enrolment_rate="mean") .reindex(["High", "Medium", "Low"]).round(4)
