"""Feature engineering.

Used by training (src/train.py), batch scoring (src/predict.py) and the API (app/model_service.py), so that every application goes through exactly the same steps.

Input columns needed: 'Created At', 'intake_date', 'Lead Source Main Category', 'Deal Owner', 'Student Country Location', 'level_of_study', 'destination'.
"""
import numpy as np
import pandas as pd

from src import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Months between creating the application and the target intake
    df["intake_lead_months"] = (df["intake_date"] - df["Created At"]).dt.days / 30.44
    out_of_range = (df["intake_lead_months"] < -12) | (df["intake_lead_months"] > 36)
    df.loc[out_of_range, "intake_lead_months"] = np.nan

    # Months as text categories
    df["intake_month"] = df["intake_date"].dt.strftime("%b").fillna("Undecided")
    df["created_month"] = df["Created At"].dt.strftime("%b")
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Only the columns the model uses, in a fixed order.
    return df[config.FEATURES]
