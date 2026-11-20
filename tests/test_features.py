"""Feature engineering used by training and the API."""
import numpy as np
import pandas as pd

from src import config, features


def make_row(created, intake):
    return pd.DataFrame({
        "Created At": [pd.Timestamp(created)], "intake_date": [pd.Timestamp(intake) if intake else pd.NaT],
        "Lead Source Main Category": ["Digital"], "Deal Owner": ["OWNER_X"], "Student Country Location": ["Kenya"],
        "level_of_study": ["Undergraduate"], "destination": ["Canada"], "Study Type": ["On-Campus"],
    })


def test_intake_lead_months_and_months():
    df = features.add_features(make_row("2025-03-01", "2025-09-01"))
    assert abs(df["intake_lead_months"].iloc[0] - 6.0) < 0.1
    assert df["intake_month"].iloc[0] == "Sep"
    assert df["created_month"].iloc[0] == "Mar"


def test_undecided_intake():
    df = features.add_features(make_row("2025-03-01", None))
    assert np.isnan(df["intake_lead_months"].iloc[0])
    assert df["intake_month"].iloc[0] == "Undecided"


def test_implausible_lead_time_becomes_missing():
    df = features.add_features(make_row("2025-03-01", "2030-09-01"))
    assert np.isnan(df["intake_lead_months"].iloc[0])


def test_feature_matrix_columns():
    df = features.get_feature_matrix(features.add_features(make_row("2025-03-01", "2025-09-01")))
    assert list(df.columns) == config.FEATURES
