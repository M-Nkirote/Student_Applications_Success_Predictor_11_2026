import json
import numpy as np
import pandas as pd
from src import config


def add_outcome_and_target(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    # won / lost / lost (stale) / open, and a binary target (NaN for open applications).
    known_stages = config.WON_STAGES + config.LOST_STAGES + config.EARLY_OPEN_STAGES + config.LATE_OPEN_STAGES
    unknown = set(df["Most Advanced Stage"].dropna().unique()) - set(known_stages)
    assert not unknown, f"Stages not mapped to a group: {unknown}"

    df["outcome"] = "open"
    df.loc[df["Most Advanced Stage"].isin(config.WON_STAGES), "outcome"] = "won"
    df.loc[df["Most Advanced Stage"].isin(config.LOST_STAGES), "outcome"] = "lost"

    # Early-stage applications that have been open too long are lost (kind of no indication that the student in invested in the process
    df["age_months_at_snapshot"] = (snapshot_date - df["Created At"]).dt.days / 30.44
    is_stale = df["Most Advanced Stage"].isin(config.EARLY_OPEN_STAGES) & (df["age_months_at_snapshot"] > config.STALE_MONTHS)
    df.loc[is_stale, "outcome"] = "lost (stale)"

    df["target"] = np.where(df["outcome"] == "won", 1, np.where(df["outcome"].str.startswith("lost"), 0, np.nan))
    print("Outcome groups:", df["outcome"].value_counts().to_dict())
    return df


def fix_multiple_enrolments(df: pd.DataFrame) -> pd.DataFrame:
    """A student can enrol only once.
    - exactly one enrolled application has an Enrolled Date, then we keep it and remove the undated enrolled copies
    - none or several dated, we cannot tell which is real so its safer to remove all of that student's applications
    """
    won = df[df["outcome"] == "won"]
    multi_won = won[won["Student ID"].duplicated(keep=False)]

    n_dated = multi_won.groupby("Student ID")["Enrolled Date"].count()
    identifiable_ids = n_dated[n_dated == 1].index
    ambiguous_ids = n_dated[n_dated != 1].index

    undated_copies = multi_won.loc[multi_won["Student ID"].isin(identifiable_ids) & multi_won["Enrolled Date"].isna(), "Opportunity ID"]
    ambiguous_apps = df.loc[df["Student ID"].isin(ambiguous_ids), "Opportunity ID"]
    remove_ids = set(undated_copies) | set(ambiguous_apps)

    df = df[~df["Opportunity ID"].isin(remove_ids)].reset_index(drop=True)
    assert df.loc[df["outcome"] == "won", "Student ID"].duplicated().sum() == 0

    print(f"Multiple enrolments: {len(identifiable_ids)} identifiable, {len(ambiguous_ids)} ambiguous students "
          f"-> removed {len(remove_ids)} applications")
    return df


def add_training_window(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    intake_passed = (df["intake_date"] + pd.DateOffset(months=config.INTAKE_BUFFER_MONTHS)) <= snapshot_date
    old_without_intake = df["intake_date"].isna() & (df["age_months_at_snapshot"] > config.MATURE_MONTHS_NO_INTAKE)
    df["label_mature"] = intake_passed | old_without_intake

    closed = df["target"].notna()
    print(f"Closed applications inside the training window: {int((closed & df['label_mature']).sum()):,} | "
          f"outside (excluded): {int((closed & ~df['label_mature']).sum()):,}")
    return df


def main() -> None:
    df = pd.read_pickle(config.CLEAN_DATA_PATH)
    snapshot_date = pd.Timestamp(json.loads(config.DATA_INFO_PATH.read_text())["snapshot_date"])

    df = add_outcome_and_target(df, snapshot_date)
    df = fix_multiple_enrolments(df)
    df = add_training_window(df, snapshot_date)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(config.LABELLED_DATA_PATH)
    print(f"Saved labelled data ({len(df):,} rows) to {config.LABELLED_DATA_PATH}")


if __name__ == "__main__":
    main()
