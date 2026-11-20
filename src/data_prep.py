import json
import numpy as np
import pandas as pd

from src import cleaning, config


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATA_PATH, dtype={"Student ID": str, "Opportunity ID": str}, low_memory=False)
    assert df["Opportunity ID"].is_unique, "Duplicate Opportunity IDs in the raw data"
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {config.RAW_DATA_PATH.name}")
    return df


def get_snapshot_date(raw_df: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(raw_df["Inserted At"], format="%B %d, %Y, %H:%M", errors="coerce").max()


def clean_data(raw_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    df = raw_df[config.KEEP_COLUMNS].copy()

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = cleaning.strip_text(df[column])
    df["Most Advanced Stage"] = df["Most Advanced Stage"].replace(config.STAGE_FIXES)

    # Dates
    df["Created At"] = cleaning.parse_crm_datetime(df["Created At"])
    df["Enrolled Date"] = cleaning.parse_crm_datetime(df["Enrolled Date"])
    df["intake_date"] = cleaning.parse_intake(df["Correct Intake"])

    # Junk event dates are set as missing
    date_min = pd.Timestamp(config.DATE_MIN)
    date_max = snapshot_date + pd.DateOffset(years=config.DATE_MAX_YEARS_AFTER_SNAPSHOT)
    for column in ["Created At", "Enrolled Date"]:
        is_junk = (df[column] < date_min) | (df[column] > date_max)
        df.loc[is_junk, column] = pd.NaT
        print(f"Junk dates set to missing in '{column}': {int(is_junk.sum())}")

    # On the student country column, there should be one country name per student. The most frequently occurring one is selected
    df["Student Country Location"] = cleaning.clean_country(df["Student Country Location"])
    df["Student Country Location"] = df.groupby("Student ID")["Student Country Location"].transform(
        lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)

    # Level of study and destination
    df["level_of_study"] = cleaning.clean_level_of_study(df["What Level Of Study Are You Applying For"])
    df["destination"] = cleaning.clean_destination(df["Study Destination"])

    return df


def main() -> None:
    raw_df = load_raw_data()
    snapshot_date = get_snapshot_date(raw_df)
    print(f"Snapshot date: {snapshot_date}")

    clean_df = clean_data(raw_df, snapshot_date)

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_pickle(config.CLEAN_DATA_PATH)
    config.DATA_INFO_PATH.write_text(json.dumps({"snapshot_date": str(snapshot_date)}, indent=2))
    print(f"Saved cleaned data ({clean_df.shape[0]:,} rows) to {config.CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
