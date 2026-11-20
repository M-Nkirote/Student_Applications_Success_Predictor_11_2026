import numpy as np
import pandas as pd

from src import config


def strip_text(series: pd.Series) -> pd.Series:
    # Remove non-breaking spaces and leading/trailing whitespace.
    return series.str.replace("\xa0", " ", regex=False).str.strip()


def parse_crm_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="ISO8601", errors="coerce", utc=True)
    return parsed.dt.tz_convert("Africa/Nairobi").dt.tz_localize(None)


def parse_intake(series: pd.Series) -> pd.Series:
    # Parse intakes stored like 'Sep-23'. Values like 'Undecided', '0' and blanks become NaT.
    return pd.to_datetime(series, format="%b-%y", errors="coerce")


def clean_country(series: pd.Series) -> pd.Series:
    country = series.astype("string").str.strip().str.title()
    looks_like_junk = country.str.contains(r"@|http|\d", regex=True, na=False)
    country = country.mask(looks_like_junk)
    return country.map(lambda value: config.COUNTRY_MAPPING.get(value, value) if pd.notna(value) else np.nan)


def clean_level_of_study(series: pd.Series) -> pd.Series:
    return series.map(config.LEVEL_MAPPING)


def clean_destination(series: pd.Series) -> pd.Series:
    return series.replace(config.DESTINATION_MAPPING).fillna("Undecided")
