import pandas as pd
from src import cleaning


def test_country_cities_and_typos_are_mapped():
    result = cleaning.clean_country(pd.Series(["nairobi", "UK", "Zimbambwe", "Kenya"]))
    assert result.tolist() == ["Kenya", "United Kingdom", "Zimbabwe", "Kenya"]


def test_country_contact_details_are_removed():
    result = cleaning.clean_country(pd.Series(["someone@example.com", "+254 700 000000"]))
    assert result.isna().all()


def test_level_of_study_groups():
    result = cleaning.clean_level_of_study(pd.Series(["Masters", "PHD", "Degreee", "High School", "Undecided"]))
    assert result.tolist() == ["Postgraduate", "Postgraduate", "Undergraduate", "Diploma/Certificate", "Other/Undecided"]


def test_destination_mapping_and_missing():
    result = cleaning.clean_destination(pd.Series(["England", None, "Canada"]))
    assert result.tolist() == ["United Kingdom", "Undecided", "Canada"]


def test_intake_parsing():
    result = cleaning.parse_intake(pd.Series(["Sep-23", "Undecided", "0"]))
    assert result.iloc[0] == pd.Timestamp("2023-09-01")
    assert result.iloc[1:].isna().all()
