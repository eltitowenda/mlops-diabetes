"""Unit tests for data validation and cleaning (no MLflow needed)."""
import numpy as np
import pandas as pd
import pytest

from src.config import ALL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.data.clean import clean_data
from src.data.extract import extract_data
from src.data.validate import DataValidationError, validate_data


@pytest.fixture
def good_df():
    return extract_data()


def test_extract_returns_expected_schema(good_df):
    assert list(good_df.columns) == ALL_COLUMNS
    assert len(good_df) == 442


def test_validation_passes_on_good_data(good_df):
    report = validate_data(good_df)
    assert report["passed"] is True
    assert report["errors"] == []


def test_validation_fails_when_feature_missing(good_df):
    with pytest.raises(DataValidationError, match="bmi"):
        validate_data(good_df.drop(columns=["bmi"]))


def test_validation_fails_when_target_missing(good_df):
    with pytest.raises(DataValidationError, match="Target"):
        validate_data(good_df.drop(columns=[TARGET_COLUMN]))


def test_validation_fails_on_non_numeric_column(good_df):
    bad = good_df.copy()
    bad["age"] = "not-a-number"
    with pytest.raises(DataValidationError, match="age"):
        validate_data(bad)


def test_validation_fails_on_too_many_missing_values(good_df):
    bad = good_df.copy()
    bad.loc[: len(bad) // 2, "bp"] = np.nan  # ~50% missing
    with pytest.raises(DataValidationError, match="bp"):
        validate_data(bad)


def test_validation_fails_on_empty_dataframe(good_df):
    with pytest.raises(DataValidationError, match="empty"):
        validate_data(good_df.iloc[0:0])


def test_cleaning_removes_missing_duplicates_and_invalid_target(good_df):
    dirty = good_df.copy()
    dirty.loc[0, "bmi"] = np.nan                       # missing value
    dirty = pd.concat([dirty, dirty.iloc[[5]]])        # duplicated row
    dirty.loc[10, TARGET_COLUMN] = -5                  # invalid target
    dirty["extra_col"] = 1                             # unexpected column

    cleaned, report = clean_data(dirty)

    assert list(cleaned.columns) == ALL_COLUMNS
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned[TARGET_COLUMN] > 0).all()
    assert report["rows_removed_missing"] == 1
    assert report["rows_removed_duplicates"] == 1
    assert report["rows_removed_invalid_target"] == 1
    assert report["rows_after"] == report["rows_before"] - 3


def test_cleaning_coerces_numeric_strings(good_df):
    df = good_df.copy()
    df["age"] = df["age"].astype(str)
    cleaned, _ = clean_data(df)
    assert pd.api.types.is_numeric_dtype(cleaned["age"])


def test_cleaning_keeps_all_rows_of_clean_data(good_df):
    cleaned, report = clean_data(good_df)
    assert len(cleaned) == len(good_df)
    assert report["rows_after"] == 442
    assert all(c in cleaned.columns for c in FEATURE_COLUMNS)
