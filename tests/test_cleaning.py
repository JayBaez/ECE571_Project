"""
Tests for src/cleaning.py.

The Amherst-specific test uses the REAL dataset on purpose: Phase 0
(course_context/DATASET_PROFILE.md) found exactly 4 missing
"Output Power" rows in Amherst, and this test checks the cleaning
report reproduces that exact real number - this is as much a
regression test on the Phase 0 finding as it is a test of the code.
"""

import numpy as np
import pandas as pd

from src import cleaning, data_loader


def _synthetic_df():
    return pd.DataFrame(
        {
            "GHI": [100.0, -5.0, 300.0, 1000.0],
            "Clearsky GHI": [200.0, 200.0, 300.0, 300.0],  # row 3: GHI way exceeds clearsky
            "DHI": [10.0, 10.0, 10.0, 10.0],
            "DNI": [50.0, 50.0, 50.0, 50.0],
        }
    )


def test_detect_invalid_numeric_values_finds_negative_and_exceeding_ghi():
    df = _synthetic_df()
    report = cleaning.detect_invalid_numeric_values(df)
    assert report["negative_irradiance"]["GHI"] == 1
    assert report["ghi_exceeds_clearsky"] == 1


def test_detect_invalid_numeric_values_clean_data_reports_zero():
    df = pd.DataFrame({"GHI": [100.0, 200.0], "Clearsky GHI": [150.0, 250.0]})
    report = cleaning.detect_invalid_numeric_values(df)
    assert report["negative_irradiance"]["GHI"] == 0
    assert report["ghi_exceeds_clearsky"] == 0


def test_detect_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    assert cleaning.detect_duplicate_rows(df) == 1


def test_clean_sheet_interpolates_missing_target_and_reports_counts():
    df = pd.DataFrame(
        {
            "Year": [2020] * 5,
            "Month": [1] * 5,
            "Day": [1, 2, 3, 4, 5],
            "Hour": [10] * 5,
            "Minute": [0] * 5,
            "Output Power": [10.0, 20.0, np.nan, 40.0, 50.0],
        }
    )
    cleaned_df, report = cleaning.clean_sheet(df, missing_strategy="interpolate", verbose=False)

    assert report["rows_before"] == 5
    assert report["missing_target"] == 1
    assert report["rows_after"] == 5  # interpolate doesn't remove rows
    assert cleaned_df["Output Power"].isna().sum() == 0
    assert cleaned_df.loc[2, "Output Power"] == 30.0  # midpoint of 20 and 40


def test_clean_sheet_drop_strategy_removes_rows():
    df = pd.DataFrame(
        {
            "Year": [2020] * 3,
            "Month": [1] * 3,
            "Day": [1, 2, 3],
            "Hour": [10] * 3,
            "Minute": [0] * 3,
            "Output Power": [10.0, np.nan, 30.0],
        }
    )
    cleaned_df, report = cleaning.clean_sheet(df, missing_strategy="drop", verbose=False)
    assert report["rows_after"] == 2
    assert len(cleaned_df) == 2


def test_clean_sheet_reproduces_the_real_amherst_missing_count():
    # Regression test tying back to the real Phase 0 finding
    # (course_context/DATASET_PROFILE.md): exactly 4 missing
    # Output Power rows in the real Amherst sheet.
    amherst_df = data_loader.load_city("Amherst")
    _, report = cleaning.clean_sheet(amherst_df, missing_strategy="interpolate", verbose=False)
    assert report["missing_target"] == 4
    assert report["rows_after"] == report["rows_before"]
