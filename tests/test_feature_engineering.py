"""
Tests for src/feature_engineering.py.
"""

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering


def _synthetic_df():
    return pd.DataFrame(
        {
            "Year": [2020, 2020, 2020],
            "Month": [6, 6, 12],
            "Day": [15, 15, 31],
            "Hour": [12, 0, 12],
            "Minute": [0, 0, 0],
            "GHI": [400.0, 100.0, 0.0],
            "Clearsky GHI": [500.0, 0.0, 200.0],
            "Output Power": [50.0, 5.0, 0.0],
        }
    )


def test_clear_sky_index_computed_correctly():
    df = _synthetic_df()
    result = feature_engineering.add_clear_sky_index(df)
    assert np.isclose(result.loc[0, "Clear_Sky_Index"], 400.0 / 500.0)


def test_clear_sky_index_handles_zero_clearsky_ghi_safely():
    df = _synthetic_df()
    result = feature_engineering.add_clear_sky_index(df)
    # Row 1 has Clearsky GHI == 0, should be NaN, not a division error.
    assert pd.isna(result.loc[1, "Clear_Sky_Index"])


def test_time_cyclical_features_are_bounded_and_present():
    df = _synthetic_df()
    result = feature_engineering.add_time_cyclical_features(df)
    for col in ["Hour_sin", "Hour_cos", "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"]:
        assert col in result.columns
        assert result[col].between(-1.0, 1.0).all()


def test_time_cyclical_hour_0_and_hour_24_equivalent_style_check():
    # Hour 0 and Hour 12 should NOT look identical, but Month=12 (Dec)
    # and Month=1 (Jan, next cycle) should be close on the circle -
    # check via a small wrap-around case instead of exact 24 (which
    # doesn't exist in Hour, 0-23 only).
    df = pd.DataFrame(
        {"Year": [2020, 2020], "Month": [12, 1], "Day": [31, 1], "Hour": [12, 12], "Minute": [0, 0]}
    )
    result = feature_engineering.add_time_cyclical_features(df)
    # December and January are only ~1-2 days apart on the yearly cycle,
    # so their day-of-year sin/cos should be close to each other, NOT
    # close to opposite (which a naive numeric day-of-year would wrongly
    # suggest: 365 vs 1).
    dist = np.hypot(
        result.loc[0, "DayOfYear_sin"] - result.loc[1, "DayOfYear_sin"],
        result.loc[0, "DayOfYear_cos"] - result.loc[1, "DayOfYear_cos"],
    )
    assert dist < 0.1


def test_add_lag_features_shifts_correctly():
    df = _synthetic_df()
    result = feature_engineering.add_lag_features(df, "Output Power", [1])
    assert pd.isna(result.loc[0, "Output Power_lag1"])
    assert result.loc[1, "Output Power_lag1"] == 50.0
    assert result.loc[2, "Output Power_lag1"] == 5.0


def test_add_feature_groups_applies_requested_groups_only():
    df = _synthetic_df()
    result = feature_engineering.add_feature_groups(df, ["clear_sky_index"])
    assert "Clear_Sky_Index" in result.columns
    assert "Hour_sin" not in result.columns  # time_cyclical wasn't requested


def test_add_feature_groups_unknown_group_raises():
    df = _synthetic_df()
    with pytest.raises(ValueError):
        feature_engineering.add_feature_groups(df, ["not_a_real_group"])
