"""
Tests for src/preprocessing.py, using small synthetic data (not the
real dataset) so these run fast and the expected values are easy to
hand-verify.
"""

import numpy as np
import pandas as pd

from src import preprocessing


def _synthetic_df():
    return pd.DataFrame(
        {
            "Year": [2020] * 6,
            "Month": [1] * 6,
            "Day": [1] * 6,
            "Hour": [10, 11, 12, 13, 14, 15],
            "Minute": [0] * 6,
            "GHI": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "Temperature": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "Cloud Type": [0, 0, 1, 1, 2, 2],
            "Output Power": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        }
    )


def test_get_numeric_columns_excludes_time_and_categorical():
    df = _synthetic_df()
    numeric_cols = preprocessing.get_numeric_columns(df, exclude=["Output Power"])
    assert "GHI" in numeric_cols
    assert "Temperature" in numeric_cols
    assert "Cloud Type" not in numeric_cols
    assert "Year" not in numeric_cols
    assert "Output Power" not in numeric_cols


def test_get_categorical_columns_finds_cloud_type():
    df = _synthetic_df()
    assert preprocessing.get_categorical_columns(df) == ["Cloud Type"]


def test_prepare_xy_excludes_target_and_extra_columns():
    df = _synthetic_df()
    X, y, feature_columns = preprocessing.prepare_xy(
        df, target_column="Output Power", exclude_columns=["GHI"]
    )
    assert "Output Power" not in X.columns
    assert "GHI" not in X.columns
    assert "Output Power" not in feature_columns
    assert list(y) == list(df["Output Power"])


def test_handle_missing_values_drop_removes_rows():
    df = _synthetic_df()
    df.loc[2, "Output Power"] = np.nan
    cleaned = preprocessing.handle_missing_values(df, strategy="drop", columns=["Output Power"])
    assert len(cleaned) == len(df) - 1
    assert cleaned["Output Power"].isna().sum() == 0


def test_handle_missing_values_interpolate_fills_gap():
    df = _synthetic_df()
    df.loc[2, "Output Power"] = np.nan  # true value was 15.0, neighbors are 10.0 and 20.0
    cleaned = preprocessing.handle_missing_values(df, strategy="interpolate", columns=["Output Power"])
    assert cleaned["Output Power"].isna().sum() == 0
    assert cleaned.loc[2, "Output Power"] == 15.0  # midpoint of 10 and 20


def test_scaler_is_fit_on_train_only_not_on_combined_data():
    # Train and "test" have very different distributions on purpose,
    # so if the scaler were accidentally fit on combined data, its
    # mean would land somewhere between the two - this test would then
    # fail, catching the leak.
    train_df = pd.DataFrame({"GHI": [0.0, 1.0, 2.0, 3.0, 4.0]})
    test_df = pd.DataFrame({"GHI": [1000.0, 1001.0, 1002.0]})

    scaler = preprocessing.fit_scaler(train_df, ["GHI"])

    assert np.isclose(scaler.mean_[0], train_df["GHI"].mean())
    assert not np.isclose(scaler.mean_[0], pd.concat([train_df, test_df])["GHI"].mean())

    # Applying the train-fitted scaler to test data should NOT refit it.
    test_scaled = preprocessing.apply_scaler(test_df, scaler, ["GHI"])
    assert np.isclose(scaler.mean_[0], train_df["GHI"].mean())  # unchanged
    assert test_scaled["GHI"].mean() > 0  # test values are far from train's mean, scaled result reflects that


def test_apply_scaler_result_has_roughly_zero_mean_on_train():
    df = _synthetic_df()
    scaler = preprocessing.fit_scaler(df, ["GHI"])
    scaled = preprocessing.apply_scaler(df, scaler, ["GHI"])
    assert abs(scaled["GHI"].mean()) < 1e-9


def test_encoder_is_fit_on_train_only():
    train_df = pd.DataFrame({"Cloud Type": [0, 0, 1, 1]})
    test_df = pd.DataFrame({"Cloud Type": [0, 1, 9]})  # 9 never seen in training

    encoder = preprocessing.fit_encoder(train_df, ["Cloud Type"])
    # handle_unknown="ignore" means the unseen category "9" becomes
    # all-zeros instead of crashing.
    encoded_test = preprocessing.apply_encoder(test_df, encoder, ["Cloud Type"])
    unseen_row = encoded_test.iloc[2]
    assert unseen_row.sum() == 0


def test_target_scaler_round_trip_recovers_original_values():
    train_target = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    scaler = preprocessing.fit_target_scaler(train_target)

    scaled = preprocessing.apply_target_scaler(train_target, scaler)
    recovered = preprocessing.inverse_transform_target(scaled, scaler)

    assert np.allclose(recovered, train_target.values)
