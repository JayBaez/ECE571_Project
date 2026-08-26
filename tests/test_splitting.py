"""
Tests for src/splitting.py, using small synthetic data.
"""

import pandas as pd
import pytest

from src import splitting


def _synthetic_timeseries_df(n_days=10):
    rows = []
    for day in range(1, n_days + 1):
        for hour in [10, 12, 14]:
            rows.append({"Year": 2020, "Month": 1, "Day": day, "Hour": hour, "Minute": 0, "value": day * 10 + hour})
    return pd.DataFrame(rows)


def test_chronological_split_preserves_order():
    df = _synthetic_timeseries_df()
    train_df, test_df, split_info = splitting.chronological_split(df, train_frac=0.8)

    last_train_day = train_df["Day"].max()
    first_test_day = test_df["Day"].min()
    assert last_train_day <= first_test_day
    assert split_info["n_train"] == len(train_df)
    assert split_info["n_test"] == len(test_df)


def test_chronological_split_no_overlap():
    df = _synthetic_timeseries_df()
    train_df, test_df, _ = splitting.chronological_split(df, train_frac=0.7)
    assert splitting.verify_no_overlap(train_df, test_df)


def test_chronological_split_invalid_train_frac_raises():
    df = _synthetic_timeseries_df()
    with pytest.raises(ValueError):
        splitting.chronological_split(df, train_frac=1.5)
    with pytest.raises(ValueError):
        splitting.chronological_split(df, train_frac=0.0)


def test_cross_city_split_zero_overlap_and_correct_labels():
    davis = pd.DataFrame({"Year": [2020], "Month": [1], "Day": [1], "Hour": [10], "Minute": [0]})
    davis.attrs["city"] = "Davis"
    amherst = pd.DataFrame({"Year": [2019], "Month": [6], "Day": [1], "Hour": [10], "Minute": [0]})
    amherst.attrs["city"] = "Amherst"

    result = splitting.cross_city_split(davis, amherst)
    assert result["source_city"] == "Davis"
    assert result["target_city"] == "Amherst"
    assert splitting.verify_no_overlap(result["train"], result["test"])


def test_cross_city_split_same_city_raises_valueerror():
    df = pd.DataFrame({"Year": [2020], "Month": [1], "Day": [1], "Hour": [10], "Minute": [0]})
    with pytest.raises(ValueError):
        splitting.cross_city_split(df, df, source_city="Davis", target_city="Davis")


def test_random_labeled_subset_reproducible_with_same_seed():
    df = _synthetic_timeseries_df()
    labeled_a, unlabeled_a = splitting.random_labeled_subset(df, label_fraction=0.3, seed=42)
    labeled_b, unlabeled_b = splitting.random_labeled_subset(df, label_fraction=0.3, seed=42)

    assert list(labeled_a.index) == list(labeled_b.index)
    assert list(unlabeled_a.index) == list(unlabeled_b.index)
    assert len(labeled_a) + len(unlabeled_a) == len(df)


def test_random_labeled_subset_different_seeds_can_differ():
    df = _synthetic_timeseries_df(n_days=30)
    labeled_a, _ = splitting.random_labeled_subset(df, label_fraction=0.3, seed=1)
    labeled_b, _ = splitting.random_labeled_subset(df, label_fraction=0.3, seed=2)
    assert list(labeled_a.index) != list(labeled_b.index)


def test_few_shot_sample_returns_exact_k_rows_reproducibly():
    df = _synthetic_timeseries_df(n_days=30)
    sample_a = splitting.few_shot_sample(df, k=10, seed=42)
    sample_b = splitting.few_shot_sample(df, k=10, seed=42)
    assert len(sample_a) == 10
    assert list(sample_a.index) == list(sample_b.index)


def test_few_shot_sample_k_too_large_raises_valueerror():
    df = _synthetic_timeseries_df(n_days=1)  # only 3 rows
    with pytest.raises(ValueError):
        splitting.few_shot_sample(df, k=1000, seed=42)
