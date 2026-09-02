"""
Tests for src/evaluation.py, using tiny arrays with hand-computed
expected values.
"""

import numpy as np
import pytest

from src import evaluation


def test_regression_metrics_rmse_and_mae_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 30.0, 44.0])
    # errors: 2, -2, 0, 4 -> squared: 4, 4, 0, 16 -> mean 6 -> rmse sqrt(6)
    # abs errors: 2, 2, 0, 4 -> mean 2 -> mae 2.0
    metrics = evaluation.regression_metrics(y_true, y_pred)
    assert np.isclose(metrics["rmse"], np.sqrt(6))
    assert np.isclose(metrics["mae"], 2.0)


def test_regression_metrics_nrmse_range_normalization():
    y_true = np.array([0.0, 10.0])  # range = 10
    y_pred = np.array([0.0, 8.0])  # squared errors: 0, 4 -> mean 2 -> rmse sqrt(2)
    metrics = evaluation.regression_metrics(y_true, y_pred, normalization="range")
    assert np.isclose(metrics["nrmse"], np.sqrt(2) / 10.0)


def test_regression_metrics_nrmse_mean_normalization():
    y_true = np.array([10.0, 10.0, 10.0])  # mean = 10, perfect predictions -> rmse 0
    y_pred = np.array([10.0, 10.0, 10.0])
    metrics = evaluation.regression_metrics(y_true, y_pred, normalization="mean")
    assert metrics["nrmse"] == 0.0


def test_regression_metrics_r2_perfect_and_imperfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    perfect_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert evaluation.regression_metrics(y_true, perfect_pred)["r2"] == 1.0

    # Predicting the mean every time should give r2 == 0.0
    mean_pred = np.full_like(y_true, y_true.mean())
    assert np.isclose(evaluation.regression_metrics(y_true, mean_pred)["r2"], 0.0)


def test_regression_metrics_unknown_normalization_raises():
    with pytest.raises(ValueError):
        evaluation.regression_metrics([1.0], [1.0], normalization="bogus")


def test_classification_metrics_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    metrics = evaluation.classification_metrics(y_true, y_pred)
    for key in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        assert metrics[key] == 1.0


def test_classification_metrics_known_imperfect_case():
    # 4 total, 3 correct -> accuracy = 0.75
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = evaluation.classification_metrics(y_true, y_pred)
    assert np.isclose(metrics["accuracy"], 0.75)


def test_get_confusion_matrix_shape_and_diagonal():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    cm = evaluation.get_confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[0, 1] == 0 and cm[1, 0] == 0


def test_aggregate_across_seeds_computes_mean_and_std():
    metric_dicts = [{"rmse": 10.0}, {"rmse": 20.0}, {"rmse": 30.0}]
    result = evaluation.aggregate_across_seeds(metric_dicts)
    assert result["rmse"]["mean"] == 20.0
    assert np.isclose(result["rmse"]["std"], np.std([10.0, 20.0, 30.0]))
    assert result["rmse"]["values"] == [10.0, 20.0, 30.0]


def test_aggregate_across_seeds_mismatched_keys_raises():
    with pytest.raises(ValueError):
        evaluation.aggregate_across_seeds([{"rmse": 1.0}, {"mae": 2.0}])


def test_aggregate_across_seeds_empty_list_raises():
    with pytest.raises(ValueError):
        evaluation.aggregate_across_seeds([])


def test_metric_direction_dicts_cover_expected_metrics():
    assert evaluation.REGRESSION_METRIC_DIRECTION["rmse"] == "lower_better"
    assert evaluation.CLASSIFICATION_METRIC_DIRECTION["balanced_accuracy"] == "higher_better"
