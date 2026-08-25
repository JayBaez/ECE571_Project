"""
evaluation.py

Reusable metric functions for classification (Problem 1, and part of
Problem 3/4) and regression (Problem 2, and part of Problem 3/4).

Each function returns a plain dict of {metric_name: value}, so results
are easy to print, log to a CSV (see results/), or compare across
experiments. No metric here is fabricated or estimated - these
functions only ever compute a number from real y_true/y_pred arrays
that a future phase will provide from an actual trained model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

# Documents whether a HIGHER or LOWER value is better for each metric.
# Useful when building a leaderboard: you can't just sort every metric
# the same way (a leaderboard sorted "descending" would rank the WORST
# regression model first if you're not careful).
CLASSIFICATION_METRIC_DIRECTION = {
    "accuracy": "higher_better",
    "balanced_accuracy": "higher_better",
    "precision_macro": "higher_better",
    "recall_macro": "higher_better",
    "f1_macro": "higher_better",
}

REGRESSION_METRIC_DIRECTION = {
    "rmse": "lower_better",
    "mae": "lower_better",
    "nrmse": "lower_better",
}


def classification_metrics(y_true, y_pred) -> dict:
    """
    Compute standard classification metrics.

    ML concept: because the sky-condition and generation-regime
    classes are imbalanced (clear-sky dominates - see
    course_context/DATASET_PROFILE.md), the project spec specifically
    asks for BALANCED accuracy (mean of per-class recall) as the
    headline metric, not plain accuracy, which can look artificially
    high if a model just always predicts the majority class (see
    course_context/TEACHER_EXPECTATIONS.md, Problem 1).

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    dict
        {"accuracy", "balanced_accuracy", "precision_macro",
         "recall_macro", "f1_macro"} - all as floats between 0 and 1,
        higher is better for every one of these.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_confusion_matrix(y_true, y_pred, labels: list = None) -> np.ndarray:
    """
    Compute a confusion matrix, ready to hand to
    visualization.plot_confusion_matrix().

    Parameters
    ----------
    y_true, y_pred : array-like
    labels : list, optional
        Explicit class ordering (e.g. ["Clear", "Partly cloudy",
        "Overcast"]). If not given, sklearn infers the order from the
        data.

    Returns
    -------
    numpy.ndarray
        Rows = true class, columns = predicted class.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def regression_metrics(y_true, y_pred, normalization: str = "range") -> dict:
    """
    Compute standard regression metrics, including nRMSE.

    ML concept: raw RMSE is in the same units as the target (kW for
    Output Power), which makes it hard to compare across cities with
    very different scales - Davis's plant is roughly 3-3.5x the
    capacity of Huron/Santa Barbara/La Jolla (see
    course_context/DATASET_PROFILE.md). nRMSE divides RMSE by a
    normalizer so cross-city numbers become comparable, which the
    project spec requires (course_context/TEACHER_EXPECTATIONS.md,
    Problem 2).

    Parameters
    ----------
    y_true : array-like
        Ground-truth continuous values.
    y_pred : array-like
        Predicted continuous values.
    normalization : str
        How to normalize RMSE into nRMSE:
        - "range" (default): divide by (max(y_true) - min(y_true)).
        - "mean": divide by mean(y_true).

    Returns
    -------
    dict
        {"rmse", "mae", "nrmse"} - lower is better for all three.

    Raises
    ------
    ValueError
        If `normalization` is not "range" or "mean".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))

    if normalization == "range":
        denominator = y_true.max() - y_true.min()
    elif normalization == "mean":
        denominator = y_true.mean()
    else:
        raise ValueError(f"Unknown normalization '{normalization}'. Use 'range' or 'mean'.")

    nrmse = float(rmse / denominator) if denominator != 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "nrmse": nrmse}
