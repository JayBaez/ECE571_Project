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
    r2_score,
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
    "r2": "higher_better",
}


def aggregate_across_seeds(metric_dicts: list) -> dict:
    """
    Combine metric results from several random seeds into mean/std
    summaries, while keeping every individual seed's numbers too.

    ML concept / why this matters: a single run's score can look good
    or bad partly by chance (e.g. a lucky train/test split of
    mini-batches, or a lucky weight initialization). Running the same
    experiment with several different seeds and reporting mean ± std
    shows how much the result actually varies - the project spec
    requires this for regression results (at least 3 seeds - see
    course_context/TEACHER_EXPECTATIONS.md, and utils.DEFAULT_SEEDS).

    Parameters
    ----------
    metric_dicts : list of dict
        One metrics dict per seed, e.g. [regression_metrics(...) for
        seed_1, regression_metrics(...) for seed_2, ...]. Every dict
        must have the same keys.

    Returns
    -------
    dict
        {metric_name: {"mean": float, "std": float, "values": [...]}}
        for every metric key found in the input dicts.

    Raises
    ------
    ValueError
        If `metric_dicts` is empty, or if the dicts don't all share
        the same metric keys.
    """
    if not metric_dicts:
        raise ValueError("aggregate_across_seeds() needs at least one metrics dict.")

    keys = set(metric_dicts[0].keys())
    for d in metric_dicts:
        if set(d.keys()) != keys:
            raise ValueError(
                "All metric dicts must have the same keys to aggregate them - "
                f"got {set(d.keys())} and {keys}."
            )

    result = {}
    for key in keys:
        values = [d[key] for d in metric_dicts]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    return result


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

    IMPORTANT - nRMSE definition is a DELIBERATE CHOICE, not a spec
    requirement: the real project spec says to "always report RMSE
    together with nRMSE" but never defines exactly how to normalize it
    - this is genuinely ambiguous in the spec. This framework defaults
    to RANGE normalization (dividing by max(y_true) - min(y_true)),
    which is one of the two common conventions for nRMSE in
    forecasting literature (the other being MEAN normalization,
    available via `normalization="mean"`). Whichever one you use,
    use the SAME definition consistently across every Problem 2
    experiment, and state your choice explicitly in the report - don't
    let this be an unstated detail.

    IMPORTANT - always pass ORIGINAL (kW) scale values here, never
    values still in a standardized/scaled space: if you used
    preprocessing.fit_target_scaler() for a cross-city experiment,
    call preprocessing.inverse_transform_target() on both y_true and
    y_pred BEFORE calling this function. Otherwise RMSE/MAE/nRMSE end
    up in "standard deviations," not kW, which isn't interpretable and
    isn't what the spec asks for.

    Parameters
    ----------
    y_true : array-like
        Ground-truth continuous values, in the ORIGINAL (kW) scale.
    y_pred : array-like
        Predicted continuous values, in the ORIGINAL (kW) scale.
    normalization : str
        How to normalize RMSE into nRMSE:
        - "range" (default): divide by (max(y_true) - min(y_true)).
        - "mean": divide by mean(y_true).

    Returns
    -------
    dict
        {"rmse", "mae", "nrmse", "r2"} - lower is better for the first
        three, higher is better for r2 (R-squared: the fraction of the
        target's variance the model explains - 1.0 is a perfect fit,
        0.0 is no better than always predicting the mean, and it CAN
        go negative if a model is worse than that trivial baseline -
        worth watching for in cross-city zero-shot results, where a
        large scale mismatch can easily push R² negative).

    Raises
    ------
    ValueError
        If `normalization` is not "range" or "mean".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if normalization not in ("range", "mean"):
        raise ValueError(f"Unknown normalization '{normalization}'. Use 'range' or 'mean'.")

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    denominator = (y_true.max() - y_true.min()) if normalization == "range" else y_true.mean()
    nrmse = float(rmse / denominator) if denominator != 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "r2": r2}
