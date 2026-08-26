"""
experiment_runner.py

Shared infrastructure for running and recording experiments:

- load_config() / validate_config(): read and sanity-check an
  experiment's settings from a YAML file.
- generate_experiment_id(): a consistent, readable ID for one run.
- create_experiment_dir(): a dedicated folder for one experiment's
  full artifacts (metrics, config, predictions, training log).
- save_result(): append one summary row to the shared results
  history CSV, WITHOUT overwriting previous results.
- get_leaderboard(): query the results history for the best results
  so far, correctly accounting for whether each metric is
  "higher is better" or "lower is better".

This is NOT a "run any model automatically" engine - since no models
exist yet (Problems 1-5 haven't been implemented), this module only
provides the plumbing every future problem-specific runner script will
need. A future problems/problemN_*/ script will import these
functions, plus the actual model training code specific to that
problem, and call save_result() once training finishes with a REAL
score.
"""

import csv
import json
import os
from datetime import datetime, timezone

import pandas as pd
import yaml

from src.evaluation import CLASSIFICATION_METRIC_DIRECTION, REGRESSION_METRIC_DIRECTION
from src.utils import ensure_dir

RESULTS_HISTORY_PATH = os.path.join("results", "experiment_history.csv")

# Column order for the results history file. Matches the schema
# proposed during Phase 0 (course_context/EXPERIMENT_PLAN.md), with
# "parameters" added in Phase 2 to record what model settings were used.
RESULT_FIELDS = [
    "experiment_id",
    "timestamp",
    "problem",
    "model",
    "dataset",
    "source_city",
    "target_city",
    "seed",
    "parameters",
    "metric",
    "score",
    "runtime_seconds",
    "notes",
]

# Combined metric-direction lookup, so the leaderboard knows whether to
# sort each metric ascending (lower is better, e.g. RMSE) or descending
# (higher is better, e.g. balanced_accuracy).
METRIC_DIRECTION = {**CLASSIFICATION_METRIC_DIRECTION, **REGRESSION_METRIC_DIRECTION}

# Config keys every experiment config must have - see
# configs/example_config.yaml for the full recommended shape.
REQUIRED_CONFIG_KEYS = ["problem", "model", "seed"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """
    Load an experiment configuration from a YAML file and check that
    it has the minimum required fields.

    Parameters
    ----------
    path : str
        Path to a .yaml config file, e.g. "configs/example_config.yaml".

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist.
    ValueError
        If the config is missing a required key - see validate_config().
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: '{path}'")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """
    Check that a config dict has the minimum fields every experiment
    needs. This is a deliberately small check - not a full schema
    validator - just enough to catch an obviously incomplete config
    early instead of failing confusingly partway through an experiment.

    Parameters
    ----------
    config : dict

    Raises
    ------
    ValueError
        If any of REQUIRED_CONFIG_KEYS is missing.
    """
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(
            f"Config is missing required key(s): {missing}. "
            f"Every config needs at least: {REQUIRED_CONFIG_KEYS}"
        )


# ---------------------------------------------------------------------------
# Experiment identity and artifact storage
# ---------------------------------------------------------------------------


def generate_experiment_id(problem: str, model: str, dataset: str, seed: int) -> str:
    """
    Build a readable, unique ID for one experiment run, e.g.
    "P2_Davis_random_forest_seed42_20260101-153000".

    Parameters
    ----------
    problem : str
        e.g. "P1", "P2", "problem2" - whatever short label the problem
        code uses consistently.
    model : str
        e.g. "random_forest", "mlp".
    dataset : str
        e.g. "Davis", "Davis_to_Amherst".
    seed : int
        The random seed used for this run.

    Returns
    -------
    str
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{problem}_{dataset}_{model}_seed{seed}_{timestamp}"


def create_experiment_dir(problem: str, experiment_id: str, base_dir: str = "results") -> str:
    """
    Create (if needed) and return a dedicated folder for one
    experiment's full artifacts, e.g.
    "results/problem2/P2_Davis_random_forest_seed42_20260101-153000/".

    Parameters
    ----------
    problem : str
        Matches a subfolder under `base_dir`, e.g. "problem2" or
        "framework_demo".
    experiment_id : str
        From generate_experiment_id().
    base_dir : str
        Default "results".

    Returns
    -------
    str
        Path to the created directory.
    """
    return ensure_dir(os.path.join(base_dir, problem, experiment_id))


def save_metrics_json(metrics: dict, experiment_dir: str, filename: str = "metrics.json") -> str:
    """
    Save a metrics dict as JSON inside an experiment's artifact folder.

    Parameters
    ----------
    metrics : dict
        e.g. from evaluation.regression_metrics() or
        evaluation.classification_metrics().
    experiment_dir : str
        From create_experiment_dir().
    filename : str

    Returns
    -------
    str
        The full path written.
    """
    path = os.path.join(experiment_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_experiment_config(config: dict, experiment_dir: str, filename: str = "config.yaml") -> str:
    """
    Save the exact configuration used for one experiment, so it can be
    reproduced later.

    Parameters
    ----------
    config : dict
    experiment_dir : str
        From create_experiment_dir().
    filename : str

    Returns
    -------
    str
        The full path written.
    """
    path = os.path.join(experiment_dir, filename)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def save_predictions_csv(y_true, y_pred, experiment_dir: str, filename: str = "predictions.csv") -> str:
    """
    Save a model's true/predicted values side by side, for later
    inspection or re-plotting without re-running the model.

    Parameters
    ----------
    y_true, y_pred : array-like
        Must be the same length.
    experiment_dir : str
        From create_experiment_dir().
    filename : str

    Returns
    -------
    str
        The full path written.
    """
    path = os.path.join(experiment_dir, filename)
    pd.DataFrame({"y_true": list(y_true), "y_pred": list(y_pred)}).to_csv(path, index=False)
    return path


def save_training_log(history: dict, experiment_dir: str, filename: str = "training_log.csv") -> str:
    """
    Save a neural network's per-epoch training history (only relevant
    for PyTorch experiments - see torch_utils.train_torch_model()).

    Parameters
    ----------
    history : dict
        {"epoch": [...], "train_loss": [...], "val_loss": [...]}
        from torch_utils.train_torch_model().
    experiment_dir : str
        From create_experiment_dir().
    filename : str

    Returns
    -------
    str
        The full path written.
    """
    path = os.path.join(experiment_dir, filename)
    pd.DataFrame(history).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Results history (one row per experiment, never overwritten)
# ---------------------------------------------------------------------------


def save_result(result: dict, history_path: str = RESULTS_HISTORY_PATH) -> None:
    """
    Append one experiment's result as a new row in the shared results
    history CSV. Never overwrites previous rows - every call adds a
    new line, so past experiments are always preserved (see
    course_context/AI_AGENT_INSTRUCTIONS.md, "Preserve previous
    experiment results").

    Parameters
    ----------
    result : dict
        Should contain the keys in RESULT_FIELDS. Any missing key is
        written as an empty value; unexpected extra keys are ignored
        by the CSV writer (extras are silently dropped, not an error -
        keep to RESULT_FIELDS to be safe).
    history_path : str
        Where to append the row. Defaults to
        "results/experiment_history.csv".

    Raises
    ------
    ValueError
        If `result` doesn't contain a "score" that looks like a real
        number - this is a safety check against accidentally logging
        a placeholder/fabricated result (see course_context/
        AI_AGENT_INSTRUCTIONS.md, "Never fabricate results").
    """
    if "score" not in result or result["score"] is None:
        raise ValueError(
            "save_result() requires a real 'score' from an executed "
            "experiment - refusing to save a result with no score."
        )

    ensure_dir(os.path.dirname(history_path))
    file_exists = os.path.exists(history_path)

    if file_exists:
        _check_history_header_matches(history_path)

    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def _check_history_header_matches(history_path: str) -> None:
    """
    Guard against silently misaligned columns: if RESULT_FIELDS ever
    changes (e.g. a new column is added, as happened once during this
    project - see course_context/PROJECT_STATUS.md), an existing CSV's
    header row would no longer match what DictWriter is about to
    append, and every value would silently shift into the wrong
    column. This check catches that immediately with a clear error
    instead of producing a corrupted-looking file.

    Parameters
    ----------
    history_path : str
        An existing results history CSV.

    Raises
    ------
    ValueError
        If the file's header row doesn't exactly match RESULT_FIELDS.
    """
    with open(history_path, "r", newline="") as f:
        existing_header = next(csv.reader(f), [])

    if existing_header != RESULT_FIELDS:
        raise ValueError(
            f"'{history_path}' has a different column schema than the current "
            f"RESULT_FIELDS.\n  File header:    {existing_header}\n"
            f"  Expected header: {RESULT_FIELDS}\n"
            "This usually means RESULT_FIELDS changed after this file was "
            "created. Rename/archive the old file (e.g. to "
            "experiment_history_old.csv) and start a fresh one, or manually "
            "add the missing column(s) to the existing file's header and "
            "backfill a value for every existing row, before appending more "
            "results - otherwise columns will silently misalign."
        )


# ---------------------------------------------------------------------------
# Leaderboard (a query over the results history, not a separately
# maintained file - see the Phase 2 report for why)
# ---------------------------------------------------------------------------


def get_leaderboard(
    metric: str,
    problem: str = None,
    top_n: int = 10,
    history_path: str = RESULTS_HISTORY_PATH,
    save_to: str = None,
) -> pd.DataFrame:
    """
    Get the best results so far for one metric, correctly sorted by
    whether that metric is "higher is better" or "lower is better".

    Design note: rather than maintaining a separate leaderboard.csv
    file that has to be kept in sync with experiment_history.csv every
    time a result is saved (two files that could silently drift apart
    is worse than one file queried on demand), the leaderboard is
    computed fresh from the always-complete experiment_history.csv
    every time this function is called. Pass `save_to` if you want to
    write a snapshot of the result to a file.

    Parameters
    ----------
    metric : str
        Which metric to rank by, e.g. "rmse", "balanced_accuracy".
        Must be a key in evaluation.CLASSIFICATION_METRIC_DIRECTION or
        evaluation.REGRESSION_METRIC_DIRECTION so the correct sort
        direction is known.
    problem : str, optional
        Only include rows for this problem, e.g. "problem2". Default:
        include all problems (only sensible if comparing the same
        metric across problems, e.g. two different regression tasks -
        don't compare RMSE from Problem 2 to accuracy from Problem 1).
    top_n : int
        How many rows to return.
    history_path : str
        Where to read results from. Defaults to
        "results/experiment_history.csv".
    save_to : str, optional
        If given, also write the resulting table to this path (a
        point-in-time leaderboard snapshot).

    Returns
    -------
    pandas.DataFrame
        Rows from the history file matching `metric` (and `problem` if
        given), sorted best-first, limited to `top_n` rows. Empty
        DataFrame (with the right columns) if no matching rows exist
        yet or the history file doesn't exist yet.

    Raises
    ------
    ValueError
        If `metric` isn't a known metric name.
    """
    if metric not in METRIC_DIRECTION:
        raise ValueError(
            f"Unknown metric '{metric}'. Known metrics: {list(METRIC_DIRECTION.keys())}"
        )

    if not os.path.exists(history_path):
        return pd.DataFrame(columns=RESULT_FIELDS)

    history = pd.read_csv(history_path)
    filtered = history[history["metric"] == metric]
    if problem is not None:
        filtered = filtered[filtered["problem"] == problem]

    ascending = METRIC_DIRECTION[metric] == "lower_better"
    result = filtered.sort_values("score", ascending=ascending).head(top_n).reset_index(drop=True)

    if save_to is not None:
        ensure_dir(os.path.dirname(save_to))
        result.to_csv(save_to, index=False)

    return result
