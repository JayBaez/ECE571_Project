"""
experiment_runner.py

Shared infrastructure for running and recording experiments. This is
NOT a "run any model automatically" engine - since no models exist yet
(Problems 1-5 haven't been implemented), this module only provides the
plumbing every future problem-specific runner script will need:

- load_config(): read an experiment's settings from a YAML file.
- generate_experiment_id(): a consistent, readable ID for one run.
- save_result(): append one experiment's result to the shared history
  CSV, WITHOUT overwriting previous results.

A future problems/problemN_*/ script will import these functions,
plus the actual model training code specific to that problem, and
call save_result() once training finishes with a REAL score.
"""

import csv
import os
from datetime import datetime, timezone

import yaml

RESULTS_HISTORY_PATH = os.path.join("results", "experiment_history.csv")

# Column order for the results history file. Matches the schema
# proposed during Phase 0 (course_context/EXPERIMENT_PLAN.md).
RESULT_FIELDS = [
    "experiment_id",
    "timestamp",
    "problem",
    "model",
    "dataset",
    "source_city",
    "target_city",
    "seed",
    "metric",
    "score",
    "runtime_seconds",
    "notes",
]


def load_config(path: str) -> dict:
    """
    Load an experiment configuration from a YAML file.

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
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: '{path}'")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_experiment_id(problem: str, model: str, seed: int) -> str:
    """
    Build a readable, likely-unique ID for one experiment run, e.g.
    "problem2_random_forest_seed42_20260101-153000".

    Parameters
    ----------
    problem : str
        e.g. "problem1", "problem2".
    model : str
        e.g. "random_forest", "mlp".
    seed : int
        The random seed used for this run.

    Returns
    -------
    str
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{problem}_{model}_seed{seed}_{timestamp}"


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

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    file_exists = os.path.exists(history_path)

    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
