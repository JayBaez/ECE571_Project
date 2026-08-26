"""
Tests for src/experiment_runner.py. Uses pytest's tmp_path fixture for
all file writes, so these tests never touch the real results/ folder.
"""

import os

import pandas as pd
import pytest
import yaml

from src import experiment_runner


def test_validate_config_passes_with_required_keys():
    experiment_runner.validate_config({"problem": "problem2", "model": "random_forest", "seed": 42})


def test_validate_config_raises_when_missing_required_key():
    with pytest.raises(ValueError):
        experiment_runner.validate_config({"problem": "problem2"})


def test_load_config_reads_yaml_and_validates(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"problem": "problem2", "model": "random_forest", "seed": 42}))
    config = experiment_runner.load_config(str(config_path))
    assert config["problem"] == "problem2"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        experiment_runner.load_config("does/not/exist.yaml")


def test_generate_experiment_id_is_readable_and_unique_enough():
    id_a = experiment_runner.generate_experiment_id("problem2", "random_forest", "Davis", seed=42)
    assert "problem2" in id_a and "Davis" in id_a and "random_forest" in id_a and "seed42" in id_a


def test_create_experiment_dir_creates_folder(tmp_path):
    base = str(tmp_path)
    exp_dir = experiment_runner.create_experiment_dir("problem2", "exp_001", base_dir=base)
    assert os.path.isdir(exp_dir)
    assert exp_dir == os.path.join(base, "problem2", "exp_001")


def test_save_metrics_json_round_trips(tmp_path):
    metrics = {"rmse": 1.23, "mae": 0.98}
    path = experiment_runner.save_metrics_json(metrics, str(tmp_path))
    import json

    with open(path) as f:
        loaded = json.load(f)
    assert loaded == metrics


def test_save_experiment_config_round_trips(tmp_path):
    config = {"problem": "problem2", "model": "random_forest", "seed": 42}
    path = experiment_runner.save_experiment_config(config, str(tmp_path))
    with open(path) as f:
        loaded = yaml.safe_load(f)
    assert loaded == config


def test_save_predictions_csv_has_correct_columns(tmp_path):
    path = experiment_runner.save_predictions_csv([1.0, 2.0], [1.1, 1.9], str(tmp_path))
    df = pd.read_csv(path)
    assert list(df.columns) == ["y_true", "y_pred"]
    assert len(df) == 2


def test_save_result_requires_a_real_score(tmp_path):
    history_path = str(tmp_path / "history.csv")
    with pytest.raises(ValueError):
        experiment_runner.save_result({"score": None}, history_path=history_path)


def test_save_result_appends_without_overwriting(tmp_path):
    history_path = str(tmp_path / "history.csv")
    base_result = {field: "" for field in experiment_runner.RESULT_FIELDS}

    first = {**base_result, "experiment_id": "exp1", "metric": "rmse", "score": 1.0}
    second = {**base_result, "experiment_id": "exp2", "metric": "rmse", "score": 2.0}

    experiment_runner.save_result(first, history_path=history_path)
    experiment_runner.save_result(second, history_path=history_path)

    history = pd.read_csv(history_path)
    assert len(history) == 2
    assert list(history["experiment_id"]) == ["exp1", "exp2"]


def test_save_result_detects_stale_header_schema(tmp_path):
    history_path = str(tmp_path / "stale_history.csv")
    # Write a header missing one of the current RESULT_FIELDS, simulating
    # a file created before a schema change.
    stale_fields = [f for f in experiment_runner.RESULT_FIELDS if f != "parameters"]
    with open(history_path, "w") as f:
        f.write(",".join(stale_fields) + "\n")

    base_result = {field: "" for field in experiment_runner.RESULT_FIELDS}
    base_result["score"] = 1.0

    with pytest.raises(ValueError):
        experiment_runner.save_result(base_result, history_path=history_path)


def test_get_leaderboard_sorts_lower_better_metric_ascending(tmp_path):
    history_path = str(tmp_path / "history.csv")
    base_result = {field: "" for field in experiment_runner.RESULT_FIELDS}

    for exp_id, score in [("a", 5.0), ("b", 1.0), ("c", 3.0)]:
        experiment_runner.save_result(
            {**base_result, "experiment_id": exp_id, "problem": "problem2", "metric": "rmse", "score": score},
            history_path=history_path,
        )

    leaderboard = experiment_runner.get_leaderboard(metric="rmse", history_path=history_path)
    assert list(leaderboard["experiment_id"]) == ["b", "c", "a"]  # ascending: lower RMSE first


def test_get_leaderboard_sorts_higher_better_metric_descending(tmp_path):
    history_path = str(tmp_path / "history.csv")
    base_result = {field: "" for field in experiment_runner.RESULT_FIELDS}

    for exp_id, score in [("a", 0.5), ("b", 0.9), ("c", 0.7)]:
        experiment_runner.save_result(
            {**base_result, "experiment_id": exp_id, "problem": "problem1", "metric": "balanced_accuracy", "score": score},
            history_path=history_path,
        )

    leaderboard = experiment_runner.get_leaderboard(metric="balanced_accuracy", history_path=history_path)
    assert list(leaderboard["experiment_id"]) == ["b", "c", "a"]  # descending: higher accuracy first


def test_get_leaderboard_missing_history_file_returns_empty(tmp_path):
    leaderboard = experiment_runner.get_leaderboard(metric="rmse", history_path=str(tmp_path / "nope.csv"))
    assert len(leaderboard) == 0


def test_get_leaderboard_unknown_metric_raises():
    with pytest.raises(ValueError):
        experiment_runner.get_leaderboard(metric="not_a_real_metric")
