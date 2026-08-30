"""
run_experiments.py (Problem 1 — Supervised Classification)

Main experiment script for Problem 1. Runs both classification tasks
(sky-condition, generation-regime) for both cities (Davis, Amherst):

    1. Baseline model sweep (majority / logistic regression / decision
       tree / random forest / gradient boosting / MLP), 3 seeds each.
    2. A small hyperparameter search for random forest, gradient
       boosting, and the MLP (chronological inner validation split,
       not random cross-validation).
    3. A final tuned multi-seed evaluation using the best
       hyperparameters found.
    4. A class-weighting comparison (class_weight="balanced") for the
       3 models that support it directly.
    5. A small feature ablation study.
    6. Confusion matrices, per-class metrics, error analysis, and
       feature importance for the overall best model per task.
    7. Saves every result (never overwriting), the best models, and
       all figures.

Usage (from the project root):
    python problems/problem1_classification/run_experiments.py
"""

import csv
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import cleaning, data_loader, evaluation, experiment_runner, feature_engineering, preprocessing, splitting, torch_utils, utils, visualization
from problems.problem1_classification import features, models as p1_models, targets

SEEDS = utils.DEFAULT_SEEDS  # [42, 123, 2026]
CITIES = ["Davis", "Amherst"]
TASKS = ["sky_condition", "generation_regime"]
LABEL_COLUMN = {"sky_condition": "Sky_Condition", "generation_regime": "Generation_Regime"}
CLASS_ORDER = {"sky_condition": targets.SKY_CONDITION_CLASSES, "generation_regime": targets.GENERATION_REGIME_CLASSES}

RESULTS_DIR = "results/problem1"
MODELS_DIR = "results/problem1/models"
FIGURES_DIR = "figures/problem1"

# Results accumulate here across the whole run (wide format, one row
# per task/city/model/seed/ablation - see Section 24 of the Phase 4
# instructions for the required schema).
ALL_RESULTS = []


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_task_dataset(city: str, task: str, ablation_group: str = "full", include_leakage_risk: bool = False):
    """
    Load, clean, feature-engineer, label, split, and preprocess one
    city's data for one task. This is the single function every
    experiment in this script goes through, so every model is
    compared on an identical, correctly-built dataset.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test : preprocessed features / string labels
        feature_columns : list of str, exactly which encoded columns are in X
        preprocessor : dict, from preprocessing.fit_preprocessor()
        split_info : dict, from splitting.chronological_split()
        train_df, test_df : the labeled (pre-preprocessing) DataFrames,
            kept for error analysis later
        cleaning_report : dict or None
        class_distribution_train, class_distribution_test : DataFrame
    """
    label_col = LABEL_COLUMN[task]
    raw_df = data_loader.load_city(city, years="long")

    if task == "generation_regime":
        # Task B's target IS Output Power - missing target rows must be
        # dropped, never interpolated (interpolating a target and then
        # treating it as a real label would fabricate ground truth).
        cleaned_df, cleaning_report = cleaning.clean_sheet(
            raw_df, target_column="Output Power", missing_strategy="drop", verbose=False
        )
    else:
        # Task A's inputs (GHI, Clearsky GHI) have zero missing values
        # in this dataset (verified, course_context/DATASET_PROFILE.md)
        # - nothing to clean.
        cleaned_df, cleaning_report = raw_df.copy(), None

    featured_df = feature_engineering.add_feature_groups(cleaned_df, ["clear_sky_index", "time_cyclical"])

    if task == "sky_condition":
        # Fixed-threshold rule, not data-dependent -> safe to label
        # before splitting (see targets.py docstring).
        featured_df[label_col] = targets.make_sky_condition_labels(featured_df)
        train_df, test_df, split_info = splitting.chronological_split(featured_df, train_frac=0.8)
    else:
        # Terciles ARE data-dependent -> must split first, fit on train
        # only, then apply to both (see targets.py docstring).
        train_df, test_df, split_info = splitting.chronological_split(featured_df, train_frac=0.8)
        train_df, test_df = train_df.copy(), test_df.copy()
        low, high = targets.fit_tercile_boundaries(train_df["Output Power"])
        train_df[label_col] = targets.apply_tercile_labels(train_df["Output Power"], low, high)
        test_df[label_col] = targets.apply_tercile_labels(test_df["Output Power"], low, high)
        split_info = {**split_info, "tercile_low": low, "tercile_high": high}

    # Drop any row with a missing label (defensive - see targets.py;
    # not expected to trigger given Phase 0/3 findings, but a real
    # dataset issue would show up here instead of failing silently).
    train_df = train_df[train_df[label_col].notna()].copy()
    test_df = test_df[test_df[label_col].notna()].copy()

    feature_columns = features.get_feature_columns(task, ablation_group, include_leakage_risk)
    categorical_columns = [c for c in feature_columns if c in features.CATEGORICAL_COLUMNS]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    train_selected = train_df[feature_columns + [label_col]]
    test_selected = test_df[feature_columns + [label_col]]

    preprocessor = preprocessing.fit_preprocessor(train_selected, numeric_columns, categorical_columns)
    train_processed = preprocessing.apply_preprocessor(train_selected, preprocessor)
    test_processed = preprocessing.apply_preprocessor(test_selected, preprocessor)

    X_train, y_train, encoded_feature_columns = preprocessing.prepare_xy(train_processed, target_column=label_col)
    X_test, y_test, _ = preprocessing.prepare_xy(test_processed, target_column=label_col)

    return {
        "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        "feature_columns": encoded_feature_columns,
        "preprocessor": preprocessor,
        "split_info": split_info,
        "train_df": train_df, "test_df": test_df,
        "cleaning_report": cleaning_report,
    }


# ---------------------------------------------------------------------------
# Running one classical-model experiment
# ---------------------------------------------------------------------------


def run_classical_experiment(model, dataset: dict) -> tuple:
    """
    Fit a classical (scikit-learn) model and evaluate it.

    Returns
    -------
    (metrics, y_pred, fitted_model) : tuple
        metrics : dict from evaluation.classification_metrics()
        y_pred : numpy.ndarray of predicted string labels
        fitted_model : the trained estimator
    """
    model.fit(dataset["X_train"], dataset["y_train"])
    y_pred = model.predict(dataset["X_test"])
    metrics = evaluation.classification_metrics(dataset["y_test"], y_pred)
    return metrics, y_pred, model


# ---------------------------------------------------------------------------
# Running one MLP experiment
# ---------------------------------------------------------------------------


def run_mlp_experiment(dataset: dict, task: str, seed: int, hidden_size: int = 32,
                        dropout: float = 0.2, lr: float = 1e-3, max_epochs: int = 100,
                        patience: int = 10, verbose: bool = False) -> tuple:
    """
    Train and evaluate the SimpleMLPClassifier for one (task, city,
    seed) combination.

    The train set is further split chronologically (80/20, i.e. the
    LAST 20% of the already-chronological training data) to get a
    validation set for early stopping - this keeps validation data
    strictly earlier than the real test set while still respecting
    time order within training (see the Phase 3/4 instructions on
    chronological validation).

    Returns
    -------
    (metrics, y_pred, model, history) : tuple
    """
    utils.set_seed(seed)
    device = utils.get_device()
    class_order = CLASS_ORDER[task]
    class_to_index = {c: i for i, c in enumerate(class_order)}

    X_train_full = dataset["X_train"].values.astype(np.float32).copy()
    y_train_full = dataset["y_train"].map(class_to_index).values.astype(np.int64).copy()

    # Chronological inner validation split (last 20% of training data,
    # already in time order since it came from chronological_split).
    n_inner_train = int(len(X_train_full) * 0.8)
    X_inner_train, X_inner_val = X_train_full[:n_inner_train].copy(), X_train_full[n_inner_train:].copy()
    y_inner_train, y_inner_val = y_train_full[:n_inner_train].copy(), y_train_full[n_inner_train:].copy()

    train_loader = torch_utils.make_dataloader(X_inner_train, y_inner_train, batch_size=64, shuffle=True, y_dtype=torch.long)
    val_loader = torch_utils.make_dataloader(X_inner_val, y_inner_val, batch_size=64, shuffle=False, y_dtype=torch.long)

    model = p1_models.SimpleMLPClassifier(
        input_dim=X_train_full.shape[1], num_classes=len(class_order), hidden_size=hidden_size, dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = torch_utils.train_torch_model(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        max_epochs=max_epochs, patience=patience, verbose=verbose,
    )

    X_test = dataset["X_test"].values.astype(np.float32)
    y_pred_indices = p1_models.predict_classes(model, X_test, device)
    y_pred = np.array([class_order[i] for i in y_pred_indices])

    metrics = evaluation.classification_metrics(dataset["y_test"], y_pred)
    return metrics, y_pred, model, history


# ---------------------------------------------------------------------------
# Recording results
# ---------------------------------------------------------------------------


PROBLEM1_RESULTS_PATH = os.path.join(RESULTS_DIR, "problem1_results.csv")
PROBLEM1_RESULT_FIELDS = [
    "task", "city", "model", "seed", "ablation_group",
    "balanced_accuracy", "accuracy", "macro_precision", "macro_recall", "macro_f1", "notes",
]


def record_result(task, city, model_name, seed, metrics, ablation_group="full", notes=""):
    """
    Append one result row to results/problem1/problem1_results.csv
    (wide format, matching Section 24's required schema) AND to the
    shared cross-problem results history (headline metric only - see
    below). Writes to disk immediately (not just an in-memory list) so
    results/problem1_results.csv accumulates correctly even when this
    script's stages are run as separate process invocations (as they
    were during development, to stay within a single command's runtime
    limits - see course_context/PROBLEM1_REPORT.md).

    Only balanced_accuracy goes into the shared experiment_history.csv
    (not all 5 metrics) to keep that file's schema simple and
    consistent with how other future problems will use it - the full
    5-metric detail lives only in problem1_results.csv.
    """
    row = {
        "task": task, "city": city, "model": model_name, "seed": seed,
        "ablation_group": ablation_group,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["precision_macro"],
        "macro_recall": metrics["recall_macro"],
        "macro_f1": metrics["f1_macro"],
        "notes": notes,
    }
    ALL_RESULTS.append(row)

    utils.ensure_dir(RESULTS_DIR)
    file_exists = os.path.exists(PROBLEM1_RESULTS_PATH)
    with open(PROBLEM1_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROBLEM1_RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    experiment_id = experiment_runner.generate_experiment_id("problem1", model_name, f"{city}_{task}", seed)
    experiment_runner.save_result({
        "experiment_id": experiment_id,
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "problem": "problem1",
        "model": model_name,
        "dataset": f"{city}_{task}_{ablation_group}",
        "source_city": city,
        "target_city": "",
        "seed": seed,
        "parameters": "{}",
        "metric": "balanced_accuracy",
        "score": metrics["balanced_accuracy"],
        "runtime_seconds": "",
        "notes": notes,
    })


# ---------------------------------------------------------------------------
# Stage 1: baseline model sweep
# ---------------------------------------------------------------------------


def run_baseline_sweep():
    """
    Section 13: baseline -> logistic regression -> decision tree ->
    random forest -> gradient boosting -> MLP, for every (task, city),
    at every seed, on the FULL (primary, non-ablation) feature set.

    Returns
    -------
    dict of {(task, city): {model_name: [metrics_per_seed]}} - used to
    pick which model type to hyperparameter-tune next.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: BASELINE MODEL SWEEP")
    print("=" * 70)

    sweep_results = {}
    for task in TASKS:
        for city in CITIES:
            print(f"\n--- {task} / {city} ---")
            dataset = build_task_dataset(city, task, ablation_group="full")
            sweep_results[(task, city)] = {}

            for seed in SEEDS:
                classical_models = p1_models.get_classical_models(seed)
                for model_name, model in classical_models.items():
                    metrics, y_pred, fitted = run_classical_experiment(model, dataset)
                    record_result(task, city, model_name, seed, metrics)
                    sweep_results[(task, city)].setdefault(model_name, []).append(metrics)
                    print(f"  seed={seed} {model_name:20s} balanced_acc={metrics['balanced_accuracy']:.3f}")

                metrics, y_pred, model, history = run_mlp_experiment(dataset, task, seed, max_epochs=60, patience=8)
                record_result(task, city, "mlp", seed, metrics)
                sweep_results[(task, city)].setdefault("mlp", []).append(metrics)
                print(f"  seed={seed} {'mlp':20s} balanced_acc={metrics['balanced_accuracy']:.3f}")

    return sweep_results


# ---------------------------------------------------------------------------
# Stage 2: small hyperparameter search (chronological inner validation)
# ---------------------------------------------------------------------------

RF_SEARCH_SPACE = [
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2},
]
GB_SEARCH_SPACE = [
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
    {"n_estimators": 150, "learning_rate": 0.2, "max_depth": 2},
]
MLP_SEARCH_SPACE = [
    {"hidden_size": 32, "dropout": 0.2, "lr": 1e-3},
    {"hidden_size": 64, "dropout": 0.3, "lr": 1e-3},
    {"hidden_size": 16, "dropout": 0.1, "lr": 1e-3},
    {"hidden_size": 32, "dropout": 0.2, "lr": 5e-4},
]


def chronological_inner_split(X_train, y_train, val_frac=0.2):
    """
    Split an already-chronologically-ordered training set into an
    earlier inner-train portion and a later inner-validation portion,
    for hyperparameter selection without ever touching the real test
    set - see the Phase 4 instructions' "time-series aware validation"
    requirement.
    """
    n_inner_train = int(len(X_train) * (1 - val_frac))
    return (
        X_train.iloc[:n_inner_train], y_train.iloc[:n_inner_train],
        X_train.iloc[n_inner_train:], y_train.iloc[n_inner_train:],
    )


def search_rf_gb_hyperparameters(dataset: dict, search_seed: int = 42) -> dict:
    """
    Try each candidate config in RF_SEARCH_SPACE / GB_SEARCH_SPACE on
    the chronological inner validation split, and return the
    best-performing config for each model type (by balanced accuracy).

    Returns
    -------
    dict: {"random_forest": {...best params...}, "gradient_boosting": {...}}
    """
    X_inner_train, y_inner_train, X_inner_val, y_inner_val = chronological_inner_split(
        dataset["X_train"], dataset["y_train"]
    )

    best = {}
    for model_name, search_space, param_key in [
        ("random_forest", RF_SEARCH_SPACE, "rf_params"),
        ("gradient_boosting", GB_SEARCH_SPACE, "gb_params"),
    ]:
        best_score, best_params = -1.0, None
        for params in search_space:
            model = p1_models.get_classical_models(search_seed, **{param_key: params})[model_name]
            model.fit(X_inner_train, y_inner_train)
            y_val_pred = model.predict(X_inner_val)
            score = evaluation.classification_metrics(y_inner_val, y_val_pred)["balanced_accuracy"]
            if score > best_score:
                best_score, best_params = score, params
        best[model_name] = best_params
        print(f"    {model_name}: best={best_params} (inner-val balanced_acc={best_score:.3f})")

    return best


def search_mlp_hyperparameters(dataset: dict, task: str, search_seed: int = 42) -> dict:
    """
    Try each candidate config in MLP_SEARCH_SPACE, trained on the
    chronological inner-train split and scored on inner-validation
    (never the real test set) - same protocol as
    search_rf_gb_hyperparameters().
    """
    class_order = CLASS_ORDER[task]
    class_to_index = {c: i for i, c in enumerate(class_order)}

    X_train_full = dataset["X_train"].values.astype(np.float32).copy()
    y_train_full = dataset["y_train"].map(class_to_index).values.astype(np.int64).copy()
    n_inner_train = int(len(X_train_full) * 0.8)
    X_inner_train, X_inner_val = X_train_full[:n_inner_train].copy(), X_train_full[n_inner_train:].copy()
    y_inner_train, y_inner_val = y_train_full[:n_inner_train].copy(), y_train_full[n_inner_train:].copy()

    best_score, best_params = -1.0, None
    for params in MLP_SEARCH_SPACE:
        utils.set_seed(search_seed)
        device = utils.get_device()
        train_loader = torch_utils.make_dataloader(X_inner_train, y_inner_train, batch_size=64, shuffle=True, y_dtype=torch.long)
        val_loader = torch_utils.make_dataloader(X_inner_val, y_inner_val, batch_size=64, shuffle=False, y_dtype=torch.long)

        model = p1_models.SimpleMLPClassifier(
            input_dim=X_train_full.shape[1], num_classes=len(class_order),
            hidden_size=params["hidden_size"], dropout=params["dropout"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        torch_utils.train_torch_model(
            model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), device,
            max_epochs=40, patience=6, verbose=False,
        )
        y_val_pred_idx = p1_models.predict_classes(model, X_inner_val, device)
        y_val_pred = np.array([class_order[i] for i in y_val_pred_idx])
        y_val_true = np.array([class_order[i] for i in y_inner_val])
        score = evaluation.classification_metrics(y_val_true, y_val_pred)["balanced_accuracy"]
        if score > best_score:
            best_score, best_params = score, params

    print(f"    mlp: best={best_params} (inner-val balanced_acc={best_score:.3f})")
    return best_params


def run_hyperparameter_search():
    """
    Section 17: small hyperparameter search for random_forest,
    gradient_boosting, and mlp, per (task, city), using chronological
    inner validation (never the real test set).

    Returns
    -------
    dict: {"task|city": {"random_forest": {...}, "gradient_boosting": {...}, "mlp": {...}}}
    """
    print("\n" + "=" * 70)
    print("STAGE 2: HYPERPARAMETER SEARCH (chronological inner validation)")
    print("=" * 70)

    best_params_by_combo = {}
    for task in TASKS:
        for city in CITIES:
            print(f"\n--- {task} / {city} ---")
            dataset = build_task_dataset(city, task, ablation_group="full")
            rf_gb_best = search_rf_gb_hyperparameters(dataset)
            mlp_best = search_mlp_hyperparameters(dataset, task)
            best_params_by_combo[f"{task}|{city}"] = {**rf_gb_best, "mlp": mlp_best}

    return best_params_by_combo


# ---------------------------------------------------------------------------
# Stage 3: final tuned multi-seed evaluation
# ---------------------------------------------------------------------------


def run_tuned_final_evaluation(best_params_by_combo: dict, task: str, city: str):
    """
    Re-run random_forest, gradient_boosting, and mlp with the tuned
    hyperparameters found in Stage 2, at all 3 seeds, evaluated on the
    REAL test set. Logistic regression and decision tree keep their
    Stage 1 default-hyperparameter results (they weren't part of the
    tuning scope per Section 17's example search spaces) - both are
    already recorded in results/problem1_results.csv from Stage 1.

    Parameters
    ----------
    best_params_by_combo : dict
        From run_hyperparameter_search() (or loaded from the saved
        results/problem1/hyperparameter_search_results.json).
    task, city : str
        Run for one combo at a time (kept separate, not looped over
        internally, so this stays within a single command's runtime
        limit - see the notes in course_context/PROBLEM1_REPORT.md).
    """
    print(f"\n--- Stage 3 (tuned final eval): {task} / {city} ---")
    dataset = build_task_dataset(city, task, ablation_group="full")
    tuned = best_params_by_combo[f"{task}|{city}"]

    for seed in SEEDS:
        rf = p1_models.get_classical_models(seed, rf_params=tuned["random_forest"])["random_forest"]
        metrics, _, _ = run_classical_experiment(rf, dataset)
        record_result(task, city, "random_forest_tuned", seed, metrics, notes=json.dumps(tuned["random_forest"]))
        print(f"  seed={seed} random_forest_tuned  balanced_acc={metrics['balanced_accuracy']:.3f}")

        gb = p1_models.get_classical_models(seed, gb_params=tuned["gradient_boosting"])["gradient_boosting"]
        metrics, _, _ = run_classical_experiment(gb, dataset)
        record_result(task, city, "gradient_boosting_tuned", seed, metrics, notes=json.dumps(tuned["gradient_boosting"]))
        print(f"  seed={seed} gradient_boosting_tuned  balanced_acc={metrics['balanced_accuracy']:.3f}")

        metrics, _, _, _ = run_mlp_experiment(dataset, task, seed, max_epochs=80, patience=10, **tuned["mlp"])
        record_result(task, city, "mlp_tuned", seed, metrics, notes=json.dumps(tuned["mlp"]))
        print(f"  seed={seed} mlp_tuned  balanced_acc={metrics['balanced_accuracy']:.3f}")


# ---------------------------------------------------------------------------
# Stage 4: class-weighting comparison
# ---------------------------------------------------------------------------


def run_class_weighting_comparison(task: str, city: str):
    """
    Section 12: test class_weight="balanced" for the 3 models that
    support it directly (logistic_regression, decision_tree,
    random_forest - see models.py's CLASS_WEIGHT_CAPABLE_MODELS), at
    all 3 seeds, on the default (untuned) hyperparameters for a clean
    apples-to-apples comparison against Stage 1's unweighted results.
    """
    print(f"\n--- Stage 4 (class weighting): {task} / {city} ---")
    dataset = build_task_dataset(city, task, ablation_group="full")

    for seed in SEEDS:
        weighted_models = p1_models.get_classical_models(seed, class_weight="balanced")
        for model_name in p1_models.CLASS_WEIGHT_CAPABLE_MODELS:
            metrics, _, _ = run_classical_experiment(weighted_models[model_name], dataset)
            record_result(task, city, f"{model_name}_balanced_weight", seed, metrics, notes="class_weight=balanced")
            print(f"  seed={seed} {model_name}_balanced_weight  balanced_acc={metrics['balanced_accuracy']:.3f}")


# ---------------------------------------------------------------------------
# Stage 5: feature ablation
# ---------------------------------------------------------------------------

ABLATION_GROUPS = ["time_only", "weather_only", "irradiance_weather", "full"]


def run_feature_ablation(task: str, city: str):
    """
    Section 22: a small feature ablation using Random Forest as the
    single consistent "workhorse" model across every ablation group
    and every (task, city) - chosen because it was competitive
    (top-2) in 3 of the 4 combos in Stage 1/3's results and gives
    clean, comparable numbers across groups (using a different "best"
    model per combo here would make the ablation groups themselves
    harder to compare). This is a deliberate simplification, not an
    oversight - documented in course_context/PROBLEM1_REPORT.md.

    For sky_condition, also runs one extra group -
    "full_with_leakage_risk" - which adds back DHI/DNI/Solar Zenith
    Angle (excluded from every other group per the resolved leakage
    decision) specifically to quantify the leakage effect discussed in
    course_context/LEAKAGE_MAP.md.
    """
    print(f"\n--- Stage 5 (feature ablation): {task} / {city} ---")

    for ablation_group in ABLATION_GROUPS:
        dataset = build_task_dataset(city, task, ablation_group=ablation_group)
        for seed in SEEDS:
            model = p1_models.get_classical_models(seed)["random_forest"]
            metrics, _, _ = run_classical_experiment(model, dataset)
            record_result(task, city, "random_forest", seed, metrics, ablation_group=ablation_group)
        mean_score = np.mean([r["balanced_accuracy"] for r in ALL_RESULTS
                               if r["task"] == task and r["city"] == city
                               and r["ablation_group"] == ablation_group and r["model"] == "random_forest"])
        print(f"  {ablation_group:22s} mean balanced_acc={mean_score:.3f} (n_features={len(dataset['feature_columns'])})")

    if task == "sky_condition":
        dataset = build_task_dataset(city, task, ablation_group="full", include_leakage_risk=True)
        for seed in SEEDS:
            model = p1_models.get_classical_models(seed)["random_forest"]
            metrics, _, _ = run_classical_experiment(model, dataset)
            record_result(task, city, "random_forest", seed, metrics, ablation_group="full_with_leakage_risk",
                           notes="includes DHI/DNI/Solar Zenith Angle - leakage demonstration only, not the primary result")
        mean_score = np.mean([r["balanced_accuracy"] for r in ALL_RESULTS
                               if r["task"] == task and r["city"] == city
                               and r["ablation_group"] == "full_with_leakage_risk"])
        print(f"  {'full_with_leakage_risk':22s} mean balanced_acc={mean_score:.3f} (n_features={len(dataset['feature_columns'])})")


# ---------------------------------------------------------------------------
# Best-model selection (for Stage 6)
# ---------------------------------------------------------------------------


def select_best_model(task: str, city: str, results_path: str = PROBLEM1_RESULTS_PATH) -> tuple:
    """
    Read results/problem1/problem1_results.csv and return the
    best-performing model for one (task, city), by mean balanced
    accuracy across seeds, restricted to the "full" ablation group
    (the primary, non-ablation feature set) so ablation-group rows
    don't accidentally win by comparing a smaller feature set.

    Returns
    -------
    (model_name, mean_balanced_accuracy) : tuple
    """
    df = pd.read_csv(results_path)
    subset = df[(df["task"] == task) & (df["city"] == city) & (df["ablation_group"] == "full")]
    summary = subset.groupby("model")["balanced_accuracy"].mean().sort_values(ascending=False)
    return summary.index[0], float(summary.iloc[0])


# ---------------------------------------------------------------------------
# Stage 6: detailed analysis for the best model of each (task, city)
# ---------------------------------------------------------------------------


def retrain_named_model(model_name: str, dataset: dict, task: str, city: str, seed: int, hp_search_results: dict):
    """
    Reconstruct and retrain whichever specific model configuration a
    results-table `model` name refers to (e.g. "random_forest_tuned",
    "logistic_regression_balanced_weight", "mlp"), so Stage 6 can
    regenerate predictions from the actual best-performing setup
    identified by select_best_model(), not just a generic default.

    Returns
    -------
    (y_pred, model_object, model_kind) : tuple
        model_kind is "sklearn" or "torch" - tells the caller how to
        get feature importance later.
    """
    tuned = hp_search_results.get(f"{task}|{city}", {})

    if model_name == "mlp" or model_name == "mlp_tuned":
        params = tuned.get("mlp", {}) if model_name == "mlp_tuned" else {}
        metrics, y_pred, model, _ = run_mlp_experiment(dataset, task, seed, max_epochs=80, patience=10, **params)
        return y_pred, model, "torch"

    class_weight = "balanced" if model_name.endswith("_balanced_weight") else None
    base_name = model_name.replace("_tuned", "").replace("_balanced_weight", "")
    rf_params = tuned.get("random_forest", {}) if model_name == "random_forest_tuned" else {}
    gb_params = tuned.get("gradient_boosting", {}) if model_name == "gradient_boosting_tuned" else {}

    model = p1_models.get_classical_models(seed, class_weight=class_weight, rf_params=rf_params, gb_params=gb_params)[base_name]
    model.fit(dataset["X_train"], dataset["y_train"])
    y_pred = model.predict(dataset["X_test"])
    return y_pred, model, "sklearn"


def compute_permutation_importance(predict_fn, X_test, y_test, n_repeats: int = 5, seed: int = 42) -> np.ndarray:
    """
    Simple, explicit permutation importance: for each feature column,
    shuffle its values across test rows (breaking its relationship
    with the target while leaving every other column untouched), see
    how much balanced accuracy drops, and average over a few repeats.
    A feature the model actually relies on causes a big drop when
    scrambled; a feature it ignores causes almost no drop.

    Works identically for any model type - the caller supplies
    `predict_fn`, a plain function that takes a feature array and
    returns predicted labels (e.g. a scikit-learn model's `.predict`,
    or a small wrapper around the PyTorch MLP's forward pass - see
    analyze_best_model() for both usages). This avoids needing a
    scikit-learn-compatible wrapper class just to reuse
    `sklearn.inspection.permutation_importance()` - one direct,
    ~15-line function does the same job for every model in this
    project.

    Parameters
    ----------
    predict_fn : callable
        `predict_fn(X) -> y_pred`.
    X_test : pandas.DataFrame or numpy.ndarray
    y_test : array-like
    n_repeats : int
    seed : int

    Returns
    -------
    numpy.ndarray
        One importance value per feature column, same order as
        `X_test`'s columns.
    """
    rng = np.random.default_rng(seed)
    column_names = list(X_test.columns) if hasattr(X_test, "columns") else None
    X_array = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)

    def as_model_input(array):
        # Preserve column names when the caller's predict_fn wraps a
        # scikit-learn model fit on a DataFrame - otherwise scikit-learn
        # prints a harmless but noisy "X does not have valid feature
        # names" warning on every single prediction call here.
        return pd.DataFrame(array, columns=column_names) if column_names is not None else array

    baseline_pred = predict_fn(as_model_input(X_array))
    baseline_score = evaluation.classification_metrics(y_test, baseline_pred)["balanced_accuracy"]

    importances = []
    for col_index in range(X_array.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_array.copy()
            rng.shuffle(X_permuted[:, col_index])
            permuted_pred = predict_fn(as_model_input(X_permuted))
            permuted_score = evaluation.classification_metrics(y_test, permuted_pred)["balanced_accuracy"]
            drops.append(baseline_score - permuted_score)
        importances.append(float(np.mean(drops)))

    return np.array(importances)


def analyze_best_model(task: str, city: str, hp_search_results: dict, seed: int = 42):
    """
    For the actual best model of one (task, city) - identified by
    select_best_model() from the real recorded results - produce:
    a confusion matrix figure, a per-class metrics table, a small
    error-analysis table, and a permutation-importance figure.

    Permutation importance (not built-in tree impurity importance) is
    used for ALL model types here, including logistic regression and
    the MLP, since the actual best model differs by type across the
    four (task, city) combos - permutation importance works
    identically regardless of model type, giving one consistent method
    to compare across combos (Section 23's suggestion).
    """
    print(f"\n--- Stage 6 (detailed analysis): {task} / {city} ---")
    model_name, mean_score = select_best_model(task, city)
    print(f"  Best model: {model_name} (mean balanced_accuracy={mean_score:.4f})")

    dataset = build_task_dataset(city, task, ablation_group="full")
    y_pred, model, model_kind = retrain_named_model(model_name, dataset, task, city, seed, hp_search_results)
    y_test = dataset["y_test"]
    class_order = CLASS_ORDER[task]

    # Confusion matrix
    cm = evaluation.get_confusion_matrix(y_test, y_pred, labels=class_order)
    cm_path = os.path.join(FIGURES_DIR, f"problem1_{task}_confusion_matrix_{city}.png")
    visualization.plot_confusion_matrix(cm, labels=class_order, save_path=cm_path,
                                         title=f"{task.replace('_', ' ').title()} — {city} ({model_name})")
    print(f"  Saved {cm_path}")

    # Per-class metrics (precision/recall/f1/support)
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=class_order, zero_division=0)
    per_class_table = pd.DataFrame({
        "class": class_order, "precision": precision.round(3), "recall": recall.round(3),
        "f1": f1.round(3), "support": support,
    })
    per_class_path = os.path.join(RESULTS_DIR, f"per_class_metrics_{task}_{city}.csv")
    per_class_table.to_csv(per_class_path, index=False)
    print(f"  Per-class metrics:\n{per_class_table.to_string(index=False)}")

    hardest_class = per_class_table.loc[per_class_table["f1"].idxmin(), "class"]
    print(f"  Hardest class (lowest F1): {hardest_class}")

    # Error analysis: look at misclassified rows and check for patterns.
    test_df = dataset["test_df"].reset_index(drop=True)
    label_col = LABEL_COLUMN[task]
    errors_mask = (y_pred != y_test.values)
    error_df = test_df.loc[errors_mask.values if hasattr(errors_mask, "values") else errors_mask].copy()
    error_df["predicted"] = np.array(y_pred)[errors_mask.values if hasattr(errors_mask, "values") else errors_mask]

    error_summary = {
        "total_test_rows": len(test_df),
        "total_errors": int(errors_mask.sum()),
        "error_rate": float(errors_mask.sum() / len(test_df)),
    }
    if task == "sky_condition" and "Clear_Sky_Index" in error_df.columns:
        error_summary["mean_clear_sky_index_on_errors"] = float(error_df["Clear_Sky_Index"].mean())
        error_summary["mean_clear_sky_index_overall"] = float(test_df["Clear_Sky_Index"].mean())
        # Are errors concentrated near the 0.4/0.85 decision boundaries?
        near_boundary = ((error_df["Clear_Sky_Index"] - 0.4).abs() < 0.05) | ((error_df["Clear_Sky_Index"] - 0.85).abs() < 0.05)
        error_summary["pct_errors_near_threshold_boundary"] = float(near_boundary.mean()) if len(error_df) else 0.0
    if task == "generation_regime" and "Output Power" in error_df.columns:
        error_summary["mean_output_power_on_errors"] = float(error_df["Output Power"].mean())
        error_summary["mean_output_power_overall"] = float(test_df["Output Power"].mean())

    print(f"  Error analysis: {error_summary}")
    error_summary_path = os.path.join(RESULTS_DIR, f"error_analysis_{task}_{city}.json")
    with open(error_summary_path, "w") as f:
        json.dump(error_summary, f, indent=2)

    # A small sample table of actual misclassified rows, saved for the report.
    sample_cols = [c for c in [label_col, "predicted", "Clear_Sky_Index", "Output Power", "Cloud Type"] if c in error_df.columns or c == "predicted"]
    error_sample = error_df[sample_cols].head(15) if len(error_df) else pd.DataFrame(columns=sample_cols)
    error_sample.to_csv(os.path.join(RESULTS_DIR, f"error_sample_{task}_{city}.csv"), index=False)

    # Permutation importance, implemented directly (not via
    # sklearn.inspection.permutation_importance) so the exact same
    # simple logic works for both scikit-learn models and the
    # PyTorch MLP, without needing a wrapper class to satisfy
    # scikit-learn's estimator interface. The idea: shuffle one
    # feature column at a time, see how much balanced accuracy drops,
    # repeat a few times and average - a feature that matters a lot
    # causes a big drop when scrambled; one that doesn't matter causes
    # almost no drop.
    if model_kind == "sklearn":
        predict_fn = model.predict
    else:
        device = utils.get_device()

        def predict_fn(X):
            idx = p1_models.predict_classes(model, np.asarray(X, dtype=np.float32), device)
            return np.array([class_order[i] for i in idx])

    importances = compute_permutation_importance(
        predict_fn, dataset["X_test"], y_test, n_repeats=5, seed=seed
    )

    importance_table = pd.DataFrame({
        "feature": dataset["feature_columns"], "importance": importances
    }).sort_values("importance", ascending=False)
    importance_table.to_csv(os.path.join(RESULTS_DIR, f"feature_importance_{task}_{city}.csv"), index=False)

    top_features = importance_table.head(12)
    fig_path = os.path.join(FIGURES_DIR, f"problem1_{task}_feature_importance_{city}.png")
    visualization.set_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_features["feature"][::-1], top_features["importance"][::-1])
    ax.set_xlabel("Permutation importance (drop in balanced accuracy)")
    ax.set_title(f"{task.replace('_', ' ').title()} — {city} — top features ({model_name})")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")

    return {
        "model_name": model_name, "mean_score": mean_score, "per_class_table": per_class_table,
        "error_summary": error_summary, "importance_table": importance_table,
    }


# ---------------------------------------------------------------------------
# Save best models + preprocessors, and summary figures
# ---------------------------------------------------------------------------


def save_best_model_artifacts(task: str, city: str, hp_search_results: dict, seed: int = 42) -> dict:
    """
    Section 25: retrain the actual best model (identified from the
    real recorded results) one more time and save it, plus its
    preprocessor, to results/problem1/models/, so it can be loaded
    later without re-running the whole experiment sweep.
    """
    model_name, mean_score = select_best_model(task, city)
    dataset = build_task_dataset(city, task, ablation_group="full")
    y_pred, model, model_kind = retrain_named_model(model_name, dataset, task, city, seed, hp_search_results)

    utils.ensure_dir(MODELS_DIR)
    preprocessor_path = os.path.join(MODELS_DIR, f"{task}_{city}_preprocessor.joblib")
    joblib.dump(dataset["preprocessor"], preprocessor_path)

    if model_kind == "sklearn":
        model_path = os.path.join(MODELS_DIR, f"{task}_{city}_best_model.joblib")
        joblib.dump(model, model_path)
        model_params = model.get_params()
    else:
        model_path = os.path.join(MODELS_DIR, f"{task}_{city}_best_model_mlp.pt")
        torch.save(model.state_dict(), model_path)
        # Record the exact architecture so a future loader doesn't have
        # to guess it from the saved weight shapes (works, but this is
        # more direct/reliable) - "mlp" uses defaults, "mlp_tuned" uses
        # the hyperparameter-search result for this (task, city).
        tuned_mlp_params = hp_search_results.get(f"{task}|{city}", {}).get("mlp", {})
        defaults = {"hidden_size": 32, "dropout": 0.2, "lr": 1e-3}
        model_params = {**defaults, **tuned_mlp_params} if model_name == "mlp_tuned" else defaults
        model_params["input_dim"] = dataset["X_train"].shape[1]
        model_params["num_classes"] = len(CLASS_ORDER[task])

    return {
        "task": task, "city": city, "model_name": model_name, "model_kind": model_kind,
        "mean_balanced_accuracy": mean_score, "model_path": model_path, "preprocessor_path": preprocessor_path,
        "feature_columns": dataset["feature_columns"], "model_params": model_params,
    }


def make_class_distribution_figure():
    """Section 26: problem1_class_distribution.png - class balance for both tasks, both cities, in one figure."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, task in enumerate(TASKS):
        for col, city in enumerate(CITIES):
            dataset = build_task_dataset(city, task, ablation_group="full")
            label_col = LABEL_COLUMN[task]
            all_labels = pd.concat([dataset["train_df"][label_col], dataset["test_df"][label_col]])
            counts = all_labels.value_counts().reindex(CLASS_ORDER[task])
            axes[row, col].bar(counts.index, counts.values)
            axes[row, col].set_title(f"{task.replace('_', ' ').title()} — {city}")
            axes[row, col].set_ylabel("Row count")
    fig.suptitle("Class distribution by task and city (train + test combined)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem1_class_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_model_comparison_figure():
    """Section 26: problem1_model_comparison.png - mean balanced accuracy per model, per (task, city), full feature set only."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM1_RESULTS_PATH)
    df = df[df["ablation_group"] == "full"]
    summary = df.groupby(["task", "city", "model"])["balanced_accuracy"].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, task in enumerate(TASKS):
        for col, city in enumerate(CITIES):
            subset = summary[(summary["task"] == task) & (summary["city"] == city)].sort_values("balanced_accuracy")
            axes[row, col].barh(subset["model"], subset["balanced_accuracy"])
            axes[row, col].set_title(f"{task.replace('_', ' ').title()} — {city}")
            axes[row, col].set_xlabel("Mean balanced accuracy")
            axes[row, col].set_xlim(0, 1)
    fig.suptitle("Model comparison — mean balanced accuracy across 3 seeds (full feature set)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem1_model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
