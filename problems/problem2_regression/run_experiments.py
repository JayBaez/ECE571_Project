"""
run_experiments.py (Problem 2 — Supervised Regression)

Main experiment script for Problem 2, the project's "core" task:

    1. Same-city regression (Davis, Amherst): baseline model sweep,
       hyperparameter search, tuned final multi-seed evaluation.
    2. Cross-city zero-shot: Davis (source) -> Huron / Santa Barbara /
       La Jolla (targets), zero target-city training.
    3. 3-year vs 6-year ablation (Davis).
    4. Sequence forecasting (K=12, GRU) vs. a persistence baseline.
    5. Learning curve, prediction-vs-truth plots, error analysis.
    6. Saves every result (never overwriting), the best models, and
       all figures.

Usage (from the project root):
    python problems/problem2_regression/run_experiments.py
Note: in practice this script's stages were run as separate, smaller
invocations during development, to stay within a single command's
execution-time limit - see course_context/PROBLEM2_REPORT.md.
"""

import csv
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import cleaning, data_loader, evaluation, experiment_runner, feature_engineering, preprocessing, splitting, torch_utils, utils, visualization
from problems.problem2_regression import features, models as p2_models, sequence

SEEDS = utils.DEFAULT_SEEDS  # [42, 123, 2026]
SAME_CITY_CITIES = ["Davis", "Amherst"]
CROSS_CITY_TARGETS = ["Huron", "Santa Barbara", "La Jolla"]
SOURCE_CITY = "Davis"

RESULTS_DIR = "results/problem2"
MODELS_DIR = "results/problem2/models"
FIGURES_DIR = "figures/problem2"

PROBLEM2_RESULTS_PATH = os.path.join(RESULTS_DIR, "problem2_results.csv")
PROBLEM2_RESULT_FIELDS = [
    "experiment", "city", "source_city", "target_city", "model", "model_type",
    "seed", "split_type", "feature_set", "target_scaling",
    "rmse", "mae", "nrmse", "r2", "notes",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_city_dataset(city: str, years: str = "long") -> dict:
    """
    Load, clean, feature-engineer, chronologically split, and
    preprocess one city's data for the non-sequence regression
    experiments (same-city, cross-city, 3yr-vs-6yr).

    Missing Output Power rows are DROPPED, not interpolated -
    consistent with Problem 1's precedent
    (course_context/PROBLEM1_REPORT.md, Section 8): interpolating a
    value and then training a model to treat it as real ground truth
    would fabricate data, which is a bigger concern for a small
    dataset (Amherst: 4 rows) than the minor loss of those 4 rows.

    Parameters
    ----------
    city : str
    years : str
        "long" (default, 6-year sheet where available) or "short"
        (3-year sheet) - see src/data_loader.py's load_city(). Used by
        the 3yr-vs-6yr ablation.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test : preprocessed features / raw-kW target
        feature_columns : list of str
        preprocessor : dict, from preprocessing.fit_preprocessor()
        split_info : dict, from splitting.chronological_split()
        train_df, test_df : labeled (pre-preprocessing) DataFrames, kept for
            error analysis / sequence-building later
        cleaning_report : dict
    """
    raw_df = data_loader.load_city(city, years=years)
    cleaned_df, cleaning_report = cleaning.clean_sheet(
        raw_df, target_column=features.TARGET_COLUMN, missing_strategy="drop", verbose=False
    )
    featured_df = feature_engineering.add_feature_groups(cleaned_df, ["clear_sky_index", "time_cyclical"])

    train_df, test_df, split_info = splitting.chronological_split(featured_df, train_frac=0.8)

    feature_columns = features.get_feature_columns()
    categorical_columns = [c for c in feature_columns if c in features.CATEGORICAL_COLUMNS]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    train_selected = train_df[feature_columns + [features.TARGET_COLUMN]]
    test_selected = test_df[feature_columns + [features.TARGET_COLUMN]]

    preprocessor = preprocessing.fit_preprocessor(train_selected, numeric_columns, categorical_columns)
    train_processed = preprocessing.apply_preprocessor(train_selected, preprocessor)
    test_processed = preprocessing.apply_preprocessor(test_selected, preprocessor)

    X_train, y_train, encoded_feature_columns = preprocessing.prepare_xy(train_processed, target_column=features.TARGET_COLUMN)
    X_test, y_test, _ = preprocessing.prepare_xy(test_processed, target_column=features.TARGET_COLUMN)

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


def run_classical_experiment(model, X_train, y_train, X_test, y_test) -> tuple:
    """
    Fit a classical (scikit-learn) regressor and evaluate it.

    Returns
    -------
    (metrics, y_pred, fitted_model) : tuple
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluation.regression_metrics(y_test, y_pred)
    return metrics, y_pred, model


# ---------------------------------------------------------------------------
# Running one MLP experiment
# ---------------------------------------------------------------------------


def run_mlp_experiment(X_train, y_train, X_test, y_test, seed: int, hidden_size: int = 32,
                        dropout: float = 0.0, lr: float = 1e-3, max_epochs: int = 100,
                        patience: int = 10, verbose: bool = False) -> tuple:
    """
    Train and evaluate SimpleMLPRegressor for one (dataset, seed).

    Uses a chronological inner validation split (last 20% of
    training data) for early stopping, keeping the real test set
    completely untouched during training.

    Returns
    -------
    (metrics, y_pred, model, history) : tuple
    """
    utils.set_seed(seed)
    device = utils.get_device()

    X_train_full = X_train.values.astype(np.float32).copy()
    y_train_full = y_train.values.astype(np.float32).copy()

    n_inner_train = int(len(X_train_full) * 0.8)
    X_inner_train, X_inner_val = X_train_full[:n_inner_train].copy(), X_train_full[n_inner_train:].copy()
    y_inner_train, y_inner_val = y_train_full[:n_inner_train].copy(), y_train_full[n_inner_train:].copy()

    train_loader = torch_utils.make_dataloader(X_inner_train, y_inner_train, batch_size=64, shuffle=True)
    val_loader = torch_utils.make_dataloader(X_inner_val, y_inner_val, batch_size=64, shuffle=False)

    model = p2_models.SimpleMLPRegressor(input_dim=X_train_full.shape[1], hidden_size=hidden_size, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = torch_utils.train_torch_model(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        max_epochs=max_epochs, patience=patience, verbose=verbose,
    )

    y_pred = p2_models.predict_values(model, X_test.values.astype(np.float32), device)
    metrics = evaluation.regression_metrics(y_test, y_pred)
    return metrics, y_pred, model, history


# ---------------------------------------------------------------------------
# Recording results
# ---------------------------------------------------------------------------


def record_result(experiment, city, model_name, model_type, seed, metrics, source_city="",
                   target_city="", split_type="chronological_80_20", feature_set="primary",
                   target_scaling="none", notes=""):
    """
    Append one result row to results/problem2/problem2_results.csv
    (Section 34's required schema) AND log the headline metric (rmse)
    into the shared cross-problem results history. Writes to disk
    immediately, matching Problem 1's precedent
    (course_context/PROBLEM1_REPORT.md) - this project's stages are
    run as several separate invocations to stay within a single
    command's runtime limit, so in-memory-only accumulation would lose
    data between calls.
    """
    row = {
        "experiment": experiment, "city": city, "source_city": source_city, "target_city": target_city,
        "model": model_name, "model_type": model_type, "seed": seed, "split_type": split_type,
        "feature_set": feature_set, "target_scaling": target_scaling,
        "rmse": metrics["rmse"], "mae": metrics["mae"], "nrmse": metrics["nrmse"], "r2": metrics["r2"],
        "notes": notes,
    }

    utils.ensure_dir(RESULTS_DIR)
    file_exists = os.path.exists(PROBLEM2_RESULTS_PATH)
    with open(PROBLEM2_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROBLEM2_RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    experiment_id = experiment_runner.generate_experiment_id("problem2", model_name, city or f"{source_city}_to_{target_city}", seed)
    experiment_runner.save_result({
        "experiment_id": experiment_id,
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "problem": "problem2",
        "model": model_name,
        "dataset": experiment,
        "source_city": source_city or city,
        "target_city": target_city,
        "seed": seed,
        "parameters": "{}",
        "metric": "rmse",
        "score": metrics["rmse"],
        "runtime_seconds": "",
        "notes": notes,
    })


# ---------------------------------------------------------------------------
# Stage 1: same-city baseline model sweep
# ---------------------------------------------------------------------------


def run_same_city_sweep(city: str):
    """
    Section 18: mean baseline -> linear regression -> ridge -> decision
    tree -> random forest -> gradient boosting -> MLP, at 3 seeds, on
    the primary feature set, for one city.
    """
    print(f"\n--- Stage 1 (same-city sweep): {city} ---")
    data = build_city_dataset(city)

    for seed in SEEDS:
        classical_models = p2_models.get_classical_models(seed)
        for model_name, model in classical_models.items():
            metrics, y_pred, fitted = run_classical_experiment(
                model, data["X_train"], data["y_train"], data["X_test"], data["y_test"]
            )
            record_result("same_city", city, model_name, "classical", seed, metrics, source_city=city)
            print(f"  seed={seed} {model_name:20s} rmse={metrics['rmse']:.2f} nrmse={metrics['nrmse']:.4f} r2={metrics['r2']:.3f}")

        metrics, y_pred, model, history = run_mlp_experiment(
            data["X_train"], data["y_train"], data["X_test"], data["y_test"], seed, max_epochs=80, patience=10
        )
        record_result("same_city", city, "mlp", "neural_network", seed, metrics, source_city=city)
        print(f"  seed={seed} {'mlp':20s} rmse={metrics['rmse']:.2f} nrmse={metrics['nrmse']:.4f} r2={metrics['r2']:.3f}")


# ---------------------------------------------------------------------------
# Stage 2: small hyperparameter search (chronological inner validation)
# ---------------------------------------------------------------------------

RIDGE_SEARCH_SPACE = [0.1, 1.0, 10.0]
RF_SEARCH_SPACE = [
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 3},
    {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 1},
]
GB_SEARCH_SPACE = [
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
]
MLP_SEARCH_SPACE = [
    {"hidden_size": 32, "dropout": 0.0, "lr": 1e-3},
    {"hidden_size": 64, "dropout": 0.1, "lr": 1e-3},
    {"hidden_size": 32, "dropout": 0.0, "lr": 5e-4},
]


def chronological_inner_split(X_train, y_train, val_frac=0.2):
    """Same protocol as Problem 1's - split already-chronological training data into an earlier inner-train and later inner-validation portion."""
    n_inner_train = int(len(X_train) * (1 - val_frac))
    return (
        X_train.iloc[:n_inner_train], y_train.iloc[:n_inner_train],
        X_train.iloc[n_inner_train:], y_train.iloc[n_inner_train:],
    )


def search_classical_hyperparameters(data: dict, search_seed: int = 42) -> dict:
    """
    Small grid search for ridge/random_forest/gradient_boosting on the
    chronological inner-validation split (never the real test set).

    Returns
    -------
    dict: {"ridge": alpha, "random_forest": {...}, "gradient_boosting": {...}}
    """
    X_inner_train, y_inner_train, X_inner_val, y_inner_val = chronological_inner_split(data["X_train"], data["y_train"])

    best = {}

    best_score, best_alpha = float("inf"), None
    for alpha in RIDGE_SEARCH_SPACE:
        model = p2_models.get_classical_models(search_seed, ridge_alpha=alpha)["ridge"]
        model.fit(X_inner_train, y_inner_train)
        score = evaluation.regression_metrics(y_inner_val, model.predict(X_inner_val))["rmse"]
        if score < best_score:
            best_score, best_alpha = score, alpha
    best["ridge"] = best_alpha
    print(f"    ridge: best alpha={best_alpha} (inner-val rmse={best_score:.2f})")

    for model_name, search_space, param_key in [
        ("random_forest", RF_SEARCH_SPACE, "rf_params"),
        ("gradient_boosting", GB_SEARCH_SPACE, "gb_params"),
    ]:
        best_score, best_params = float("inf"), None
        for params in search_space:
            model = p2_models.get_classical_models(search_seed, **{param_key: params})[model_name]
            model.fit(X_inner_train, y_inner_train)
            score = evaluation.regression_metrics(y_inner_val, model.predict(X_inner_val))["rmse"]
            if score < best_score:
                best_score, best_params = score, params
        best[model_name] = best_params
        print(f"    {model_name}: best={best_params} (inner-val rmse={best_score:.2f})")

    return best


def search_mlp_hyperparameters(data: dict, search_seed: int = 42) -> dict:
    """Small grid search for the MLP, same inner-validation protocol."""
    X_train_full = data["X_train"].values.astype(np.float32).copy()
    y_train_full = data["y_train"].values.astype(np.float32).copy()
    n_inner_train = int(len(X_train_full) * 0.8)
    X_inner_train, X_inner_val = X_train_full[:n_inner_train].copy(), X_train_full[n_inner_train:].copy()
    y_inner_train, y_inner_val = y_train_full[:n_inner_train].copy(), y_train_full[n_inner_train:].copy()

    best_score, best_params = float("inf"), None
    for params in MLP_SEARCH_SPACE:
        utils.set_seed(search_seed)
        device = utils.get_device()
        train_loader = torch_utils.make_dataloader(X_inner_train, y_inner_train, batch_size=64, shuffle=True)
        val_loader = torch_utils.make_dataloader(X_inner_val, y_inner_val, batch_size=64, shuffle=False)

        model = p2_models.SimpleMLPRegressor(input_dim=X_train_full.shape[1], hidden_size=params["hidden_size"], dropout=params["dropout"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        torch_utils.train_torch_model(model, train_loader, val_loader, optimizer, nn.MSELoss(), device, max_epochs=40, patience=6, verbose=False)

        y_val_pred = p2_models.predict_values(model, X_inner_val, device)
        score = evaluation.regression_metrics(y_inner_val, y_val_pred)["rmse"]
        if score < best_score:
            best_score, best_params = score, params

    print(f"    mlp: best={best_params} (inner-val rmse={best_score:.2f})")
    return best_params


def run_hyperparameter_search(city: str) -> dict:
    """Section 32: small hyperparameter search for one city, all 4 tunable model types."""
    print(f"\n--- Stage 2 (hyperparameter search): {city} ---")
    data = build_city_dataset(city)
    classical_best = search_classical_hyperparameters(data)
    mlp_best = search_mlp_hyperparameters(data)
    return {**classical_best, "mlp": mlp_best}


# ---------------------------------------------------------------------------
# Stage 3: final tuned multi-seed evaluation
# ---------------------------------------------------------------------------


def run_tuned_final_evaluation(city: str, tuned: dict):
    """
    Re-run ridge/random_forest/gradient_boosting/mlp with the tuned
    hyperparameters found in Stage 2, at all 3 seeds, evaluated on the
    REAL test set. Linear regression and decision tree keep their
    Stage 1 default-hyperparameter results (not part of the tuning
    scope, matching Problem 1's precedent).
    """
    print(f"\n--- Stage 3 (tuned final eval): {city} ---")
    data = build_city_dataset(city)

    for seed in SEEDS:
        ridge = p2_models.get_classical_models(seed, ridge_alpha=tuned["ridge"])["ridge"]
        metrics, _, _ = run_classical_experiment(ridge, data["X_train"], data["y_train"], data["X_test"], data["y_test"])
        record_result("same_city", city, "ridge_tuned", "classical", seed, metrics, source_city=city, notes=f"alpha={tuned['ridge']}")
        print(f"  seed={seed} ridge_tuned  rmse={metrics['rmse']:.2f}")

        rf = p2_models.get_classical_models(seed, rf_params=tuned["random_forest"])["random_forest"]
        metrics, _, _ = run_classical_experiment(rf, data["X_train"], data["y_train"], data["X_test"], data["y_test"])
        record_result("same_city", city, "random_forest_tuned", "classical", seed, metrics, source_city=city, notes=json.dumps(tuned["random_forest"]))
        print(f"  seed={seed} random_forest_tuned  rmse={metrics['rmse']:.2f}")

        gb = p2_models.get_classical_models(seed, gb_params=tuned["gradient_boosting"])["gradient_boosting"]
        metrics, _, _ = run_classical_experiment(gb, data["X_train"], data["y_train"], data["X_test"], data["y_test"])
        record_result("same_city", city, "gradient_boosting_tuned", "classical", seed, metrics, source_city=city, notes=json.dumps(tuned["gradient_boosting"]))
        print(f"  seed={seed} gradient_boosting_tuned  rmse={metrics['rmse']:.2f}")

        metrics, _, _, _ = run_mlp_experiment(
            data["X_train"], data["y_train"], data["X_test"], data["y_test"], seed,
            max_epochs=100, patience=12, **tuned["mlp"]
        )
        record_result("same_city", city, "mlp_tuned", "neural_network", seed, metrics, source_city=city, notes=json.dumps(tuned["mlp"]))
        print(f"  seed={seed} mlp_tuned  rmse={metrics['rmse']:.2f}")


# ---------------------------------------------------------------------------
# Stage 4: cross-city zero-shot
# ---------------------------------------------------------------------------


def build_zero_shot_target_dataset(city: str, source_preprocessor: dict, feature_columns: list) -> dict:
    """
    Prepare a TARGET city's data for zero-shot evaluation: load, clean,
    feature-engineer, and chronologically split exactly like any other
    city - but apply the SOURCE city's already-fitted preprocessor
    (not a newly-fit one) to the target's test split. This is the
    crux of a genuine zero-shot evaluation: the target-city test rows
    are scaled/encoded using EXACTLY the transformation the source
    model was trained under, with zero information from the target
    city used to fit anything.

    The target city's training split (`train_df`) is returned too, but
    is NEVER used to fit or train anything - see run_cross_city_zero_shot()'s
    oracle baseline for the one place it's used (explicitly labeled as
    not a legitimate zero-shot method).

    Parameters
    ----------
    city : str
        Target city, e.g. "Huron".
    source_preprocessor : dict
        From the SOURCE city's build_city_dataset() call.
    feature_columns : list of str
        The RAW (pre-encoding) primary feature column names -
        features.get_feature_columns() - NOT the post-one-hot-encoding
        column list. The raw "Cloud Type" column has to exist in the
        selected DataFrame for apply_preprocessor() to encode it; the
        already-encoded names (e.g. "Cloud Type_0.0") only exist AFTER
        that step runs.

    Returns
    -------
    dict with X_test, y_test, train_df, test_df, split_info
    """
    raw_df = data_loader.load_city(city, years="long")
    cleaned_df, cleaning_report = cleaning.clean_sheet(
        raw_df, target_column=features.TARGET_COLUMN, missing_strategy="drop", verbose=False
    )
    featured_df = feature_engineering.add_feature_groups(cleaned_df, ["clear_sky_index", "time_cyclical"])
    train_df, test_df, split_info = splitting.chronological_split(featured_df, train_frac=0.8)

    test_selected = test_df[feature_columns + [features.TARGET_COLUMN]]
    test_processed = preprocessing.apply_preprocessor(test_selected, source_preprocessor)
    X_test, y_test, _ = preprocessing.prepare_xy(test_processed, target_column=features.TARGET_COLUMN)

    return {"X_test": X_test, "y_test": y_test, "train_df": train_df, "test_df": test_df, "split_info": split_info}


def run_cross_city_zero_shot():
    """
    Sections 6, 7, 22, 23: train the strongest Davis model
    (gradient_boosting, empirically the best same-city Davis model -
    see results/problem2/problem2_results.csv), evaluate it with ZERO
    target-city training on Huron / Santa Barbara / La Jolla.

    TARGET SCALING DESIGN (Section 7 - read this before changing
    anything here): the model predicts RAW kW, trained on Davis's raw
    kW target, with no target scaling at all. This is the ONLY
    approach that's genuinely zero-shot: any scheme that rescales
    predictions using a target city's own statistics (mean, max, etc.)
    would require information a real zero-shot deployment wouldn't
    have. nRMSE is still computed per target city using THAT city's
    own test-period range - this is fine because nRMSE here is a
    POST-HOC EVALUATION metric (interpreting how bad the raw-kW error
    is relative to that city's own scale), not something used to
    generate the prediction itself or fit anything. A separate,
    explicitly-labeled "oracle" baseline (predict the target city's own
    historical mean) is also computed for comparison, per Section 23 -
    it is NOT a legitimate zero-shot method (it needs target-city data
    a real zero-shot system wouldn't have) and is labeled as such in
    its `notes` field.
    """
    print("\n--- Stage 4 (cross-city zero-shot) ---")
    source_data = build_city_dataset(SOURCE_CITY)

    for seed in SEEDS:
        model = p2_models.get_classical_models(seed)["gradient_boosting"]
        model.fit(source_data["X_train"], source_data["y_train"])

        for target_city in CROSS_CITY_TARGETS:
            target_data = build_zero_shot_target_dataset(target_city, source_data["preprocessor"], features.get_feature_columns())
            y_pred = model.predict(target_data["X_test"])
            metrics = evaluation.regression_metrics(target_data["y_test"], y_pred)
            record_result(
                "cross_city_zero_shot", "", "gradient_boosting", "classical", seed, metrics,
                source_city=SOURCE_CITY, target_city=target_city, target_scaling="none_raw_kW",
                notes="genuine zero-shot: zero target-city rows used in training or preprocessing fitting",
            )
            print(f"  seed={seed} {SOURCE_CITY}->{target_city}  rmse={metrics['rmse']:.2f} nrmse={metrics['nrmse']:.4f} r2={metrics['r2']:.3f}")

            # Oracle baseline (Section 23) - NOT a legitimate zero-shot
            # method, uses the target city's own historical mean.
            target_mean = target_data["train_df"][features.TARGET_COLUMN].mean()
            oracle_pred = np.full(len(target_data["y_test"]), target_mean)
            oracle_metrics = evaluation.regression_metrics(target_data["y_test"], oracle_pred)
            record_result(
                "cross_city_zero_shot", "", "oracle_target_mean_baseline", "baseline", seed, oracle_metrics,
                source_city=SOURCE_CITY, target_city=target_city, target_scaling="target_city_own_mean",
                notes="NOT a legitimate zero-shot method - uses target city's own historical mean, shown only for comparison per Section 23",
            )
            print(f"  seed={seed} oracle_mean_baseline({target_city})  rmse={oracle_metrics['rmse']:.2f}")

            # DIAGNOSTIC ONLY (not a legitimate zero-shot method either):
            # rescale Davis's raw predictions by the ratio of the target
            # city's own historical mean to Davis's training mean, to see
            # whether the model's SHAPE of prediction is reasonable and
            # it's simply off in absolute scale, or whether it's also
            # capturing the target city's day-to-day pattern poorly.
            # Requires target-city information a real zero-shot system
            # wouldn't have - purely an analysis tool, never used to
            # "improve" the reported zero-shot number above.
            scale_ratio = target_mean / source_data["y_train"].mean()
            scale_corrected_pred = y_pred * scale_ratio
            scale_corrected_metrics = evaluation.regression_metrics(target_data["y_test"], scale_corrected_pred)
            record_result(
                "cross_city_zero_shot", "", "gradient_boosting_scale_corrected_DIAGNOSTIC", "diagnostic", seed,
                scale_corrected_metrics, source_city=SOURCE_CITY, target_city=target_city,
                target_scaling="diagnostic_mean_ratio_rescale",
                notes="DIAGNOSTIC ONLY, NOT a legitimate zero-shot method - rescales predictions using the target city's own mean, to isolate whether the failure is scale mismatch vs. poor pattern-matching",
            )
            print(f"  seed={seed} scale_corrected_DIAGNOSTIC({target_city})  rmse={scale_corrected_metrics['rmse']:.2f} r2={scale_corrected_metrics['r2']:.3f}")


# ---------------------------------------------------------------------------
# Stage 5: 3-year vs 6-year ablation
# ---------------------------------------------------------------------------


def run_3yr_vs_6yr_ablation():
    """
    Section 24: compare gradient_boosting (Davis's established best
    model - see results/problem2/problem2_results.csv) trained on the
    3-year Davis sheet vs. the 6-year sheet, same chronological 80/20
    protocol and evaluation applied to each independently (each
    sheet's own 20% test period, not a shared one - the two sheets
    cover different date ranges, so there's no single test period that
    would be fair to both).
    """
    print("\n--- Stage 5 (3yr vs 6yr ablation): Davis ---")
    for years in ["short", "long"]:
        data = build_city_dataset("Davis", years=years)
        label = "3yr" if years == "short" else "6yr"
        for seed in SEEDS:
            model = p2_models.get_classical_models(seed)["gradient_boosting"]
            metrics, _, _ = run_classical_experiment(model, data["X_train"], data["y_train"], data["X_test"], data["y_test"])
            record_result(
                "3yr_vs_6yr", "Davis", f"gradient_boosting_{label}", "classical", seed, metrics,
                source_city="Davis", notes=f"trained on Davis {label} sheet, split_info={data['split_info']['n_train']}/{data['split_info']['n_test']}",
            )
            print(f"  seed={seed} {label}  rmse={metrics['rmse']:.2f} nrmse={metrics['nrmse']:.4f} r2={metrics['r2']:.3f} (n_train={data['split_info']['n_train']})")


# ---------------------------------------------------------------------------
# Stage 6: sequence forecasting (K=12, GRU)
# ---------------------------------------------------------------------------


def build_sequence_dataset(city: str, k: int = sequence.K_STEPS) -> dict:
    """
    Prepare one city's data for the sequence-forecasting sub-task:
    load, clean, feature-engineer, chronologically split, preprocess
    (scaling Output Power itself alongside the other numeric features,
    since it's used as a lag feature inside each window), then build
    train and test windows INDEPENDENTLY (never spanning the
    train/test boundary - see sequence.py's module docstring).

    Returns
    -------
    dict with X_train, y_train, X_test, y_test (y in raw kW),
    feature_columns, preprocessor, split_info, train_df, test_df.
    """
    raw_df = data_loader.load_city(city, years="long")
    cleaned_df, cleaning_report = cleaning.clean_sheet(
        raw_df, target_column=features.TARGET_COLUMN, missing_strategy="drop", verbose=False
    )
    featured_df = feature_engineering.add_feature_groups(cleaned_df, ["clear_sky_index", "time_cyclical"])
    train_df, test_df, split_info = splitting.chronological_split(featured_df, train_frac=0.8)

    # Output Power is included as a feature here (a legitimate lag
    # feature for this sub-task only - course_context/LEAKAGE_MAP.md,
    # Problem 2) so it gets scaled consistently with every other input.
    sequence_feature_columns = features.get_feature_columns() + [features.TARGET_COLUMN]
    categorical_columns = [c for c in sequence_feature_columns if c in features.CATEGORICAL_COLUMNS]
    numeric_columns = [c for c in sequence_feature_columns if c not in categorical_columns]

    train_selected = train_df[sequence_feature_columns]
    test_selected = test_df[sequence_feature_columns]

    preprocessor = preprocessing.fit_preprocessor(train_selected, numeric_columns, categorical_columns)
    train_processed = preprocessing.apply_preprocessor(train_selected, preprocessor)
    test_processed = preprocessing.apply_preprocessor(test_selected, preprocessor)

    encoded_feature_columns = list(train_processed.columns)

    # feature_df = SCALED data (for window features); target_series =
    # RAW kW (for an interpretable prediction target) - see
    # sequence.build_sequences()'s docstring for why these must differ.
    X_train, y_train = sequence.build_sequences(train_processed, train_df[features.TARGET_COLUMN], encoded_feature_columns, k=k)
    X_test, y_test = sequence.build_sequences(test_processed, test_df[features.TARGET_COLUMN], encoded_feature_columns, k=k)

    return {
        "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        "feature_columns": encoded_feature_columns, "preprocessor": preprocessor,
        "split_info": split_info, "train_df": train_df, "test_df": test_df,
    }


def run_gru_experiment(data: dict, seed: int, hidden_size: int = 32, num_layers: int = 1,
                        dropout: float = 0.0, lr: float = 1e-3, max_epochs: int = 60,
                        patience: int = 8, verbose: bool = False) -> tuple:
    """
    Train and evaluate SimpleGRURegressor for one seed. Early stopping
    uses a chronological inner-validation split of the TRAINING
    windows only (last 20%) - the real test windows are never touched
    during training.

    TARGET SCALING NOTE: the target is standardized (fit on the inner-
    training portion only) purely for neural-network training
    stability/speed - raw Output Power values up to ~260 kW made the
    GRU converge far too slowly in initial testing (still RMSE~100
    after 10 epochs) - a completely different concern from Section 7's
    CROSS-CITY target-scaling discussion, which is about whether
    scaling leaks target-city information in a zero-shot setting. Here
    there's only one city and no leakage risk: the scaler is fit on
    this city's own training data, same as any other preprocessing
    step. Predictions are inverse-transformed back to raw kW before
    computing metrics, so RMSE/MAE/nRMSE stay in interpretable units.

    Returns
    -------
    (metrics, y_pred, model, history) : tuple
    """
    utils.set_seed(seed)
    device = utils.get_device()

    n_inner_train = int(len(data["X_train"]) * 0.8)
    X_inner_train, X_inner_val = data["X_train"][:n_inner_train], data["X_train"][n_inner_train:]
    y_inner_train_raw, y_inner_val_raw = data["y_train"][:n_inner_train], data["y_train"][n_inner_train:]

    target_scaler = preprocessing.fit_target_scaler(pd.Series(y_inner_train_raw))
    y_inner_train = preprocessing.apply_target_scaler(pd.Series(y_inner_train_raw), target_scaler)
    y_inner_val = preprocessing.apply_target_scaler(pd.Series(y_inner_val_raw), target_scaler)

    train_loader = torch_utils.make_dataloader(X_inner_train, y_inner_train, batch_size=64, shuffle=True)
    val_loader = torch_utils.make_dataloader(X_inner_val, y_inner_val, batch_size=64, shuffle=False)

    n_features = data["X_train"].shape[2]
    model = sequence.SimpleGRURegressor(n_features=n_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = torch_utils.train_torch_model(
        model, train_loader, val_loader, optimizer, nn.MSELoss(), device,
        max_epochs=max_epochs, patience=patience, verbose=verbose,
    )

    y_pred_scaled = sequence.predict_sequence_values(model, data["X_test"], device)
    y_pred = preprocessing.inverse_transform_target(y_pred_scaled, target_scaler)
    metrics = evaluation.regression_metrics(data["y_test"], y_pred)
    return metrics, y_pred, model, history


def run_sequence_experiments(city: str = SOURCE_CITY):
    """
    Sections 25-30: build K=12 windows, train a GRU at 3 seeds, and
    compare against the persistence baseline and the best non-sequence
    model for the same city (from results/problem2/problem2_results.csv).
    """
    print(f"\n--- Stage 6 (sequence forecasting, K={sequence.K_STEPS}): {city} ---")
    data = build_sequence_dataset(city)
    print(f"  Train windows: {data['X_train'].shape}, Test windows: {data['X_test'].shape}")

    # Persistence baseline (Section 29) - deterministic, but logged
    # once per seed for schema consistency with the rest of the project.
    persistence_pred = sequence.persistence_baseline_predictions(data["test_df"][features.TARGET_COLUMN])
    persistence_metrics = evaluation.regression_metrics(data["y_test"], persistence_pred)
    for seed in SEEDS:
        record_result(
            "sequence", city, "persistence_baseline", "baseline", seed, persistence_metrics,
            source_city=city, feature_set=f"k={sequence.K_STEPS}_window",
            notes="predicts next Output Power = most recent observed value",
        )
    print(f"  persistence_baseline  rmse={persistence_metrics['rmse']:.2f} nrmse={persistence_metrics['nrmse']:.4f} r2={persistence_metrics['r2']:.3f}")

    for seed in SEEDS:
        metrics, y_pred, model, history = run_gru_experiment(data, seed, max_epochs=60, patience=8)
        record_result(
            "sequence", city, "gru", "neural_network", seed, metrics,
            source_city=city, feature_set=f"k={sequence.K_STEPS}_window",
            notes=f"epochs_run={len(history['epoch'])}",
        )
        print(f"  seed={seed} gru  rmse={metrics['rmse']:.2f} nrmse={metrics['nrmse']:.4f} r2={metrics['r2']:.3f} (epochs={len(history['epoch'])})")


GRU_SEARCH_SPACE = [
    {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "lr": 1e-3},
    {"hidden_size": 64, "num_layers": 1, "dropout": 0.1, "lr": 1e-3},
    {"hidden_size": 32, "num_layers": 2, "dropout": 0.1, "lr": 1e-3},
]


def search_gru_hyperparameters(data: dict, search_seed: int = 42) -> dict:
    """
    Section 32: small hyperparameter search for the GRU (hidden size,
    number of layers, dropout), scored on a chronological inner-
    validation split of the TRAINING windows only - the real test
    windows are never touched during this search, consistent with
    every other model's search in this file.
    """
    print("  Searching GRU hyperparameters...")

    n_inner_train = int(len(data["X_train"]) * 0.8)
    X_inner_train, X_inner_val = data["X_train"][:n_inner_train], data["X_train"][n_inner_train:]
    y_inner_train_raw, y_inner_val_raw = data["y_train"][:n_inner_train], data["y_train"][n_inner_train:]

    # A second-level inner split, so early stopping (which needs its
    # own validation set) doesn't use the same rows the search is
    # scoring candidates on.
    n_train_train = int(len(X_inner_train) * 0.8)
    X_tt, X_tv = X_inner_train[:n_train_train], X_inner_train[n_train_train:]
    y_tt_raw, y_tv_raw = y_inner_train_raw[:n_train_train], y_inner_train_raw[n_train_train:]

    device = utils.get_device()
    best_score, best_params = float("inf"), None
    for params in GRU_SEARCH_SPACE:
        utils.set_seed(search_seed)
        target_scaler = preprocessing.fit_target_scaler(pd.Series(y_tt_raw))
        y_tt = preprocessing.apply_target_scaler(pd.Series(y_tt_raw), target_scaler)
        y_tv = preprocessing.apply_target_scaler(pd.Series(y_tv_raw), target_scaler)

        train_loader = torch_utils.make_dataloader(X_tt, y_tt, batch_size=64, shuffle=True)
        val_loader = torch_utils.make_dataloader(X_tv, y_tv, batch_size=64, shuffle=False)

        model = sequence.SimpleGRURegressor(
            n_features=data["X_train"].shape[2], hidden_size=params["hidden_size"],
            num_layers=params["num_layers"], dropout=params["dropout"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        torch_utils.train_torch_model(model, train_loader, val_loader, optimizer, nn.MSELoss(), device, max_epochs=40, patience=6, verbose=False)

        y_inner_val_pred_scaled = sequence.predict_sequence_values(model, X_inner_val, device)
        y_inner_val_pred = preprocessing.inverse_transform_target(y_inner_val_pred_scaled, target_scaler)
        score = evaluation.regression_metrics(y_inner_val_raw, y_inner_val_pred)["rmse"]
        if score < best_score:
            best_score, best_params = score, params

    print(f"    gru: best={best_params} (inner-val rmse={best_score:.2f})")
    return best_params


# ---------------------------------------------------------------------------
# Stage 7: learning curve
# ---------------------------------------------------------------------------

LEARNING_CURVE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def run_learning_curve(city: str = SOURCE_CITY, seed: int = 42):
    """
    Section 19: for the strongest same-city model (gradient_boosting),
    show test RMSE as an increasing fraction of the (chronologically
    earliest) training data is used - does more historical data keep
    helping, or does it plateau? A single seed is used (kept small per
    the instruction not to spend excessive compute on this).

    IMPORTANT: fractions are taken from the START of the training
    period (earliest data), not a random subsample - training on
    "the first 20% of training data" and "the first 100%" both still
    respect chronological order relative to the untouched test set.
    """
    print(f"\n--- Stage 7 (learning curve): {city} ---")
    data = build_city_dataset(city)
    n_train = len(data["X_train"])

    rows = []
    for frac in LEARNING_CURVE_FRACTIONS:
        n_use = int(n_train * frac)
        X_subset, y_subset = data["X_train"].iloc[:n_use], data["y_train"].iloc[:n_use]
        model = p2_models.get_classical_models(seed)["gradient_boosting"]
        metrics, _, _ = run_classical_experiment(model, X_subset, y_subset, data["X_test"], data["y_test"])
        rows.append({"train_fraction": frac, "n_train_rows": n_use, "rmse": metrics["rmse"], "r2": metrics["r2"]})
        print(f"  train_fraction={frac:.1f} (n={n_use})  rmse={metrics['rmse']:.2f} r2={metrics['r2']:.3f}")

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(os.path.join(RESULTS_DIR, f"learning_curve_{city}.csv"), index=False)
    return curve_df


# ---------------------------------------------------------------------------
# Stage 8: error analysis and prediction-vs-truth plots (best same-city model)
# ---------------------------------------------------------------------------


def analyze_best_same_city_model(city: str = SOURCE_CITY, seed: int = 42):
    """
    Sections 20-21: for the strongest same-city model
    (gradient_boosting - see results/problem2/problem2_results.csv),
    create predicted-vs-actual scatter and time-series plots, and
    analyze WHERE the errors are largest (low/high irradiance, rapidly
    changing irradiance, cloudy periods).
    """
    print(f"\n--- Stage 8 (error analysis + plots): {city} ---")
    data = build_city_dataset(city)
    model = p2_models.get_classical_models(seed)["gradient_boosting"]
    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])
    y_test = data["y_test"].values

    # 1. Predicted vs actual scatter
    scatter_path = os.path.join(FIGURES_DIR, f"problem2_predicted_vs_actual_{city.lower().replace(' ', '_')}.png")
    visualization.plot_prediction_vs_truth(y_test, y_pred, save_path=scatter_path, title=f"{city}: Predicted vs Actual Output Power (kW)")
    print(f"  Saved {scatter_path}")

    # 2. Time-series plot over a representative test period (first 5 days of test)
    test_df = data["test_df"].reset_index(drop=True)
    n_points = min(5 * 11, len(test_df))  # ~5 days at 11 samples/day
    dates = pd.to_datetime(dict(year=test_df["Year"][:n_points], month=test_df["Month"][:n_points], day=test_df["Day"][:n_points])) + \
            pd.to_timedelta(test_df["Hour"][:n_points], unit="h") + pd.to_timedelta(test_df["Minute"][:n_points], unit="m")

    import matplotlib.pyplot as plt
    visualization.set_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, y_test[:n_points], label="Actual", marker="o", markersize=3)
    ax.plot(dates, y_pred[:n_points], label="Predicted", marker="x", markersize=3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Output Power (kW)")
    ax.set_title(f"{city}: Actual vs Predicted Output Power (representative test period)")
    ax.legend()
    fig.autofmt_xdate()
    timeseries_path = os.path.join(FIGURES_DIR, f"problem2_prediction_timeseries_{city.lower().replace(' ', '_')}.png")
    fig.savefig(timeseries_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {timeseries_path}")

    # 3. Error analysis: where do the biggest errors happen?
    error = y_test - y_pred
    abs_error = np.abs(error)
    ghi = test_df["GHI"].values
    clear_sky_index = test_df["Clear_Sky_Index"].values if "Clear_Sky_Index" in test_df.columns else None

    ghi_terciles = np.quantile(ghi, [1 / 3, 2 / 3])
    low_ghi_mask = ghi <= ghi_terciles[0]
    high_ghi_mask = ghi > ghi_terciles[1]

    # "Rapidly changing irradiance": large |GHI[t] - GHI[t-1]| within the test period.
    ghi_diff = np.abs(np.diff(ghi, prepend=ghi[0]))
    rapid_change_mask = ghi_diff > np.quantile(ghi_diff, 0.9)

    error_summary = {
        "overall_mean_abs_error": float(abs_error.mean()),
        "low_ghi_mean_abs_error": float(abs_error[low_ghi_mask].mean()),
        "high_ghi_mean_abs_error": float(abs_error[high_ghi_mask].mean()),
        "rapidly_changing_ghi_mean_abs_error": float(abs_error[rapid_change_mask].mean()),
        "cloudy_mean_abs_error": None,
        "clear_mean_abs_error": None,
    }
    if clear_sky_index is not None:
        cloudy_mask = clear_sky_index < 0.4
        clear_mask = clear_sky_index >= 0.85
        error_summary["cloudy_mean_abs_error"] = float(abs_error[cloudy_mask].mean()) if cloudy_mask.any() else None
        error_summary["clear_mean_abs_error"] = float(abs_error[clear_mask].mean()) if clear_mask.any() else None

    print(f"  Error analysis: {error_summary}")
    with open(os.path.join(RESULTS_DIR, f"error_analysis_{city}.json"), "w") as f:
        json.dump(error_summary, f, indent=2)

    return {"y_test": y_test, "y_pred": y_pred, "error_summary": error_summary}


# ---------------------------------------------------------------------------
# Summary figures
# ---------------------------------------------------------------------------


def make_model_comparison_figure():
    """problem2_model_comparison.png - mean RMSE per model, per same-city (Davis, Amherst)."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM2_RESULTS_PATH)
    same_city = df[df["experiment"] == "same_city"]
    summary = same_city.groupby(["city", "model"])["rmse"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, city in zip(axes, SAME_CITY_CITIES):
        subset = summary[summary["city"] == city].sort_values("rmse", ascending=False)
        ax.barh(subset["model"], subset["rmse"])
        ax.set_title(f"{city} — mean RMSE by model")
        ax.set_xlabel("RMSE (kW)")
    fig.suptitle("Problem 2 same-city model comparison (mean across 3 seeds)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem2_model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_cross_city_comparison_figure():
    """problem2_cross_city_comparison.png - RMSE for zero-shot, oracle mean baseline, and scale-corrected diagnostic, per target city."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM2_RESULTS_PATH)
    cross = df[df["experiment"] == "cross_city_zero_shot"]
    summary = cross.groupby(["target_city", "model"])["rmse"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    models = ["gradient_boosting", "oracle_target_mean_baseline", "gradient_boosting_scale_corrected_DIAGNOSTIC"]
    labels = ["Zero-shot (raw kW)", "Oracle: target mean\n(not real zero-shot)", "Scale-corrected\n(diagnostic only)"]
    x = np.arange(len(CROSS_CITY_TARGETS))
    width = 0.25
    for i, (model, label) in enumerate(zip(models, labels)):
        values = [summary[(summary.target_city == c) & (summary.model == model)]["rmse"].values[0] for c in CROSS_CITY_TARGETS]
        ax.bar(x + (i - 1) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CROSS_CITY_TARGETS)
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("Davis -> target city: zero-shot RMSE (lower is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem2_cross_city_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_sequence_comparison_figure():
    """problem2_sequence_comparison.png - RMSE for persistence baseline, GRU, and the best non-sequence Davis model."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM2_RESULTS_PATH)
    seq_summary = df[df["experiment"] == "sequence"].groupby("model")["rmse"].mean()
    best_non_sequence = df[(df["experiment"] == "same_city") & (df["city"] == "Davis")].groupby("model")["rmse"].mean().min()

    labels = ["persistence_baseline", "gru", "gru_tuned", "best_non_sequence_model"]
    values = [seq_summary.get("persistence_baseline", np.nan), seq_summary.get("gru", np.nan), seq_summary.get("gru_tuned", np.nan), best_non_sequence]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("Sequence forecasting (K=12) vs. persistence vs. best non-sequence model — Davis")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem2_sequence_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_learning_curve_figure(curve_df: pd.DataFrame):
    """problem2_learning_curve.png"""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_df["train_fraction"], curve_df["rmse"], marker="o")
    ax.set_xlabel("Fraction of training data used")
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_title("Learning curve — Davis, gradient_boosting")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem2_learning_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
