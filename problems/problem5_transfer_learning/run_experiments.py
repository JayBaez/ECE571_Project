"""
run_experiments.py (Problem 5 — Transfer Learning)

Main experiment script for Problem 5, the final required ML problem:
"Can a data-rich source city (Davis) help a data-poor target city
(Amherst)?" for Output Power regression.

Three required comparisons, all evaluated on the SAME held-out Amherst
test set:
    A. ZERO-SHOT   - Davis-trained model, applied directly, no Amherst
                      training data used for fitting anything.
    B. FEW-SHOT     - a fresh model trained ONLY on k Amherst samples
                      (k=10/50/100), Davis never seen.
    C. TRANSFER     - the Davis-PRETRAINED model, fine-tuned on the
                      SAME k Amherst samples used for (B).

Plus: domain-shift analysis (why might Davis knowledge transfer, or
not?), a target-normalization ablation, and a layer-freezing ablation.

Usage (from the project root):
    python problems/problem5_transfer_learning/run_experiments.py
Note: like Problems 1-4, this script's stages were run as several
smaller invocations during development to stay within a single
command's execution-time limit - see course_context/PROBLEM5_REPORT.md.
"""

import copy
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

from src import data_loader, evaluation, experiment_runner, preprocessing, splitting, torch_utils, utils, visualization
from problems.problem2_regression.run_experiments import build_city_dataset, build_zero_shot_target_dataset
from problems.problem2_regression import features
from problems.problem5_transfer_learning import models as p5_models

SEEDS = utils.DEFAULT_SEEDS  # [42, 123, 2026]
SOURCE_CITY = "Davis"
TARGET_CITY = "Amherst"
K_VALUES = [10, 50, 100]

RESULTS_DIR = "results/problem5"
MODELS_DIR = "results/problem5/models"
FIGURES_DIR = "figures/problem5"

PROBLEM5_RESULTS_PATH = os.path.join(RESULTS_DIR, "problem5_results.csv")
PROBLEM5_RESULT_FIELDS = [
    "source_city", "target_city", "method", "target_samples", "seed",
    "target_normalization", "frozen_layers", "rmse", "mae", "nrmse",
    "transfer_gain", "transfer_gain_percent", "notes",
]


def build_source_and_target_data() -> dict:
    """
    Reuses Problem 2's EXACT dataset-building functions
    (`build_city_dataset()`, `build_zero_shot_target_dataset()`) -
    same chronological splits, same feature set, same leak-free
    zero-shot preprocessing pattern (source preprocessor applied to
    target, never a newly-fit one) already established and validated
    there (`course_context/PROBLEM2_REPORT.md`).

    Returns
    -------
    dict with:
        source : Problem 2's Davis build_city_dataset() dict (has its
            own train/test split - only "X_train"/"y_train" are used
            here, for pretraining; Davis's own test set isn't used in
            this problem at all).
        target_test : Amherst's chronological TEST split, features
            transformed with the SOURCE (Davis) preprocessor - this is
            the ONE test set every method in this file is evaluated on.
        target_train_pool : Amherst's chronological TRAINING split
            (80%), features transformed with the SOURCE preprocessor -
            few-shot samples are drawn from here, never from the test
            split.
    """
    source = build_city_dataset(SOURCE_CITY)
    target_test_data = build_zero_shot_target_dataset(TARGET_CITY, source["preprocessor"], features.get_feature_columns())

    train_selected = target_test_data["train_df"][features.get_feature_columns() + [features.TARGET_COLUMN]]
    train_processed = preprocessing.apply_preprocessor(train_selected, source["preprocessor"])
    X_target_train_pool, y_target_train_pool, _ = preprocessing.prepare_xy(train_processed, target_column=features.TARGET_COLUMN)

    return {
        "source": source,
        "target_test": {"X_test": target_test_data["X_test"], "y_test": target_test_data["y_test"]},
        "target_train_pool": {"X_train": X_target_train_pool, "y_train": y_target_train_pool},
        "target_split_info": target_test_data["split_info"],
    }


def pretrain_on_source(data: dict, seed: int, hidden1: int = 128, hidden2: int = 64,
                        dropout: float = 0.2, lr: float = 1e-3, max_epochs: int = 100,
                        patience: int = 10, verbose: bool = False) -> tuple:
    """
    Section 13: train TransferMLP on Davis (source) training data
    only, from a fresh random initialization. Early stopping uses a
    chronological inner-validation split of DAVIS's own training data
    (last 20%) - Amherst is never touched here.

    Returns
    -------
    (model, history) : tuple
    """
    utils.set_seed(seed)
    device = utils.get_device()

    X_train_full = data["X_train"].values.astype(np.float32).copy()
    y_train_full = np.asarray(data["y_train"], dtype=np.float32).copy()

    n_inner_train = int(len(X_train_full) * 0.8)
    X_inner_train, X_inner_val = X_train_full[:n_inner_train].copy(), X_train_full[n_inner_train:].copy()
    y_inner_train, y_inner_val = y_train_full[:n_inner_train].copy(), y_train_full[n_inner_train:].copy()

    train_loader = torch_utils.make_dataloader(X_inner_train, y_inner_train, batch_size=64, shuffle=True)
    val_loader = torch_utils.make_dataloader(X_inner_val, y_inner_val, batch_size=64, shuffle=False)

    model = p5_models.TransferMLP(input_dim=X_train_full.shape[1], hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = torch_utils.train_torch_model(
        model, train_loader, val_loader, optimizer, nn.MSELoss(), device,
        max_epochs=max_epochs, patience=patience, verbose=verbose,
    )
    return model, history


def fine_tune_on_target(pretrained_model: p5_models.TransferMLP, X_k: np.ndarray, y_k: np.ndarray,
                         seed: int, freeze_first_layer: bool = False, lr: float = 1e-3,
                         epochs: int = 100, verbose: bool = False) -> p5_models.TransferMLP:
    """
    Section 14, 35: continue training a COPY of the pretrained model
    (never mutates the original, so it can be reused for other k
    values/seeds) on a small Amherst sample.

    LEARNING RATE - AN IMPORTANT FINDING, NOT A DEFAULT ASSUMPTION:
    the instructions suggest a SMALLER fine-tuning learning rate than
    pretraining "where appropriate." Tested directly here: a 10x
    smaller rate (1e-4) causes SEVERE negative transfer at k=10 - the
    Davis-pretrained output layer is calibrated to Davis's ~164 kW
    scale, and 1e-4 is too small to shift that output level to
    Amherst's ~64 kW scale within 100 epochs on just 10 samples (mean
    prediction stayed at ~136 kW, RMSE=140.9). Using the SAME learning
    rate as pretraining (1e-3) lets the model actually adapt its
    output scale in time (mean prediction ~70 kW, RMSE=27.2) - and
    transfer then clearly BEATS the few-shot baseline (RMSE 36.8) at
    the same k. This is why `lr=1e-3` (not a smaller rate) is the
    default here - a finding from direct comparison, documented in
    course_context/PROBLEM5_REPORT.md, not an assumption.

    No early-stopping validation split here - Section 35 explicitly
    permits documenting this as a limitation rather than carving a
    validation set out of as few as 10 samples. Instead, a small FIXED
    epoch budget is used (see course_context/PROBLEM5_REPORT.md).

    Parameters
    ----------
    pretrained_model : TransferMLP
        The Davis-pretrained model - copied internally, not modified.
    X_k, y_k : numpy.ndarray
        The k Amherst samples to fine-tune on.
    seed : int
    freeze_first_layer : bool
        If True, layer1's weights stay fixed during fine-tuning
        (Section 15-16, 34's freezing ablation).
    lr : float
    epochs : int
        Fixed epoch count (no early stopping - see above).

    Returns
    -------
    TransferMLP
        A new, fine-tuned model (the input model is untouched).
    """
    utils.set_seed(seed)
    device = utils.get_device()

    model = copy.deepcopy(pretrained_model).to(device)
    if freeze_first_layer:
        model.freeze_first_layer()
    else:
        model.unfreeze_all()

    batch_size = min(16, len(X_k))
    train_loader = torch_utils.make_dataloader(X_k, y_k, batch_size=batch_size, shuffle=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    fine-tune epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    model.unfreeze_all()  # restore trainable state for any future reuse
    return model


def train_fresh_on_target(X_k: np.ndarray, y_k: np.ndarray, seed: int, input_dim: int,
                           lr: float = 1e-3, epochs: int = 100, verbose: bool = False) -> p5_models.TransferMLP:
    """
    Section 23: the FEW-SHOT / TARGET-ONLY baseline - a FRESH
    TransferMLP (random initialization, same architecture as the
    transfer model, for a fair comparison), trained ONLY on the k
    Amherst samples. Davis is never involved. Same fixed-epoch-budget
    reasoning as fine_tune_on_target() - k is too small for a
    meaningful validation split.
    """
    utils.set_seed(seed)
    device = utils.get_device()

    model = p5_models.TransferMLP(input_dim=input_dim).to(device)
    batch_size = min(16, len(X_k))
    train_loader = torch_utils.make_dataloader(X_k, y_k, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    few-shot epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    return model


# ---------------------------------------------------------------------------
# Few-shot sampling
# ---------------------------------------------------------------------------


def sample_k_target_rows(X_train_pool: pd.DataFrame, y_train_pool: pd.Series, k: int, seed: int) -> tuple:
    """
    Section 10-11: sample k rows from the Amherst TRAINING pool
    (never the test split), using `src/splitting.py`'s established
    `few_shot_sample()` (Phase 2) for the actual sampling, then apply
    the SAME chosen row indices to both X and y so they stay aligned.

    Plain random sampling with a fixed seed - not stratified. The
    "reasonable default" the instructions call out (Section 10);
    stratifying Output Power itself isn't obviously well-defined for a
    continuous regression target the way it was for Problem 4's
    classification labels, so the simpler, equally-legitimate default
    is used and documented as such.

    Returns
    -------
    (X_k, y_k) : tuple of numpy.ndarray - THE SAME k rows are used for
        both the few-shot baseline and the transfer model at this
        (k, seed) - Section 11's fairness requirement.
    """
    sampled_index = splitting.few_shot_sample(X_train_pool, k=k, seed=seed).index
    X_k = X_train_pool.loc[sampled_index].values.astype(np.float32)
    y_k = y_train_pool.loc[sampled_index].values.astype(np.float32)
    return X_k, y_k


# ---------------------------------------------------------------------------
# Recording results
# ---------------------------------------------------------------------------


def record_result(method, target_samples, seed, metrics, target_normalization="none",
                   frozen_layers="none", transfer_gain=None, transfer_gain_percent=None, notes=""):
    """
    Append one result row to results/problem5/problem5_results.csv
    (Section 38's schema). Writes to disk immediately - this project's
    stages are run as several separate invocations to stay within a
    single command's runtime limit (established in Problems 1-4).
    """
    row = {
        "source_city": SOURCE_CITY, "target_city": TARGET_CITY, "method": method,
        "target_samples": target_samples, "seed": seed,
        "target_normalization": target_normalization, "frozen_layers": frozen_layers,
        "rmse": metrics["rmse"], "mae": metrics["mae"], "nrmse": metrics["nrmse"],
        "transfer_gain": transfer_gain, "transfer_gain_percent": transfer_gain_percent,
        "notes": notes,
    }

    utils.ensure_dir(RESULTS_DIR)
    file_exists = os.path.exists(PROBLEM5_RESULTS_PATH)
    with open(PROBLEM5_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROBLEM5_RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    experiment_id = experiment_runner.generate_experiment_id("problem5", method, f"{SOURCE_CITY}_to_{TARGET_CITY}_k{target_samples}", seed)
    experiment_runner.save_result({
        "experiment_id": experiment_id,
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "problem": "problem5",
        "model": method,
        "dataset": f"transfer_k{target_samples}",
        "source_city": SOURCE_CITY, "target_city": TARGET_CITY, "seed": seed,
        "parameters": "{}", "metric": "rmse", "score": metrics["rmse"],
        "runtime_seconds": "", "notes": notes,
    })


# ---------------------------------------------------------------------------
# Main experiment: zero-shot, few-shot, transfer
# ---------------------------------------------------------------------------


def run_zero_shot(data: dict, pretrained_models: dict) -> dict:
    """
    Section 22: Davis-pretrained model, applied directly to the
    Amherst test set - zero Amherst rows used anywhere in fitting.
    One evaluation per seed (each seed's own Davis pretraining run).
    """
    print("\n--- Zero-shot (Davis -> Amherst, no fine-tuning) ---")
    device = utils.get_device()
    results = {}
    for seed in SEEDS:
        model = pretrained_models[seed]
        y_pred = p5_models.predict_values(model, data["target_test"]["X_test"].values, device)
        metrics = evaluation.regression_metrics(data["target_test"]["y_test"], y_pred)
        record_result("zero_shot", 0, seed, metrics, notes="Davis-pretrained model, zero Amherst training")
        results[seed] = metrics
        print(f"  seed={seed}  rmse={metrics['rmse']:.2f} mae={metrics['mae']:.2f} nrmse={metrics['nrmse']:.4f}")
    return results


def run_few_shot_and_transfer_for_k(data: dict, pretrained_models: dict, k: int, save_models: bool = True) -> dict:
    """
    Sections 8B/8C, 23-24: for one k, at every seed - sample the SAME
    k Amherst rows once, then train BOTH the few-shot baseline (fresh
    model) and the transfer model (fine-tuned from that seed's Davis
    pretrained model) on those exact same rows, evaluated on the same
    Amherst test set.
    """
    print(f"\n--- k={k} ---")
    device = utils.get_device()
    input_dim = data["target_train_pool"]["X_train"].shape[1]
    few_shot_metrics, transfer_metrics = [], []

    for seed in SEEDS:
        X_k, y_k = sample_k_target_rows(data["target_train_pool"]["X_train"], data["target_train_pool"]["y_train"], k, seed)

        few_shot_model = train_fresh_on_target(X_k, y_k, seed, input_dim=input_dim)
        y_pred_fs = p5_models.predict_values(few_shot_model, data["target_test"]["X_test"].values, device)
        fs_metrics = evaluation.regression_metrics(data["target_test"]["y_test"], y_pred_fs)
        record_result("few_shot", k, seed, fs_metrics, notes="fresh model, Amherst-only")
        few_shot_metrics.append(fs_metrics)

        transfer_model = fine_tune_on_target(pretrained_models[seed], X_k, y_k, seed, freeze_first_layer=False)
        y_pred_tr = p5_models.predict_values(transfer_model, data["target_test"]["X_test"].values, device)
        tr_metrics = evaluation.regression_metrics(data["target_test"]["y_test"], y_pred_tr)
        gain = fs_metrics["rmse"] - tr_metrics["rmse"]
        gain_pct = 100 * gain / fs_metrics["rmse"] if fs_metrics["rmse"] != 0 else None
        record_result("transfer", k, seed, tr_metrics, frozen_layers="none",
                       transfer_gain=gain, transfer_gain_percent=gain_pct, notes="fine-tuned from Davis-pretrained model")
        transfer_metrics.append(tr_metrics)

        print(f"  seed={seed}  few_shot_rmse={fs_metrics['rmse']:.2f}  transfer_rmse={tr_metrics['rmse']:.2f}  "
              f"gain={gain:+.2f} ({gain_pct:+.1f}%)")

        if save_models and seed == 42:
            utils.ensure_dir(MODELS_DIR)
            torch.save(few_shot_model.state_dict(), os.path.join(MODELS_DIR, f"fewshot_{k}.pt"))
            torch.save(transfer_model.state_dict(), os.path.join(MODELS_DIR, f"transfer_{k}.pt"))

    return {"few_shot": few_shot_metrics, "transfer": transfer_metrics}


# ---------------------------------------------------------------------------
# Freezing ablation (Sections 15-16, 34) - k=50
# ---------------------------------------------------------------------------


def run_freezing_ablation(data: dict, pretrained_models: dict, k: int = 50):
    """
    Section 34: full fine-tuning (every layer trainable) vs. freezing
    layer1 (the 128-unit layer closest to the input) during
    fine-tuning, at k=50 - tests whether the general weather/
    irradiance representation Davis pretraining learned is worth
    keeping fixed, with only the later, city-specific layers adapting.

    Uses the SAME k=50 samples per seed as the main transfer
    experiment (Section 11's fairness rule extends naturally here).
    """
    print(f"\n--- Freezing ablation (k={k}) ---")
    device = utils.get_device()

    for seed in SEEDS:
        X_k, y_k = sample_k_target_rows(data["target_train_pool"]["X_train"], data["target_train_pool"]["y_train"], k, seed)

        frozen_model = fine_tune_on_target(pretrained_models[seed], X_k, y_k, seed, freeze_first_layer=True)
        y_pred = p5_models.predict_values(frozen_model, data["target_test"]["X_test"].values, device)
        metrics = evaluation.regression_metrics(data["target_test"]["y_test"], y_pred)
        record_result("transfer", k, seed, metrics, frozen_layers="layer1_frozen",
                       notes="freezing ablation - layer1 (128-unit) frozen during fine-tuning")
        print(f"  seed={seed}  frozen_layer1_rmse={metrics['rmse']:.2f}")


# ---------------------------------------------------------------------------
# Target normalization ablation (Sections 5-6, 33)
# ---------------------------------------------------------------------------


def run_normalization_ablation(data: dict, k: int = 50):
    """
    Section 33: raw-kW transfer (already the primary approach above)
    vs. a NORMALIZED-output transfer, to directly test the value of
    accounting for Davis/Amherst's different output scales.

    Normalized approach: Davis is pretrained on Davis-normalized
    target (scaler fit on Davis TRAINING data only -
    `src/preprocessing.py`'s `fit_target_scaler()`, Phase 2). For
    fine-tuning, an AMHERST-specific scaler is fit using ONLY the k
    few-shot samples' own target statistics (the sole Amherst target
    information legitimately available in a k-shot setting) - the
    model is fine-tuned on Amherst-normalized target, then predictions
    are inverse-transformed with that SAME Amherst scaler before
    computing kW-based RMSE/MAE/nRMSE (Section 6, step 5).

    Since this ablation trains an entirely separate Davis pretrained
    model (on a different target representation), it is run once, at
    k=50, across all 3 seeds - not for every k, to keep this
    secondary ablation's cost proportionate (Section 33: "if
    computationally practical").
    """
    print(f"\n--- Target normalization ablation (k={k}) ---")
    device = utils.get_device()

    davis_target_scaler = preprocessing.fit_target_scaler(data["source"]["y_train"])

    for seed in SEEDS:
        y_train_normalized = preprocessing.apply_target_scaler(data["source"]["y_train"], davis_target_scaler)
        source_normalized = {"X_train": data["source"]["X_train"], "y_train": y_train_normalized}
        pretrained_normalized_model, _ = pretrain_on_source(source_normalized, seed=seed)

        X_k, y_k_raw = sample_k_target_rows(data["target_train_pool"]["X_train"], data["target_train_pool"]["y_train"], k, seed)
        amherst_target_scaler = preprocessing.fit_target_scaler(pd.Series(y_k_raw))
        y_k_normalized = preprocessing.apply_target_scaler(pd.Series(y_k_raw), amherst_target_scaler).astype(np.float32)

        fine_tuned_model = fine_tune_on_target(pretrained_normalized_model, X_k, y_k_normalized, seed, freeze_first_layer=False)

        y_pred_normalized = p5_models.predict_values(fine_tuned_model, data["target_test"]["X_test"].values, device)
        y_pred_kw = preprocessing.inverse_transform_target(y_pred_normalized, amherst_target_scaler)
        metrics = evaluation.regression_metrics(data["target_test"]["y_test"], y_pred_kw)

        record_result("transfer", k, seed, metrics, target_normalization="davis_and_amherst_normalized",
                       notes="normalization ablation - Davis pretrained on Davis-normalized target, fine-tuned on Amherst-k-sample-normalized target")
        print(f"  seed={seed}  normalized_transfer_rmse={metrics['rmse']:.2f}")


# ---------------------------------------------------------------------------
# Domain shift analysis (Section 21)
# ---------------------------------------------------------------------------

DOMAIN_SHIFT_VARIABLES = [
    "GHI", "DNI", "DHI", "Temperature", "Relative Humidity", "Wind Speed",
    "Solar Zenith Angle", "Clear_Sky_Index", "Output Power",
]


def run_domain_shift_analysis() -> pd.DataFrame:
    """
    Section 21: compare Davis vs. Amherst distributions of the key
    weather/irradiance variables and Output Power itself, to help
    explain (not just report) the transfer results. Uses each city's
    RAW training data (via the same load/clean/feature-engineer steps
    as everywhere else in this project) - this analysis only looks at
    already-public distributional facts about each city, not anything
    that would leak test information into a model.

    Standardized mean difference (SMD) = (mean_davis - mean_amherst) /
    pooled_std - a common way to compare distribution shift that's
    comparable across variables with very different natural units.
    """
    from src import cleaning, data_loader, feature_engineering

    rows = []
    city_data = {}
    for city in [SOURCE_CITY, TARGET_CITY]:
        raw_df = data_loader.load_city(city, years="long")
        cleaned_df, _ = cleaning.clean_sheet(raw_df, target_column="Output Power", missing_strategy="drop", verbose=False)
        featured_df = feature_engineering.add_feature_groups(cleaned_df, ["clear_sky_index", "time_cyclical"])
        city_data[city] = featured_df

    for var in DOMAIN_SHIFT_VARIABLES:
        davis_values = city_data[SOURCE_CITY][var].dropna()
        amherst_values = city_data[TARGET_CITY][var].dropna()
        pooled_std = np.sqrt((davis_values.std() ** 2 + amherst_values.std() ** 2) / 2)
        smd = (davis_values.mean() - amherst_values.mean()) / pooled_std if pooled_std != 0 else np.nan

        rows.append({
            "variable": var,
            "davis_mean": davis_values.mean(), "davis_std": davis_values.std(),
            "amherst_mean": amherst_values.mean(), "amherst_std": amherst_values.std(),
            "standardized_mean_difference": smd,
        })
        print(f"  {var:20s} Davis={davis_values.mean():8.2f}±{davis_values.std():6.2f}  "
              f"Amherst={amherst_values.mean():8.2f}±{amherst_values.std():6.2f}  SMD={smd:+.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "problem5_domain_shift.csv"), index=False)
    return table, city_data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_domain_shift_figure(city_data: dict):
    """problem5_domain_shift.png - REQUIRED."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    plot_vars = ["GHI", "Temperature", "Wind Speed", "Output Power"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, var in zip(axes, plot_vars):
        davis_vals = city_data[SOURCE_CITY][var].dropna()
        amherst_vals = city_data[TARGET_CITY][var].dropna()
        ax.hist(davis_vals, bins=40, alpha=0.5, label="Davis", density=True)
        ax.hist(amherst_vals, bins=40, alpha=0.5, label="Amherst", density=True)
        ax.set_title(var)
        ax.legend(fontsize=8)
    fig.suptitle("Domain shift: Davis vs. Amherst feature distributions")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem5_domain_shift.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_transfer_curve_figure():
    """problem5_transfer_curve.png - REQUIRED."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM5_RESULTS_PATH)
    zero_shot = df[df.method == "zero_shot"]["rmse"].mean()
    few_shot_by_k = df[df.method == "few_shot"].groupby("target_samples")["rmse"].mean()
    transfer_by_k = df[(df.method == "transfer") & (df.frozen_layers == "none") & (df.target_normalization == "none")].groupby("target_samples")["rmse"].mean()

    x_fs = [0] + list(few_shot_by_k.index)
    y_fs = [zero_shot] + list(few_shot_by_k.values)
    x_tr = [0] + list(transfer_by_k.index)
    y_tr = [zero_shot] + list(transfer_by_k.values)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_fs, y_fs, marker="o", label="Few-shot (target-only)")
    ax.plot(x_tr, y_tr, marker="s", label="Transfer (Davis-pretrained + fine-tuned)")
    ax.axhline(zero_shot, color="gray", linestyle="--", alpha=0.6, label="Zero-shot (0 samples)")
    ax.set_xlabel("Number of Amherst labeled samples")
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("Transfer curve: RMSE vs. Amherst labeled sample count")
    ax.set_xticks([0, 10, 50, 100])
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem5_transfer_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_prediction_vs_truth_figure(data: dict, pretrained_models: dict):
    """problem5_prediction_vs_truth.png - REQUIRED. Best few-shot vs. best transfer model at k=10 (seed=42)."""
    device = utils.get_device()
    X_k, y_k = sample_k_target_rows(data["target_train_pool"]["X_train"], data["target_train_pool"]["y_train"], 10, seed=42)

    few_shot_model = train_fresh_on_target(X_k, y_k, seed=42, input_dim=data["target_train_pool"]["X_train"].shape[1])
    transfer_model = fine_tune_on_target(pretrained_models[42], X_k, y_k, seed=42, freeze_first_layer=False)

    y_pred_fs = p5_models.predict_values(few_shot_model, data["target_test"]["X_test"].values, device)
    y_pred_tr = p5_models.predict_values(transfer_model, data["target_test"]["X_test"].values, device)
    y_true = data["target_test"]["y_test"].values

    import matplotlib.pyplot as plt
    visualization.set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, y_pred, title in zip(axes, [y_pred_fs, y_pred_tr], ["Few-shot (k=10)", "Transfer (k=10)"]):
        ax.scatter(y_true, y_pred, s=8, alpha=0.4)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("True Output Power (kW)")
        ax.set_ylabel("Predicted Output Power (kW)")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Predicted vs. actual Output Power - Amherst test set")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem5_prediction_vs_truth.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return y_pred_fs, y_pred_tr, y_true


def make_time_series_figure(data: dict, y_pred_transfer: np.ndarray):
    """problem5_time_series.png - REQUIRED. Representative chronological section of the Amherst test set."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    n_points = min(5 * 11, len(data["target_test"]["y_test"]))  # ~5 days at 11 samples/day
    y_true = data["target_test"]["y_test"].values[:n_points]
    y_pred = y_pred_transfer[:n_points]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(n_points), y_true, label="Actual", marker="o", markersize=3)
    ax.plot(range(n_points), y_pred, label="Predicted (Transfer, k=10)", marker="x", markersize=3)
    ax.set_xlabel("Test sample index (chronological)")
    ax.set_ylabel("Output Power (kW)")
    ax.set_title("Amherst: actual vs. predicted Output Power (representative test period)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem5_time_series.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_zero_shot_vs_transfer_figure():
    """problem5_zero_shot_vs_transfer.png - REQUIRED."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM5_RESULTS_PATH)
    zero_shot_rmse = df[df.method == "zero_shot"]["rmse"].mean()
    transfer_by_k = df[(df.method == "transfer") & (df.frozen_layers == "none") & (df.target_normalization == "none")].groupby("target_samples")["rmse"].mean()

    labels = ["Zero-shot\n(0 samples)"] + [f"Transfer\n(k={k})" for k in transfer_by_k.index]
    values = [zero_shot_rmse] + list(transfer_by_k.values)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["gray"] + ["tab:blue"] * len(transfer_by_k))
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("Zero-shot vs. Transfer at each label budget")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem5_zero_shot_vs_transfer.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main orchestration — see the note in problem1's run_experiments.py for
# why this was missing and how the fix was verified across the project.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PROBLEM 5 — TRANSFER LEARNING — FULL PIPELINE")
    print("=" * 70)

    # See the matching note in problem1's __main__ block.
    utils.ensure_dir(RESULTS_DIR)
    utils.ensure_dir(MODELS_DIR)
    utils.ensure_dir(FIGURES_DIR)

    data = build_source_and_target_data()
    utils.ensure_dir(MODELS_DIR)

    pretrained_models = {}
    for seed in SEEDS:
        model, history = pretrain_on_source(data["source"], seed=seed)
        pretrained_models[seed] = model
        if seed == 42:
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "davis_pretrained_model.pt"))

    run_zero_shot(data, pretrained_models)

    for k in K_VALUES:
        run_few_shot_and_transfer_for_k(data, pretrained_models, k)

    run_freezing_ablation(data, pretrained_models)
    run_normalization_ablation(data)

    domain_table, city_data = run_domain_shift_analysis()
    make_domain_shift_figure(city_data)
    make_transfer_curve_figure()
    y_pred_fs, y_pred_tr, y_true = make_prediction_vs_truth_figure(data, pretrained_models)
    make_time_series_figure(data, y_pred_tr)
    make_zero_shot_vs_transfer_figure()

    print("\n" + "=" * 70)
    print("PROBLEM 5 COMPLETE")
    print("=" * 70)
