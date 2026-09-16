"""
run_experiments.py (Problem 3 — Dimension Reduction)

Main experiment script for Problem 3:

    1. Build ONE unified Davis dataset carrying both downstream
       targets (sky-condition label, Output Power) alongside a single
       leakage-safe feature set (see features.py for why).
    2. Fit PCA at d=2/5/10 on training features only; report explained
       variance and reconstruction MSE.
    3. Train a small autoencoder at d=2/5/10 (3 seeds each) on
       training features only; report reconstruction MSE.
    4. Downstream classification (sky-condition, Random Forest) and
       downstream regression (Output Power, Gradient Boosting) on:
       raw features, each PCA dimension, each autoencoder dimension.
    5. 2-D visualizations (PCA, autoencoder, t-SNE) colored by label
       AFTER the representation was learned - never used to learn it.
    6. A small feature ablation, a central comparison table, and all
       required figures/results/models.

Usage (from the project root):
    python problems/problem3_dimension_reduction/run_experiments.py
Note: like Problems 1-2, this script's stages were run as several
smaller invocations during development to stay within a single
command's execution-time limit - see course_context/PROBLEM3_REPORT.md.
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
from problems.problem1_classification import targets as p1_targets
from problems.problem2_regression import models as p2_models
from problems.problem1_classification import models as p1_models
from problems.problem3_dimension_reduction import features, reduction

SEEDS = utils.DEFAULT_SEEDS  # [42, 123, 2026]
CITY = "Davis"
DIMENSIONS = [2, 5, 10]

RESULTS_DIR = "results/problem3"
MODELS_DIR = "results/problem3/models"
FIGURES_DIR = "figures/problem3"

PROBLEM3_RESULTS_PATH = os.path.join(RESULTS_DIR, "problem3_results.csv")
PROBLEM3_RESULT_FIELDS = [
    "task", "city", "representation", "reduction_method", "dimension", "downstream_model",
    "seed", "explained_variance", "reconstruction_mse",
    "balanced_accuracy", "macro_f1", "rmse", "mae", "nrmse", "notes",
]


def build_unified_dataset(city: str = CITY) -> dict:
    """
    Build ONE dataset carrying both downstream targets (sky-condition
    label, Output Power) alongside the single unified, leakage-safe
    feature set from features.py.

    Uses the exact same loading/cleaning/feature-engineering/splitting
    pipeline as Problems 1 and 2 (same functions, same order), so the
    resulting chronological split lands on the same row boundary they
    used - required so raw-vs-reduced comparisons are apples-to-apples
    (Section 5 of the Phase 6 instructions).

    Returns
    -------
    dict with keys:
        X_train, X_test : preprocessed (scaled/encoded) unified features
        y_train_class, y_test_class : Sky_Condition string labels
        y_train_reg, y_test_reg : raw-kW Output Power
        feature_columns : list of str (post-encoding)
        preprocessor : dict, from preprocessing.fit_preprocessor()
        split_info : dict
        train_df, test_df : labeled DataFrames (pre-preprocessing)
    """
    raw_df = data_loader.load_city(city, years="long")
    cleaned_df, cleaning_report = cleaning.clean_sheet(
        raw_df, target_column="Output Power", missing_strategy="drop", verbose=False
    )
    featured_df = feature_engineering.add_feature_groups(cleaned_df, ["clear_sky_index", "time_cyclical"])
    featured_df["Sky_Condition"] = p1_targets.make_sky_condition_labels(featured_df)
    featured_df = featured_df[featured_df["Sky_Condition"].notna()].copy()

    train_df, test_df, split_info = splitting.chronological_split(featured_df, train_frac=0.8)

    feature_columns = features.get_feature_columns()
    categorical_columns = [c for c in feature_columns if c in features.CATEGORICAL_COLUMNS]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    extra_columns = feature_columns + ["Sky_Condition", "Output Power"]
    train_selected = train_df[extra_columns]
    test_selected = test_df[extra_columns]

    preprocessor = preprocessing.fit_preprocessor(train_selected, numeric_columns, categorical_columns)
    train_processed = preprocessing.apply_preprocessor(train_selected, preprocessor)
    test_processed = preprocessing.apply_preprocessor(test_selected, preprocessor)

    non_feature_columns = {"Sky_Condition", "Output Power"}
    encoded_feature_columns = [c for c in train_processed.columns if c not in non_feature_columns]

    X_train = train_processed[encoded_feature_columns]
    X_test = test_processed[encoded_feature_columns]

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train_class": train_processed["Sky_Condition"], "y_test_class": test_processed["Sky_Condition"],
        "y_train_reg": train_processed["Output Power"], "y_test_reg": test_processed["Output Power"],
        "feature_columns": encoded_feature_columns,
        "preprocessor": preprocessor, "split_info": split_info,
        "train_df": train_df, "test_df": test_df,
    }


def record_result(task, representation, dimension, downstream_model, seed, city=CITY,
                   reduction_method="none", explained_variance=None, reconstruction_mse=None,
                   metrics=None, notes=""):
    """
    Append one result row to results/problem3/problem3_results.csv
    (Section 31's schema). Writes to disk immediately - this project's
    stages are run as several separate invocations to stay within a
    single command's runtime limit (established in Problems 1-2).
    """
    metrics = metrics or {}
    row = {
        "task": task, "city": city, "representation": representation, "reduction_method": reduction_method,
        "dimension": dimension, "downstream_model": downstream_model, "seed": seed,
        "explained_variance": explained_variance, "reconstruction_mse": reconstruction_mse,
        "balanced_accuracy": metrics.get("balanced_accuracy"), "macro_f1": metrics.get("f1_macro"),
        "rmse": metrics.get("rmse"), "mae": metrics.get("mae"), "nrmse": metrics.get("nrmse"),
        "notes": notes,
    }

    utils.ensure_dir(RESULTS_DIR)
    file_exists = os.path.exists(PROBLEM3_RESULTS_PATH)
    with open(PROBLEM3_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROBLEM3_RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    headline_metric = "balanced_accuracy" if task == "classification" else "rmse"
    if headline_metric in metrics:
        experiment_id = experiment_runner.generate_experiment_id("problem3", f"{representation}_{downstream_model}", city, seed)
        experiment_runner.save_result({
            "experiment_id": experiment_id,
            "timestamp": pd.Timestamp.now("UTC").isoformat(),
            "problem": "problem3",
            "model": downstream_model,
            "dataset": f"{task}_{representation}",
            "source_city": city, "target_city": "", "seed": seed,
            "parameters": "{}", "metric": headline_metric, "score": metrics[headline_metric],
            "runtime_seconds": "", "notes": notes,
        })


def run_pca_stage(data: dict) -> dict:
    """
    Sections 9-11: fit PCA at d=2/5/10 (and a few extra values for a
    clearer explained-variance elbow) on TRAINING features only.
    Report cumulative explained variance and reconstruction MSE (on
    the TEST set, since that's the meaningful "how well does this
    generalize" number - PCA itself never sees test data while being
    fit).

    Returns
    -------
    dict: {d: fitted PCA object}, for d in DIMENSIONS
    """
    print("\n--- Stage 1: PCA ---")
    pca_models = {}

    elbow_dimensions = sorted(set(DIMENSIONS + [1, 3, 7, 15, 20, len(data["feature_columns"])]))
    variance_curve = []
    for d in elbow_dimensions:
        if d > len(data["feature_columns"]):
            continue
        pca = reduction.fit_pca(data["X_train"], n_components=d)
        cumulative_variance = float(np.sum(pca.explained_variance_ratio_))
        variance_curve.append({"d": d, "cumulative_explained_variance": cumulative_variance})
        if d in DIMENSIONS:
            pca_models[d] = pca
            recon_mse = reduction.pca_reconstruction_mse(pca, data["X_test"])
            print(f"  d={d}: cumulative_explained_variance={cumulative_variance:.4f}, reconstruction_mse(test)={recon_mse:.4f}")
            joblib.dump(pca, os.path.join(MODELS_DIR, f"pca_{d}.joblib"))

    pd.DataFrame(variance_curve).to_csv(os.path.join(RESULTS_DIR, "problem3_pca_summary.csv"), index=False)
    return pca_models


def train_autoencoder(data: dict, latent_dim: int, seed: int, hidden_size: int = 32,
                       lr: float = 1e-3, max_epochs: int = 80, patience: int = 10,
                       verbose: bool = False) -> tuple:
    """
    Train a SimpleAutoencoder at one latent dimension and seed, using
    TRAINING features only (the autoencoder's "target" is its own
    input - it never sees a label). Early stopping uses a
    chronological inner-validation split of the training data.

    Returns
    -------
    (model, history) : tuple
    """
    utils.set_seed(seed)
    device = utils.get_device()

    X_train_full = data["X_train"].values.astype(np.float32).copy()
    n_inner_train = int(len(X_train_full) * 0.8)
    X_inner_train, X_inner_val = X_train_full[:n_inner_train].copy(), X_train_full[n_inner_train:].copy()

    train_loader = torch_utils.make_dataloader(X_inner_train, X_inner_train, batch_size=64, shuffle=True)
    val_loader = torch_utils.make_dataloader(X_inner_val, X_inner_val, batch_size=64, shuffle=False)

    model = reduction.SimpleAutoencoder(input_dim=X_train_full.shape[1], latent_dim=latent_dim, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = torch_utils.train_torch_model(
        model, train_loader, val_loader, optimizer, nn.MSELoss(), device,
        max_epochs=max_epochs, patience=patience, verbose=verbose,
    )
    return model, history


def run_autoencoder_stage(data: dict) -> dict:
    """
    Sections 16-18: train the autoencoder at d=2/5/10, 3 seeds each,
    report reconstruction MSE (test set). Saves the seed=42 model of
    each dimension (used for downstream experiments and 2-D
    visualization).

    Returns
    -------
    dict: {d: model (seed=42)}, for d in DIMENSIONS
    """
    print("\n--- Stage 2: Autoencoder ---")
    device = utils.get_device()
    ae_models = {}
    summary_rows = []

    for d in DIMENSIONS:
        recon_mses = []
        for seed in SEEDS:
            model, history = train_autoencoder(data, latent_dim=d, seed=seed)
            recon_mse = reduction.autoencoder_reconstruction_mse(model, data["X_test"].values, device)
            recon_mses.append(recon_mse)
            print(f"  d={d} seed={seed}  reconstruction_mse(test)={recon_mse:.4f} (epochs={len(history['epoch'])})")
            if seed == 42:
                ae_models[d] = model
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"autoencoder_{d}.pt"))

        summary_rows.append({
            "d": d, "reconstruction_mse_mean": float(np.mean(recon_mses)), "reconstruction_mse_std": float(np.std(recon_mses)),
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, "problem3_autoencoder_summary.csv"), index=False)
    return ae_models


def get_representation(data: dict, representation: str, dimension: int = None,
                        pca_models: dict = None, ae_models: dict = None, device: str = None) -> tuple:
    """
    Return (X_train_repr, X_test_repr) for one representation - "raw"
    (the unified leakage-safe features, unchanged), "pca" (transform
    with an already-fitted PCA), or "autoencoder" (encode with an
    already-trained autoencoder). Never re-fits anything here - every
    representation was already fit on training data only, upstream.
    """
    if representation == "raw":
        return data["X_train"], data["X_test"]
    elif representation == "pca":
        pca = pca_models[dimension]
        return pca.transform(data["X_train"]), pca.transform(data["X_test"])
    elif representation == "autoencoder":
        model = ae_models[dimension]
        X_train_repr = reduction.encode_with_autoencoder(model, data["X_train"].values, device)
        X_test_repr = reduction.encode_with_autoencoder(model, data["X_test"].values, device)
        return X_train_repr, X_test_repr
    else:
        raise ValueError(f"Unknown representation '{representation}'")


ALL_REPRESENTATIONS = [("raw", None)] + [("pca", d) for d in DIMENSIONS] + [("autoencoder", d) for d in DIMENSIONS]


def run_downstream_classification(data: dict, pca_models: dict, ae_models: dict):
    """
    Sections 12-13, 19: RAW vs PCA(2,5,10) vs Autoencoder(2,5,10) as
    input to a Random Forest sky-condition classifier - the same
    consistent "workhorse" model choice used for Problem 1's feature
    ablation, for the same reason (comparable results across
    representations, not confounded by also varying the model).
    """
    print("\n--- Stage 3: Downstream classification (sky-condition) ---")
    device = utils.get_device()

    for representation, dimension in ALL_REPRESENTATIONS:
        X_train_repr, X_test_repr = get_representation(data, representation, dimension, pca_models, ae_models, device)
        label = f"{representation}" + (f"-{dimension}" if dimension else "")
        seed_metrics = []
        for seed in SEEDS:
            model = p1_models.get_classical_models(seed)["random_forest"]
            model.fit(X_train_repr, data["y_train_class"])
            y_pred = model.predict(X_test_repr)
            metrics = evaluation.classification_metrics(data["y_test_class"], y_pred)
            seed_metrics.append(metrics)
            record_result(
                "classification", representation, dimension or "raw", "random_forest", seed,
                reduction_method="none" if representation == "raw" else representation, metrics=metrics,
            )
        mean_balanced_acc = np.mean([m["balanced_accuracy"] for m in seed_metrics])
        mean_macro_f1 = np.mean([m["f1_macro"] for m in seed_metrics])
        print(f"  {label:16s} balanced_acc={mean_balanced_acc:.3f} macro_f1={mean_macro_f1:.3f} (mean of 3 seeds)")


def run_downstream_regression(data: dict, pca_models: dict, ae_models: dict):
    """
    Sections 14, 20: RAW vs PCA(2,5,10) vs Autoencoder(2,5,10) as
    input to a Gradient Boosting Output Power regressor - Problem 2's
    established Davis champion model, used consistently here too.
    """
    print("\n--- Stage 4: Downstream regression (Output Power) ---")
    device = utils.get_device()

    for representation, dimension in ALL_REPRESENTATIONS:
        X_train_repr, X_test_repr = get_representation(data, representation, dimension, pca_models, ae_models, device)
        label = f"{representation}" + (f"-{dimension}" if dimension else "")
        seed_metrics = []
        for seed in SEEDS:
            model = p2_models.get_classical_models(seed)["gradient_boosting"]
            model.fit(X_train_repr, data["y_train_reg"])
            y_pred = model.predict(X_test_repr)
            metrics = evaluation.regression_metrics(data["y_test_reg"], y_pred)
            seed_metrics.append(metrics)
            record_result(
                "regression", representation, dimension or "raw", "gradient_boosting", seed,
                reduction_method="none" if representation == "raw" else representation, metrics=metrics,
            )
        mean_rmse = np.mean([m["rmse"] for m in seed_metrics])
        mean_mae = np.mean([m["mae"] for m in seed_metrics])
        mean_nrmse = np.mean([m["nrmse"] for m in seed_metrics])
        print(f"  {label:16s} rmse={mean_rmse:.2f} mae={mean_mae:.2f} nrmse={mean_nrmse:.4f} (mean of 3 seeds)")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def make_explained_variance_figure():
    """problem3_explained_variance.png - cumulative explained variance vs. d."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    curve = pd.read_csv(os.path.join(RESULTS_DIR, "problem3_pca_summary.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["d"], curve["cumulative_explained_variance"], marker="o")
    for d in DIMENSIONS:
        row = curve[curve["d"] == d]
        if len(row):
            ax.axvline(d, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Number of PCA components (d)")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA explained variance vs. number of components")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem3_explained_variance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_reconstruction_error_figure():
    """problem3_reconstruction_error.png - PCA vs. autoencoder reconstruction MSE, by d."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM3_RESULTS_PATH)
    # Reconstruction MSE isn't in the downstream results rows (those
    # are per-seed downstream metrics) - read from the two summary files
    # saved directly by the PCA/AE stages instead.
    pca_summary = pd.read_csv(os.path.join(RESULTS_DIR, "problem3_pca_summary.csv"))
    pca_summary = pca_summary[pca_summary["d"].isin(DIMENSIONS)]
    ae_summary = pd.read_csv(os.path.join(RESULTS_DIR, "problem3_autoencoder_summary.csv"))

    # PCA summary doesn't include reconstruction MSE (only explained
    # variance) - recompute it directly here from the saved PCA models
    # for a clean, single source of truth for this figure.
    pca_recon = []
    data = build_unified_dataset()
    for d in DIMENSIONS:
        pca = joblib.load(os.path.join(MODELS_DIR, f"pca_{d}.joblib"))
        pca_recon.append(reduction.pca_reconstruction_mse(pca, data["X_test"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(DIMENSIONS, pca_recon, marker="o", label="PCA")
    ax.plot(ae_summary["d"], ae_summary["reconstruction_mse_mean"], marker="s", label="Autoencoder")
    ax.set_xlabel("Latent dimension (d)")
    ax.set_ylabel("Reconstruction MSE (test set)")
    ax.set_title("Reconstruction error: PCA vs. Autoencoder")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem3_reconstruction_error.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_downstream_comparison_figures():
    """problem3_downstream_classification.png, problem3_downstream_regression.png"""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    df = pd.read_csv(PROBLEM3_RESULTS_PATH, dtype={"dimension": str})
    labels = ["raw"] + [f"pca-{d}" for d in DIMENSIONS] + [f"autoencoder-{d}" for d in DIMENSIONS]

    # Classification
    cls = df[df["task"] == "classification"]
    cls_summary = cls.groupby(["representation", "dimension"])["balanced_accuracy"].mean()
    cls_values = [cls_summary.get(("raw", "raw"), np.nan)] + \
                 [cls_summary.get(("pca", str(d)), np.nan) for d in DIMENSIONS] + \
                 [cls_summary.get(("autoencoder", str(d)), np.nan) for d in DIMENSIONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, cls_values)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Downstream classification (sky-condition): raw vs. reduced")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem3_downstream_classification.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Regression
    reg = df[df["task"] == "regression"]
    reg_summary = reg.groupby(["representation", "dimension"])["rmse"].mean()
    reg_values = [reg_summary.get(("raw", "raw"), np.nan)] + \
                 [reg_summary.get(("pca", str(d)), np.nan) for d in DIMENSIONS] + \
                 [reg_summary.get(("autoencoder", str(d)), np.nan) for d in DIMENSIONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, reg_values, color="orange")
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("Downstream regression (Output Power): raw vs. reduced")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem3_downstream_regression.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_2d_scatter_figures(data: dict, pca_models: dict, ae_models: dict):
    """
    problem3_pca_2d_sky.png, problem3_autoencoder_2d_sky.png. Labels
    are used ONLY to color the points here - never to learn PCA/the
    autoencoder (both were already fully fit before this function is
    even called).
    """
    import matplotlib.pyplot as plt
    visualization.set_plot_style()
    device = utils.get_device()

    # A representative sample for readability (Section 24) - not
    # cherry-picked by label, a plain random sample of test rows.
    rng = np.random.default_rng(42)
    n_sample = min(2000, len(data["X_test"]))
    sample_idx = rng.choice(len(data["X_test"]), size=n_sample, replace=False)

    labels = data["y_test_class"].values[sample_idx]
    class_order = ["Clear", "Partly Cloudy", "Overcast"]
    colors = {"Clear": "tab:orange", "Partly Cloudy": "tab:gray", "Overcast": "tab:blue"}

    for method_name, path_suffix in [("pca", "pca_2d_sky"), ("autoencoder", "autoencoder_2d_sky")]:
        if method_name == "pca":
            pca = pca_models[2]
            coords = pca.transform(data["X_test"])[sample_idx]
        else:
            model = ae_models[2]
            coords = reduction.encode_with_autoencoder(model, data["X_test"].values, device)[sample_idx]

        fig, ax = plt.subplots(figsize=(7, 6))
        for cls in class_order:
            mask = labels == cls
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.4, label=cls, color=colors[cls])
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"2-D {method_name.upper()} representation, colored by sky-condition\n(labels used for coloring ONLY, not for learning the representation)")
        ax.legend(markerscale=2)
        fig.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"problem3_{path_suffix}.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path} (n={n_sample} sampled test rows)")


def make_tsne_figure(data: dict):
    """
    problem3_tsne.png - t-SNE is for visualization only, never used as
    a production feature transformation. Fit on a representative
    SAMPLE of the raw (unified) test features for speed - t-SNE scales
    poorly with dataset size (Section 24).
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    rng = np.random.default_rng(42)
    n_sample = min(2000, len(data["X_test"]))
    sample_idx = rng.choice(len(data["X_test"]), size=n_sample, replace=False)

    X_sample = data["X_test"].values[sample_idx]
    labels = data["y_test_class"].values[sample_idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    coords = tsne.fit_transform(X_sample)

    class_order = ["Clear", "Partly Cloudy", "Overcast"]
    colors = {"Clear": "tab:orange", "Partly Cloudy": "tab:gray", "Overcast": "tab:blue"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in class_order:
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.4, label=cls, color=colors[cls])
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title(f"t-SNE of raw unified features (n={n_sample} sampled test rows), colored by sky-condition")
    ax.legend(markerscale=2)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem3_tsne.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Stage 5: small feature ablation
# ---------------------------------------------------------------------------


def run_feature_ablation(data: dict):
    """
    Section 28: a small ablation - PCA with vs. without Cloud Type
    (the one-hot-encoded categorical columns make up ~9 of the 23
    unified features, a substantial chunk, and Cloud Type was found to
    dominate Problem 1's sky-condition feature importance - see
    course_context/PROBLEM1_REPORT.md - so this is a genuinely
    meaningful ablation, not an arbitrary one).
    """
    print("\n--- Stage 5: Feature ablation (PCA with/without Cloud Type) ---")
    cloud_type_columns = [c for c in data["feature_columns"] if c.startswith("Cloud Type")]
    non_cloud_columns = [c for c in data["feature_columns"] if c not in cloud_type_columns]

    for label, columns in [("with_cloud_type", data["feature_columns"]), ("without_cloud_type", non_cloud_columns)]:
        X_train_subset = data["X_train"][columns]
        X_test_subset = data["X_test"][columns]
        pca = reduction.fit_pca(X_train_subset, n_components=5)
        recon_mse = reduction.pca_reconstruction_mse(pca, X_test_subset)
        cumulative_variance = float(np.sum(pca.explained_variance_ratio_))

        for seed in SEEDS:
            model = p1_models.get_classical_models(seed)["random_forest"]
            X_train_reduced = pca.transform(X_train_subset)
            X_test_reduced = pca.transform(X_test_subset)
            model.fit(X_train_reduced, data["y_train_class"])
            y_pred = model.predict(X_test_reduced)
            metrics = evaluation.classification_metrics(data["y_test_class"], y_pred)
            record_result(
                "classification", "pca", 5, "random_forest", seed,
                reduction_method="pca", explained_variance=cumulative_variance, reconstruction_mse=recon_mse,
                metrics=metrics, notes=f"feature_ablation={label}",
            )
        print(f"  PCA-5 {label:20s} n_features={len(columns)} explained_var={cumulative_variance:.4f} "
              f"recon_mse={recon_mse:.4f} balanced_acc(seed42)={metrics['balanced_accuracy']:.3f}")


# ---------------------------------------------------------------------------
# Central comparison table
# ---------------------------------------------------------------------------


def build_comparison_table() -> pd.DataFrame:
    """
    Section 26: one central table - Raw, PCA-2/5/10, AE-2/5/10 - with
    explained variance, reconstruction MSE, and both downstream tasks'
    metrics side by side. Uses "N/A" where a metric doesn't apply
    (e.g. explained variance for an autoencoder, or reconstruction MSE
    for raw features).
    """
    df = pd.read_csv(PROBLEM3_RESULTS_PATH, dtype={"dimension": str})
    df = df[~df["notes"].astype(str).str.contains("feature_ablation", na=False)]  # exclude the ablation rows

    pca_summary = pd.read_csv(os.path.join(RESULTS_DIR, "problem3_pca_summary.csv")).set_index("d")
    ae_summary = pd.read_csv(os.path.join(RESULTS_DIR, "problem3_autoencoder_summary.csv")).set_index("d")

    rows = []
    for representation, dim_label, dim_value in [("raw", "raw", "raw")] + \
            [("pca", f"PCA-{d}", d) for d in DIMENSIONS] + \
            [("autoencoder", f"AE-{d}", d) for d in DIMENSIONS]:

        cls_subset = df[(df.task == "classification") & (df.representation == representation) & (df.dimension == str(dim_value))]
        reg_subset = df[(df.task == "regression") & (df.representation == representation) & (df.dimension == str(dim_value))]

        if representation == "raw":
            explained_variance, recon_mse = "N/A", "N/A"
        elif representation == "pca":
            explained_variance = round(pca_summary.loc[dim_value, "cumulative_explained_variance"], 4)
            data_full = build_unified_dataset()
            pca = joblib.load(os.path.join(MODELS_DIR, f"pca_{dim_value}.joblib"))
            recon_mse = round(reduction.pca_reconstruction_mse(pca, data_full["X_test"]), 4)
        else:
            explained_variance = "N/A"
            recon_mse = round(ae_summary.loc[dim_value, "reconstruction_mse_mean"], 4)

        rows.append({
            "Representation": dim_label,
            "Explained Variance": explained_variance,
            "Reconstruction MSE": recon_mse,
            "Classification Balanced Accuracy": round(cls_subset["balanced_accuracy"].mean(), 4) if len(cls_subset) else "N/A",
            "Classification Macro F1": round(cls_subset["macro_f1"].mean(), 4) if len(cls_subset) else "N/A",
            "Regression RMSE": round(reg_subset["rmse"].mean(), 2) if len(reg_subset) else "N/A",
            "Regression MAE": round(reg_subset["mae"].mean(), 2) if len(reg_subset) else "N/A",
            "Regression nRMSE": round(reg_subset["nrmse"].mean(), 4) if len(reg_subset) else "N/A",
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "problem3_comparison_table.csv"), index=False)
    return table


# ---------------------------------------------------------------------------
# Optional: PCA + MLP (Section 15)
# ---------------------------------------------------------------------------


def run_pca_mlp_comparison(data: dict, pca_models: dict):
    """
    Section 15 (optional): does a nonlinear downstream model (MLP)
    change the raw-vs-PCA conclusion from Stage 3's Random Forest
    results? Tests only PCA-10 (the strongest PCA dimension) vs. raw,
    at 3 seeds, to keep this optional addition small per the
    instruction not to spend excessive compute here.
    """
    print("\n--- Optional: PCA + MLP ---")

    for representation, dimension in [("raw", None), ("pca", 10)]:
        X_train_repr, X_test_repr = get_representation(data, representation, dimension, pca_models, None, None)
        label = f"{representation}" + (f"-{dimension}" if dimension else "")

        from problems.problem1_classification import models as p1_mlp_models
        class_order = ["Overcast", "Partly Cloudy", "Clear"]
        class_to_index = {c: i for i, c in enumerate(class_order)}

        seed_metrics = []
        for seed in SEEDS:
            utils.set_seed(seed)
            device = utils.get_device()
            X_train_arr = np.asarray(X_train_repr, dtype=np.float32)
            X_test_arr = np.asarray(X_test_repr, dtype=np.float32)
            y_train_idx = data["y_train_class"].map(class_to_index).values.astype(np.int64)
            y_test_idx = data["y_test_class"].map(class_to_index).values

            n_inner = int(len(X_train_arr) * 0.8)
            train_loader = torch_utils.make_dataloader(X_train_arr[:n_inner], y_train_idx[:n_inner], batch_size=64, shuffle=True, y_dtype=torch.long)
            val_loader = torch_utils.make_dataloader(X_train_arr[n_inner:], y_train_idx[n_inner:], batch_size=64, shuffle=False, y_dtype=torch.long)

            mlp = p1_mlp_models.SimpleMLPClassifier(input_dim=X_train_arr.shape[1], num_classes=3, hidden_size=32, dropout=0.2)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            torch_utils.train_torch_model(mlp, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), device, max_epochs=50, patience=8, verbose=False)

            y_pred_idx = p1_mlp_models.predict_classes(mlp, X_test_arr, device)
            y_pred = np.array([class_order[i] for i in y_pred_idx])
            y_test_labels = np.array([class_order[i] for i in y_test_idx])
            metrics = evaluation.classification_metrics(y_test_labels, y_pred)
            seed_metrics.append(metrics)
            record_result(
                "classification", representation, dimension or "raw", "mlp", seed,
                reduction_method="none" if representation == "raw" else representation, metrics=metrics,
                notes="optional PCA+MLP comparison (Section 15)",
            )
        mean_acc = np.mean([m["balanced_accuracy"] for m in seed_metrics])
        print(f"  {label:10s} + MLP  balanced_acc={mean_acc:.3f} (mean of 3 seeds)")


# ---------------------------------------------------------------------------
# Main orchestration — see the note in problem1's run_experiments.py for
# why this was missing and how the fix was verified across the project.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PROBLEM 3 — DIMENSION REDUCTION — FULL PIPELINE")
    print("=" * 70)

    # This is the exact fix for the FileNotFoundError this project hit on
    # a fresh run: run_pca_stage() and run_autoencoder_stage() (and
    # several figure-saving functions below) assume results/problem3/
    # models/ and figures/problem3/ already exist rather than creating
    # them. Ensuring this upfront closes that gap for every function
    # called below, in one place.
    utils.ensure_dir(RESULTS_DIR)
    utils.ensure_dir(MODELS_DIR)
    utils.ensure_dir(FIGURES_DIR)

    data = build_unified_dataset()
    pca_models = run_pca_stage(data)
    ae_models = run_autoencoder_stage(data)

    run_downstream_classification(data, pca_models, ae_models)
    run_downstream_regression(data, pca_models, ae_models)

    make_explained_variance_figure()
    make_reconstruction_error_figure()
    make_downstream_comparison_figures()
    make_2d_scatter_figures(data, pca_models, ae_models)
    make_tsne_figure(data)

    run_feature_ablation(data)
    build_comparison_table()
    run_pca_mlp_comparison(data, pca_models)

    print("\n" + "=" * 70)
    print("PROBLEM 3 COMPLETE")
    print("=" * 70)
