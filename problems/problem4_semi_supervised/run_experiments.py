"""
run_experiments.py (Problem 4 — Semi-Supervised Learning)

Main experiment script for Problem 4. Reuses Problem 1's sky-condition
classification task and its EXACT chronological train/test split
(Section 5 requires this - no new split is created here).

For each label fraction (10%, 30%, 50%) and seed:
    1. Stratified-sample that fraction of the TRAINING POOL to stay
       labeled; the rest becomes "unlabeled" (X kept, y hidden).
    2. Train a supervised-only baseline on just the labeled subset.
    3. Run pseudo-labeling/self-training using the SAME labeled subset
       plus the unlabeled X.
    4. Evaluate both on the SAME untouched test set.

Then: label-efficiency curve, SSL gain, label-efficiency AUC,
pseudo-label analysis, confusion matrices, and an optional second SSL
method (Label Spreading).

Usage (from the project root):
    python problems/problem4_semi_supervised/run_experiments.py
Note: like Problems 1-3, this script's stages were run as several
smaller invocations during development to stay within a single
command's execution-time limit - see course_context/PROBLEM4_REPORT.md.
"""

import csv
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading

from src import evaluation, experiment_runner, utils, visualization
from problems.problem1_classification.run_experiments import build_task_dataset
from problems.problem4_semi_supervised import pseudo_labeling

SEEDS = utils.DEFAULT_SEEDS  # [42, 123, 2026]
LABEL_FRACTIONS = [0.10, 0.30, 0.50]
CITY = "Davis"
CLASS_ORDER = ["Overcast", "Partly Cloudy", "Clear"]

RESULTS_DIR = "results/problem4"
MODELS_DIR = "results/problem4/models"
FIGURES_DIR = "figures/problem4"

PROBLEM4_RESULTS_PATH = os.path.join(RESULTS_DIR, "problem4_results.csv")
PROBLEM4_RESULT_FIELDS = [
    "task", "city", "seed", "label_fraction", "supervised_or_ssl", "method",
    "balanced_accuracy", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "pseudo_labels_added", "pseudo_label_fraction", "confidence_threshold", "notes",
]


def base_model_factory(seed: int = 42):
    """
    Section 13: Logistic Regression with balanced class weight - this
    is Problem 1's ACTUAL best Davis sky-condition model (verified:
    results/problem1/problem1_results.csv, balanced_accuracy=0.772,
    beating every Random Forest/Gradient Boosting/MLP variant tried
    there), so the SSL-vs-supervised comparison here is against a
    genuinely strong baseline, not an arbitrary one. Also naturally
    well-suited to confidence-based pseudo-labeling, since logistic
    regression's predict_proba already IS its decision function.

    `random_state` is passed explicitly for consistency with every
    other model factory in this project (Problems 1-3), even though
    verified empirically deterministic here regardless (the default
    'lbfgs' solver has no inherent randomness for this data) - passing
    it anyway avoids relying on that solver-specific fact staying true
    if the solver ever changes.
    """
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)


def get_full_training_pool() -> dict:
    """
    Reuse Problem 1's exact sky-condition dataset build for Davis -
    same chronological split, same leakage-safe primary feature set
    (GHI/Clearsky GHI excluded per the label's definition; DHI/DNI/
    Solar Zenith Angle also excluded, per Problem 1's resolved
    decision - course_context/TEACHER_EXPECTATIONS.md - carried
    forward here for full methodological consistency, even though
    Phase 7's instructions only explicitly name GHI/Clearsky GHI).

    Returns
    -------
    dict - same shape as problem1_classification.build_task_dataset()'s
    return value. "X_train"/"y_train" here is the FULL 80% training
    pool, before any labeled/unlabeled split - that split happens
    separately, per label fraction, in split_labeled_unlabeled().
    """
    return build_task_dataset(CITY, "sky_condition", ablation_group="full")


def split_labeled_unlabeled(X_train: pd.DataFrame, y_train: pd.Series, label_fraction: float, seed: int) -> tuple:
    """
    Section 7, 15: randomly (STRATIFIED, to guarantee every class
    survives even at 10% labels) select `label_fraction` of the
    training pool to stay labeled; the rest becomes the unlabeled
    pool (X kept, y hidden from every downstream step except the
    offline diagnostic analysis).

    Parameters
    ----------
    X_train, y_train : the FULL training pool (Problem 1's 80% split).
    label_fraction : float, e.g. 0.10.
    seed : int - the exact labeled/unlabeled split is reproducible and
        saved (Section 7's "exact labeled indices must be saved").

    Returns
    -------
    (X_labeled, y_labeled, X_unlabeled, y_unlabeled_hidden) : tuple
        y_unlabeled_hidden is returned ONLY for later offline
        diagnostic analysis (Section 24) - every training/selection
        step in this file uses X_unlabeled WITHOUT this value.
    """
    X_labeled, X_unlabeled, y_labeled, y_unlabeled_hidden = train_test_split(
        X_train, y_train, train_size=label_fraction, random_state=seed, stratify=y_train
    )
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled_hidden


def check_class_distribution(y_labeled: pd.Series, label_fraction: float, seed: int) -> dict:
    """
    Section 15: verify every class survived the labeled sampling -
    especially important at 10% labels, where a small enough sample
    could theoretically miss a class entirely.
    """
    counts = y_labeled.value_counts().reindex(CLASS_ORDER, fill_value=0)
    missing_classes = [c for c in CLASS_ORDER if counts[c] == 0]
    if missing_classes:
        print(f"  WARNING: label_fraction={label_fraction} seed={seed} - classes missing from labeled subset: {missing_classes}")
    return counts.to_dict()


# ---------------------------------------------------------------------------
# Recording results
# ---------------------------------------------------------------------------


def record_result(label_fraction, supervised_or_ssl, method, seed, metrics, city=CITY,
                   pseudo_labels_added=None, pseudo_label_fraction=None,
                   confidence_threshold=None, notes=""):
    """
    Append one result row to results/problem4/problem4_results.csv
    (Section 29's schema). Writes to disk immediately - this project's
    stages are run as several separate invocations to stay within a
    single command's runtime limit (established in Problems 1-3).
    """
    row = {
        "task": "sky_condition", "city": city, "seed": seed, "label_fraction": label_fraction,
        "supervised_or_ssl": supervised_or_ssl, "method": method,
        "balanced_accuracy": metrics["balanced_accuracy"], "accuracy": metrics["accuracy"],
        "macro_precision": metrics["precision_macro"], "macro_recall": metrics["recall_macro"],
        "macro_f1": metrics["f1_macro"],
        "pseudo_labels_added": pseudo_labels_added, "pseudo_label_fraction": pseudo_label_fraction,
        "confidence_threshold": confidence_threshold, "notes": notes,
    }

    utils.ensure_dir(RESULTS_DIR)
    file_exists = os.path.exists(PROBLEM4_RESULTS_PATH)
    with open(PROBLEM4_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROBLEM4_RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    experiment_id = experiment_runner.generate_experiment_id(
        "problem4", f"{method}_{supervised_or_ssl}", f"{city}_frac{int(label_fraction*100)}", seed
    )
    experiment_runner.save_result({
        "experiment_id": experiment_id,
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "problem": "problem4",
        "model": f"{method}_{supervised_or_ssl}",
        "dataset": f"sky_condition_frac{int(label_fraction*100)}",
        "source_city": city, "target_city": "", "seed": seed,
        "parameters": "{}", "metric": "balanced_accuracy", "score": metrics["balanced_accuracy"],
        "runtime_seconds": "", "notes": notes,
    })


# ---------------------------------------------------------------------------
# Main experiment: supervised baseline vs. pseudo-labeling, per fraction/seed
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.90  # chosen via select_confidence_threshold() - see course_context/PROBLEM4_REPORT.md
MAX_PSEUDO_LABELS_PER_ITERATION = 1000  # safeguard - see course_context/PROBLEM4_REPORT.md for the comparison that justified this
MAX_ITERATIONS = 5


def run_main_experiment(data: dict, label_fraction: float, seed: int, save_models: bool = False) -> dict:
    """
    Sections 4, 8, 9: for one (label_fraction, seed), build the
    labeled/unlabeled split, train the supervised-only baseline, run
    pseudo-labeling/self-training, and evaluate BOTH on the SAME
    untouched test set.

    Returns
    -------
    dict with "supervised_metrics", "ssl_metrics", "ssl_result"
        (the full self_train() return dict, for pseudo-label analysis).
    """
    X_labeled, y_labeled, X_unlabeled, y_unlabeled_hidden = split_labeled_unlabeled(
        data["X_train"], data["y_train"], label_fraction, seed
    )
    class_dist = check_class_distribution(y_labeled, label_fraction, seed)

    # A. Supervised-only baseline - same labeled subset, nothing else.
    supervised_model = base_model_factory(seed)
    supervised_model.fit(X_labeled, y_labeled)
    y_pred_supervised = supervised_model.predict(data["X_test"])
    supervised_metrics = evaluation.classification_metrics(data["y_test"], y_pred_supervised)
    record_result(
        label_fraction, "supervised", "logistic_regression_balanced", seed, supervised_metrics,
        notes=f"class_distribution_labeled={json.dumps(class_dist)}",
    )

    # B. SSL: same labeled subset + unlabeled X, via pseudo-labeling.
    ssl_result = pseudo_labeling.self_train(
        lambda: base_model_factory(seed), X_labeled, y_labeled, X_unlabeled,
        confidence_threshold=CONFIDENCE_THRESHOLD, max_iterations=MAX_ITERATIONS,
        max_pseudo_labels_per_iteration=MAX_PSEUDO_LABELS_PER_ITERATION,
    )
    y_pred_ssl = ssl_result["model"].predict(data["X_test"])
    ssl_metrics = evaluation.classification_metrics(data["y_test"], y_pred_ssl)
    pseudo_label_fraction = ssl_result["pseudo_labels_added_total"] / len(X_unlabeled)
    record_result(
        label_fraction, "ssl", "pseudo_labeling", seed, ssl_metrics,
        pseudo_labels_added=ssl_result["pseudo_labels_added_total"],
        pseudo_label_fraction=round(pseudo_label_fraction, 4),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        notes=f"iterations_run={ssl_result['iterations_run']}",
    )

    print(f"  frac={label_fraction:.0%} seed={seed}  supervised_bal_acc={supervised_metrics['balanced_accuracy']:.4f}  "
          f"ssl_bal_acc={ssl_metrics['balanced_accuracy']:.4f}  gain={ssl_metrics['balanced_accuracy']-supervised_metrics['balanced_accuracy']:+.4f}  "
          f"pseudo_labels={ssl_result['pseudo_labels_added_total']}")

    if save_models and seed == 42:
        utils.ensure_dir(MODELS_DIR)
        joblib.dump(supervised_model, os.path.join(MODELS_DIR, f"supervised_{int(label_fraction*100)}pct.joblib"))
        joblib.dump(ssl_result["model"], os.path.join(MODELS_DIR, f"ssl_{int(label_fraction*100)}pct.joblib"))

    return {
        "supervised_metrics": supervised_metrics, "ssl_metrics": ssl_metrics, "ssl_result": ssl_result,
        "X_labeled": X_labeled, "y_labeled": y_labeled, "X_unlabeled": X_unlabeled, "y_unlabeled_hidden": y_unlabeled_hidden,
        "y_pred_supervised": y_pred_supervised, "y_pred_ssl": y_pred_ssl,
    }


def run_main_experiments_for_fraction(data: dict, label_fraction: float, save_models: bool = True) -> list:
    """Run run_main_experiment() across all 3 seeds for one label fraction."""
    print(f"\n--- Label fraction: {label_fraction:.0%} ---")
    results = []
    for seed in SEEDS:
        result = run_main_experiment(data, label_fraction, seed, save_models=save_models)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Optional second SSL method: Label Spreading (Section 12)
# ---------------------------------------------------------------------------


def run_label_spreading_experiment(data: dict, label_fraction: float, seed: int) -> dict:
    """
    Section 12: Label Spreading - the course-taught (Week12,
    graph-based label propagation family) SSL method chosen back in
    Phase 3 as the "breadth" addition (course_context/
    TEACHER_EXPECTATIONS.md). `sklearn.semi_supervised.LabelSpreading`
    takes ALL rows (labeled + unlabeled) at once, with unlabeled rows
    marked -1 - it builds a similarity graph and propagates labels
    along it, never seeing the true hidden labels of the -1 rows.

    Uses the SAME labeled/unlabeled split as the pseudo-labeling run
    for this (label_fraction, seed), for a fair comparison.
    """
    X_labeled, y_labeled, X_unlabeled, y_unlabeled_hidden = split_labeled_unlabeled(
        data["X_train"], data["y_train"], label_fraction, seed
    )

    class_to_index = {c: i for i, c in enumerate(CLASS_ORDER)}
    X_combined = pd.concat([X_labeled, X_unlabeled])
    y_combined = np.concatenate([
        y_labeled.map(class_to_index).values,
        np.full(len(X_unlabeled), -1),
    ])

    model = LabelSpreading(kernel="knn", n_neighbors=7, max_iter=30)
    model.fit(X_combined.values, y_combined)

    y_pred_indices = model.predict(data["X_test"].values)
    y_pred = np.array([CLASS_ORDER[i] for i in y_pred_indices])
    metrics = evaluation.classification_metrics(data["y_test"], y_pred)

    record_result(
        label_fraction, "ssl", "label_spreading", seed, metrics,
        notes="graph-based SSL (course-taught, Week12) - optional second SSL method",
    )
    print(f"  frac={label_fraction:.0%} seed={seed}  label_spreading_bal_acc={metrics['balanced_accuracy']:.4f}")
    return {"metrics": metrics, "model": model}


# ---------------------------------------------------------------------------
# OFFLINE DIAGNOSTIC ONLY: pseudo-label accuracy (Section 24)
# ---------------------------------------------------------------------------


def offline_diagnostic_pseudo_label_accuracy(experiment_result: dict) -> dict:
    """
    ============================================================
    OFFLINE DIAGNOSTIC ONLY - NOT USED DURING TRAINING OR SELECTION.
    ============================================================
    Uses the TRUE hidden labels of the pseudo-labeled rows - which
    self_train() never saw - purely to measure, after the fact, how
    accurate the pseudo-labels actually were. This number never
    influences the model, the confidence threshold, or any other
    training/selection decision (Section 16, 24) - it's computed here,
    once, after run_main_experiment() has already finished and saved
    its real result.

    Parameters
    ----------
    experiment_result : dict
        The return value of run_main_experiment() for one
        (label_fraction, seed).

    Returns
    -------
    dict: {"n_pseudo_labels", "pseudo_label_accuracy", "confusion_summary"}
    """
    ssl_result = experiment_result["ssl_result"]
    y_unlabeled_hidden = experiment_result["y_unlabeled_hidden"]

    pseudo_indices = ssl_result["pseudo_label_indices"]
    pseudo_predicted = ssl_result["y_labeled_final"].loc[pseudo_indices]
    pseudo_true = y_unlabeled_hidden.loc[pseudo_indices]

    correct = (pseudo_predicted.values == pseudo_true.values)
    accuracy = float(correct.mean()) if len(correct) > 0 else None

    return {
        "n_pseudo_labels": len(pseudo_indices),
        "pseudo_label_accuracy_OFFLINE_DIAGNOSTIC_ONLY": accuracy,
    }


# ---------------------------------------------------------------------------
# Label-efficiency curve, SSL gain, AUC (Sections 21-23)
# ---------------------------------------------------------------------------


def build_label_efficiency_table() -> pd.DataFrame:
    """
    Section 21-23: mean±std of the primary metric (balanced accuracy
    and macro F1) for supervised-only and pseudo-labeling SSL, at each
    label fraction, plus SSL gain and label-efficiency AUC (trapezoidal,
    over x=[0.10, 0.30, 0.50] - NOT extrapolated to the full 0-1 range,
    since those are the only 3 points measured - see
    course_context/PROBLEM4_REPORT.md for why this scope is stated
    explicitly).
    """
    df = pd.read_csv(PROBLEM4_RESULTS_PATH)

    rows = []
    for frac in LABEL_FRACTIONS:
        sup = df[(df.label_fraction == frac) & (df.supervised_or_ssl == "supervised")]
        ssl = df[(df.label_fraction == frac) & (df.supervised_or_ssl == "ssl") & (df.method == "pseudo_labeling")]
        rows.append({
            "label_fraction": frac,
            "supervised_balanced_accuracy_mean": sup["balanced_accuracy"].mean(),
            "supervised_balanced_accuracy_std": sup["balanced_accuracy"].std(),
            "supervised_macro_f1_mean": sup["macro_f1"].mean(),
            "supervised_macro_f1_std": sup["macro_f1"].std(),
            "ssl_balanced_accuracy_mean": ssl["balanced_accuracy"].mean(),
            "ssl_balanced_accuracy_std": ssl["balanced_accuracy"].std(),
            "ssl_macro_f1_mean": ssl["macro_f1"].mean(),
            "ssl_macro_f1_std": ssl["macro_f1"].std(),
            "ssl_gain_balanced_accuracy": ssl["balanced_accuracy"].mean() - sup["balanced_accuracy"].mean(),
            "ssl_gain_macro_f1": ssl["macro_f1"].mean() - sup["macro_f1"].mean(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "problem4_label_efficiency.csv"), index=False)
    return table


def compute_label_efficiency_auc(table: pd.DataFrame, metric: str = "macro_f1") -> dict:
    """
    Section 23: trapezoidal area under the label-efficiency curve for
    both supervised-only and SSL, using x=label_fraction (already in
    0-1 form: 0.10/0.30/0.50) and y=the chosen metric. Only covers
    x in [0.10, 0.50] - not normalized/extrapolated to [0, 1] - see
    course_context/PROBLEM4_REPORT.md.
    """
    x = table["label_fraction"].values
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy renamed trapz -> trapezoid in 2.0+
    supervised_auc = float(trapezoid(table[f"supervised_{metric}_mean"].values, x))
    ssl_auc = float(trapezoid(table[f"ssl_{metric}_mean"].values, x))
    return {"metric": metric, "supervised_auc": supervised_auc, "ssl_auc": ssl_auc, "ssl_better": ssl_auc > supervised_auc}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_label_efficiency_figure(table: pd.DataFrame, metric: str = "macro_f1"):
    """problem4_label_efficiency_curve.png - REQUIRED (Section 21)."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    x = table["label_fraction"] * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, table[f"supervised_{metric}_mean"], yerr=table[f"supervised_{metric}_std"],
                marker="o", label="Supervised-only", capsize=4)
    ax.errorbar(x, table[f"ssl_{metric}_mean"], yerr=table[f"ssl_{metric}_std"],
                marker="s", label="SSL (pseudo-labeling)", capsize=4)
    ax.set_xlabel("Percentage of labeled training data")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Label-efficiency curve: {metric.replace('_', ' ').title()} (mean ± std, 3 seeds)")
    ax.set_xticks([10, 30, 50])
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem4_label_efficiency_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_ssl_gain_figure(table: pd.DataFrame):
    """problem4_ssl_gain.png"""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    fig, ax = plt.subplots(figsize=(7, 5))
    x = (table["label_fraction"] * 100).astype(int).astype(str) + "%"
    ax.bar(x, table["ssl_gain_macro_f1"], color=["tab:red" if v < 0 else "tab:green" for v in table["ssl_gain_macro_f1"]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Labeled fraction")
    ax.set_ylabel("SSL Gain (Macro F1): SSL - Supervised")
    ax.set_title("SSL Gain by label fraction (positive = unlabeled data helped)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem4_ssl_gain.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_class_distribution_figure(data: dict):
    """problem4_class_distribution.png - labeled subset class balance at each fraction, seed=42."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, frac in zip(axes, LABEL_FRACTIONS):
        X_labeled, y_labeled, X_unlabeled, y_unlabeled_hidden = split_labeled_unlabeled(data["X_train"], data["y_train"], frac, seed=42)
        counts = y_labeled.value_counts().reindex(CLASS_ORDER, fill_value=0)
        ax.bar(counts.index, counts.values)
        ax.set_title(f"{int(frac*100)}% labeled (n={len(y_labeled)})")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Class distribution of the labeled subset, by label fraction (seed=42)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "problem4_class_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_pseudolabel_figures(data: dict):
    """problem4_pseudolabel_confidence.png, problem4_pseudolabel_class_distribution.png (seed=42, all 3 fractions)."""
    import matplotlib.pyplot as plt
    visualization.set_plot_style()

    fig_conf, axes_conf = plt.subplots(1, 3, figsize=(13, 4.5))
    fig_class, axes_class = plt.subplots(1, 3, figsize=(13, 4.5))

    for i, frac in enumerate(LABEL_FRACTIONS):
        X_labeled, y_labeled, X_unlabeled, y_unlabeled_hidden = split_labeled_unlabeled(data["X_train"], data["y_train"], frac, seed=42)
        result = pseudo_labeling.self_train(
            lambda: base_model_factory(42), X_labeled, y_labeled, X_unlabeled,
            confidence_threshold=CONFIDENCE_THRESHOLD, max_iterations=MAX_ITERATIONS,
            max_pseudo_labels_per_iteration=MAX_PSEUDO_LABELS_PER_ITERATION,
        )
        confidences = [h["mean_confidence_of_added"] for h in result["pseudo_label_history"] if h["mean_confidence_of_added"] is not None]
        iterations = [h["iteration"] for h in result["pseudo_label_history"] if h["mean_confidence_of_added"] is not None]
        axes_conf[i].bar(iterations, confidences)
        axes_conf[i].set_title(f"{int(frac*100)}% labeled")
        axes_conf[i].set_xlabel("Iteration")
        axes_conf[i].set_ylabel("Mean confidence of added pseudo-labels")
        axes_conf[i].set_ylim(0.85, 1.0)

        pseudo_labels = result["y_labeled_final"].loc[result["pseudo_label_indices"]]
        class_counts = pseudo_labels.value_counts().reindex(CLASS_ORDER, fill_value=0)
        axes_class[i].bar(class_counts.index, class_counts.values)
        axes_class[i].set_title(f"{int(frac*100)}% labeled (n={len(pseudo_labels)} pseudo-labels)")
        axes_class[i].tick_params(axis="x", rotation=15)

    fig_conf.suptitle("Mean confidence of pseudo-labels added per iteration (seed=42)")
    fig_conf.tight_layout()
    path_conf = os.path.join(FIGURES_DIR, "problem4_pseudolabel_confidence.png")
    fig_conf.savefig(path_conf, bbox_inches="tight")
    plt.close(fig_conf)
    print(f"Saved {path_conf}")

    fig_class.suptitle("Class distribution of pseudo-labels added (seed=42) - note the skew toward 'Clear'")
    fig_class.tight_layout()
    path_class = os.path.join(FIGURES_DIR, "problem4_pseudolabel_class_distribution.png")
    fig_class.savefig(path_class, bbox_inches="tight")
    plt.close(fig_class)
    print(f"Saved {path_class}")


def make_confusion_matrices(data: dict):
    """Confusion matrices for 10% and 30% labeled: supervised vs SSL (Section 25)."""
    for frac in [0.10, 0.30]:
        result = run_main_experiment(data, frac, seed=42, save_models=False)
        for method_name, y_pred in [("supervised", result["y_pred_supervised"]), ("ssl", result["y_pred_ssl"])]:
            cm = evaluation.get_confusion_matrix(data["y_test"], y_pred, labels=CLASS_ORDER)
            path = os.path.join(FIGURES_DIR, f"problem4_confusion_matrix_{int(frac*100)}pct_{method_name}.png")
            visualization.plot_confusion_matrix(cm, labels=CLASS_ORDER, save_path=path,
                                                 title=f"{int(frac*100)}% labeled — {method_name} (seed=42)")
            print(f"Saved {path}")
