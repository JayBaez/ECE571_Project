"""
visualization.py

Reusable plotting functions. Every function here SAVES its figure to
a file (instead of only showing it on screen) and closes the figure
afterward, so the framework can run unattended and figures aren't lost
when running from a script.

Design note: rather than writing a separate function for every plot
type mentioned in the project spec (learning curves, explained
variance curves, label-efficiency curves, ...), most of those share
the same shape - "one line, x vs y, saved to a file" - so they all use
the single generic plot_metric_curve() function below. Only
confusion matrices and prediction-vs-truth plots get their own
dedicated functions, since those have plot-specific logic (e.g. a
confusion matrix needs a heatmap and axis labels made from class
names, not just a line).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import ensure_dir

_STYLE_APPLIED = False


def set_plot_style() -> None:
    """
    Apply one consistent visual style to all figures in this project.
    Safe to call multiple times - only applies the style once.
    """
    global _STYLE_APPLIED
    if not _STYLE_APPLIED:
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams["figure.dpi"] = 100
        _STYLE_APPLIED = True


def build_figure_path(problem: str, experiment_id: str, filename: str, base_dir: str = "figures") -> str:
    """
    Build a save path for a figure that (a) lives in the right
    per-problem folder and (b) includes the experiment ID in the
    filename, so re-running an experiment never silently overwrites a
    previous run's figure.

    Parameters
    ----------
    problem : str
        e.g. "problem2", or "framework_demo" for the Phase 2 demo.
    experiment_id : str
        From experiment_runner.generate_experiment_id().
    filename : str
        A short description of the figure, e.g. "confusion_matrix.png".
    base_dir : str
        Root figures folder. Default "figures".

    Returns
    -------
    str
        e.g. "figures/problem2/P2_Davis_random_forest_seed42_.../confusion_matrix.png"
        The directory is created if it doesn't exist yet.
    """
    directory = ensure_dir(os.path.join(base_dir, problem, experiment_id))
    return os.path.join(directory, filename)


def plot_confusion_matrix(cm: np.ndarray, labels: list, save_path: str, title: str = "Confusion Matrix") -> str:
    """
    Plot and save a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : numpy.ndarray
        Confusion matrix from evaluation.get_confusion_matrix().
    labels : list of str
        Class names, in the same order used to compute `cm`.
    save_path : str
        Where to save the figure, e.g. "figures/problem1/confusion_matrix.png".
    title : str

    Returns
    -------
    str
        The save_path, for convenience.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    if os.path.dirname(save_path):
        ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_prediction_vs_truth(y_true, y_pred, save_path: str, title: str = "Prediction vs. Truth") -> str:
    """
    Scatter plot of predicted vs. true values for a regression task,
    with a diagonal reference line (perfect predictions would sit
    exactly on this line).

    Parameters
    ----------
    y_true, y_pred : array-like
    save_path : str
    title : str

    Returns
    -------
    str
        The save_path, for convenience.
    """
    set_plot_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15)

    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())
    ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="Perfect prediction")

    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if os.path.dirname(save_path):
        ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_residuals(y_true, y_pred, save_path: str, title: str = "Residuals") -> str:
    """
    Scatter plot of residuals (true - predicted) against the true
    value, to check whether a regression model's errors are random
    or show a pattern (e.g. consistently under-predicting high
    Output Power values).

    Parameters
    ----------
    y_true, y_pred : array-like
    save_path : str
    title : str

    Returns
    -------
    str
        The save_path, for convenience.
    """
    set_plot_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, residuals, alpha=0.4, s=15)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("True value")
    ax.set_ylabel("Residual (true - predicted)")
    ax.set_title(title)
    fig.tight_layout()
    if os.path.dirname(save_path):
        ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_metric_curve(
    x: list,
    y: list,
    save_path: str,
    xlabel: str = "x",
    ylabel: str = "metric",
    title: str = "",
    series_label: str = None,
) -> str:
    """
    Generic "metric vs. something" line plot. This is the shared
    foundation for several plot types the project will need later:
    learning curves (metric vs. training set size), explained-variance
    curves (variance vs. number of PCA components), and
    label-efficiency curves (metric vs. % labeled data) all have the
    same basic shape, so they can all call this one function instead
    of needing near-duplicate dedicated functions.

    Parameters
    ----------
    x, y : list of float
        Values to plot. Must be the same length.
    save_path : str
    xlabel, ylabel, title : str
    series_label : str, optional
        Legend label, useful when this function is called multiple
        times on the same axes to compare methods (e.g. "SSL" vs.
        "Supervised-only" on a label-efficiency curve) - see the
        `ax` parameter pattern below for that use case.

    Returns
    -------
    str
        The save_path, for convenience.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, marker="o", label=series_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if series_label:
        ax.legend()
    fig.tight_layout()
    if os.path.dirname(save_path):
        ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
