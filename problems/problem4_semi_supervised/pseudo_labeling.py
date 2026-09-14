"""
pseudo_labeling.py (Problem 4 — Semi-Supervised Learning)

Implements pseudo-labeling / self-training, the primary SSL method for
this problem:

    1. Train a supervised classifier on the small labeled subset.
    2. Predict class probabilities for the unlabeled pool.
    3. Keep only predictions at or above a confidence threshold.
    4. Add those rows (with their PREDICTED label, never a true one)
       to the labeled set.
    5. Remove them from the unlabeled pool.
    6. Retrain on the enlarged labeled set.
    7. Repeat until no more confident predictions are found, a
       maximum-iteration limit is hit, or the unlabeled pool is empty.

The classifier NEVER sees an unlabeled row's true label - only its X
values, and (once pseudo-labeled) the model's OWN predicted label.
True labels of "unlabeled" rows exist in this dataset only because we
started from a fully-labeled dataset and hid them on purpose - see
course_context/PROBLEM4_REPORT.md, Section 6, for how they're used
ONLY for offline diagnostic analysis after training, never during it.
"""

import numpy as np
import pandas as pd


def self_train(
    base_model_factory,
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_unlabeled: pd.DataFrame,
    confidence_threshold: float = 0.90,
    max_iterations: int = 5,
    max_pseudo_labels_per_iteration: int = None,
    verbose: bool = False,
) -> dict:
    """
    Run pseudo-labeling / self-training.

    Parameters
    ----------
    base_model_factory : callable
        `() -> unfitted scikit-learn classifier`, e.g.
        `lambda: LogisticRegression(class_weight="balanced")` - a
        factory (not a single shared instance) so every retrain starts
        from a clean, unfitted model.
    X_labeled, y_labeled : pandas.DataFrame / pandas.Series
        The starting labeled data.
    X_unlabeled : pandas.DataFrame
        The pool of unlabeled X - its true y is intentionally never
        passed to this function.
    confidence_threshold : float
        Minimum predicted probability (of the predicted class) to
        accept a pseudo-label.
    max_iterations : int
        Upper bound on self-training rounds - a safeguard against
        unbounded looping (Section 11).
    max_pseudo_labels_per_iteration : int, optional
        Safety cap: if more than this many rows clear the confidence
        threshold in one iteration, only the MOST confident ones (up
        to the cap) are added - prevents one noisy iteration from
        dumping a huge, possibly-biased batch into the labeled set at
        once (Section 11's "maximum pseudo-labels per iteration"
        safeguard).
    verbose : bool

    Returns
    -------
    dict with keys:
        model : the final fitted classifier (retrained once more after
            the loop ends, on the fully-grown labeled set).
        iterations_run : int
        pseudo_labels_added_total : int
        pseudo_label_history : list of dict, one entry per iteration -
            {"iteration", "n_added", "n_remaining_unlabeled",
             "mean_confidence_of_added", "class_distribution_added"}
        pseudo_label_indices : the original X_unlabeled index values
            that ended up pseudo-labeled (for offline diagnostic
            analysis - see course_context/PROBLEM4_REPORT.md).
        X_labeled_final, y_labeled_final : the grown labeled set the
            final model was trained on.
    """
    X_labeled_current = X_labeled.copy()
    y_labeled_current = y_labeled.copy()
    X_unlabeled_remaining = X_unlabeled.copy()

    history = []
    model = base_model_factory()
    model.fit(X_labeled_current, y_labeled_current)

    for iteration in range(1, max_iterations + 1):
        if len(X_unlabeled_remaining) == 0:
            break

        probabilities = model.predict_proba(X_unlabeled_remaining)
        max_confidence = probabilities.max(axis=1)
        predicted_classes = model.classes_[probabilities.argmax(axis=1)]

        confident_mask = max_confidence >= confidence_threshold
        n_confident = int(confident_mask.sum())

        if n_confident == 0:
            history.append({
                "iteration": iteration, "n_added": 0,
                "n_remaining_unlabeled": len(X_unlabeled_remaining),
                "mean_confidence_of_added": None, "class_distribution_added": {},
            })
            if verbose:
                print(f"  iter {iteration}: no predictions cleared the {confidence_threshold} threshold - stopping")
            break

        confident_indices = X_unlabeled_remaining.index[confident_mask]
        confident_conf = max_confidence[confident_mask]
        confident_labels = predicted_classes[confident_mask]

        # Safety cap: keep only the most-confident ones if there are too many.
        if max_pseudo_labels_per_iteration is not None and n_confident > max_pseudo_labels_per_iteration:
            top_order = np.argsort(-confident_conf)[:max_pseudo_labels_per_iteration]
            confident_indices = confident_indices[top_order]
            confident_conf = confident_conf[top_order]
            confident_labels = confident_labels[top_order]

        confident_X = X_unlabeled_remaining.loc[confident_indices]
        confident_y = pd.Series(confident_labels, index=confident_indices)
        class_distribution = confident_y.value_counts().to_dict()

        history.append({
            "iteration": iteration, "n_added": len(confident_indices),
            "n_remaining_unlabeled": len(X_unlabeled_remaining) - len(confident_indices),
            "mean_confidence_of_added": float(np.mean(confident_conf)),
            "class_distribution_added": class_distribution,
        })

        X_labeled_current = pd.concat([X_labeled_current, confident_X])
        y_labeled_current = pd.concat([y_labeled_current, confident_y])
        X_unlabeled_remaining = X_unlabeled_remaining.drop(index=confident_indices)

        if verbose:
            print(f"  iter {iteration}: added {len(confident_indices)} pseudo-labels "
                  f"(mean confidence={np.mean(confident_conf):.3f}), {len(X_unlabeled_remaining)} unlabeled remain, "
                  f"class distribution: {class_distribution}")

        model = base_model_factory()
        model.fit(X_labeled_current, y_labeled_current)

    pseudo_label_indices = X_labeled_current.index[len(X_labeled):]
    return {
        "model": model,
        "iterations_run": len(history),
        "pseudo_labels_added_total": len(X_labeled_current) - len(X_labeled),
        "pseudo_label_history": history,
        "pseudo_label_indices": pseudo_label_indices,
        "X_labeled_final": X_labeled_current,
        "y_labeled_final": y_labeled_current,
    }


def select_confidence_threshold(
    base_model_factory,
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_unlabeled: pd.DataFrame,
    candidate_thresholds: list,
    val_frac: float = 0.2,
    max_iterations: int = 5,
) -> tuple:
    """
    Choose a confidence threshold using an inner VALIDATION split of
    the labeled data only - never the real test set (Section 10, 28).

    The labeled subset is split further: an inner-train portion
    (used to run self_train) and an inner-validation portion (used
    only to SCORE each candidate threshold). This keeps the choice of
    threshold honest - it's picked the same way any other
    hyperparameter in this project is picked (see
    course_context/PROBLEM2_REPORT.md's chronological inner-validation
    precedent), just using a random split here since the labeled
    subset itself is already a random sample of the training pool
    (not a further chronologically-ordered sequence).

    Parameters
    ----------
    base_model_factory : callable
    X_labeled, y_labeled : the labeled subset (one label fraction).
    X_unlabeled : the unlabeled pool for that same label fraction.
    candidate_thresholds : list of float, e.g. [0.80, 0.90, 0.95].
    val_frac : float
        Fraction of X_labeled held out as inner-validation.
    max_iterations : int
        Passed through to self_train() for each candidate.

    Returns
    -------
    (best_threshold, scores) : tuple
        best_threshold : float
        scores : dict {threshold: validation balanced_accuracy}
    """
    from sklearn.model_selection import train_test_split

    from src.evaluation import classification_metrics

    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
        X_labeled, y_labeled, test_size=val_frac, random_state=42, stratify=y_labeled
    )

    scores = {}
    for threshold in candidate_thresholds:
        result = self_train(
            base_model_factory, X_inner_train, y_inner_train, X_unlabeled,
            confidence_threshold=threshold, max_iterations=max_iterations,
        )
        y_val_pred = result["model"].predict(X_inner_val)
        metrics = classification_metrics(y_inner_val, y_val_pred)
        scores[threshold] = metrics["balanced_accuracy"]

    best_threshold = max(scores, key=scores.get)
    return best_threshold, scores
