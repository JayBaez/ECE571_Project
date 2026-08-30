"""
models.py (Problem 1 — Supervised Classification)

Two kinds of models used for Problem 1:

1. Classical models (get_classical_models()): plain scikit-learn
   estimators. No wrapper needed - they all share .fit(X, y)/.predict(X)
   for free (see src/torch_utils.py's module docstring for why the
   framework doesn't build a custom "Model" class at all).

2. SimpleMLPClassifier: a small feed-forward neural network, trained
   with src/torch_utils.py's generic train_torch_model() (built in
   Phase 2, works with any nn.Module).
"""

import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Models that accept a `class_weight` constructor argument directly.
# GradientBoostingClassifier does NOT support class_weight in
# scikit-learn (it would need `sample_weight` passed to .fit() instead)
# - to keep this understandable, class weighting is only tested for
# the three models that support it the simple way. This is a
# deliberate scope decision, not an oversight - see
# course_context/PROBLEM1_REPORT.md.
CLASS_WEIGHT_CAPABLE_MODELS = ["logistic_regression", "decision_tree", "random_forest"]


def get_classical_models(seed: int, class_weight: str = None, rf_params: dict = None, gb_params: dict = None) -> dict:
    """
    Build the classical model progression: majority-class baseline ->
    Logistic Regression -> Decision Tree -> Random Forest -> Gradient
    Boosting.

    Parameters
    ----------
    seed : int
        Passed as `random_state` to every model that accepts one.
    class_weight : str, optional
        e.g. "balanced". Only applied to models in
        CLASS_WEIGHT_CAPABLE_MODELS - see the module docstring.
    rf_params, gb_params : dict, optional
        Extra keyword arguments for RandomForestClassifier /
        GradientBoostingClassifier (used to plug in tuned
        hyperparameters after the small search in
        run_hyperparameter_search.py).

    Returns
    -------
    dict
        {model_name: unfitted scikit-learn estimator}
    """
    rf_params = rf_params or {}
    gb_params = gb_params or {}

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=seed,
            class_weight=class_weight if "logistic_regression" in CLASS_WEIGHT_CAPABLE_MODELS else None,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=seed,
            class_weight=class_weight if "decision_tree" in CLASS_WEIGHT_CAPABLE_MODELS else None,
        ),
        "random_forest": RandomForestClassifier(
            random_state=seed, n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", None),
            min_samples_leaf=rf_params.get("min_samples_leaf", 1),
            class_weight=class_weight if "random_forest" in CLASS_WEIGHT_CAPABLE_MODELS else None,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=seed, n_estimators=gb_params.get("n_estimators", 100),
            learning_rate=gb_params.get("learning_rate", 0.1),
            max_depth=gb_params.get("max_depth", 3),
        ),
    }
    return models


class SimpleMLPClassifier(nn.Module):
    """
    A small feed-forward classifier: Input -> Dense -> ReLU -> Dropout
    -> Dense -> ReLU -> Output (raw class scores / logits).

    Deliberately simple per the project owner's stated background -
    two hidden layers of the same size, one dropout layer. No
    batch-norm, no residual connections, no attention - none of that
    is justified for a small tabular 3-class problem like this one.

    Outputs raw logits (not probabilities) - use with
    `nn.CrossEntropyLoss`, which applies softmax internally. To get
    predicted class labels, take `argmax` of the output; to get
    probabilities, apply `softmax` separately (not needed for this
    project's metrics, which only need predicted class labels).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def predict_classes(model: SimpleMLPClassifier, X, device: str) -> torch.Tensor:
    """
    Run a trained SimpleMLPClassifier on X and return predicted class
    indices (not logits/probabilities) - the form
    evaluation.classification_metrics() expects.

    Parameters
    ----------
    model : SimpleMLPClassifier
        Already trained (e.g. via torch_utils.train_torch_model()).
    X : array-like
    device : str
        "cuda" or "cpu" - see src/utils.py's get_device().

    Returns
    -------
    numpy.ndarray
        1-D array of predicted class indices.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        predictions = torch.argmax(logits, dim=1)
    return predictions.cpu().numpy()
