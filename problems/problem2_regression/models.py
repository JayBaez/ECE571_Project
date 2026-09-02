"""
models.py (Problem 2 — Supervised Regression)

Classical models: plain scikit-learn regressors, no wrapper needed
(same reasoning as Problem 1's models.py and src/torch_utils.py's
module docstring - they already share .fit(X, y)/.predict(X)).

SimpleMLPRegressor: a small feed-forward network, trained with
src/torch_utils.py's generic train_torch_model().
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


def get_classical_models(seed: int, ridge_alpha: float = 1.0, rf_params: dict = None, gb_params: dict = None) -> dict:
    """
    Build the classical model progression: mean baseline -> linear
    regression -> ridge -> decision tree -> random forest -> gradient
    boosting.

    Parameters
    ----------
    seed : int
        Passed as `random_state` to every model that accepts one.
    ridge_alpha : float
        Ridge's regularization strength.
    rf_params, gb_params : dict, optional
        Extra keyword arguments for RandomForestRegressor /
        GradientBoostingRegressor (tuned hyperparameters).

    Returns
    -------
    dict
        {model_name: unfitted scikit-learn estimator}
    """
    rf_params = rf_params or {}
    gb_params = gb_params or {}

    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=ridge_alpha, random_state=seed),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(
            random_state=seed, n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", None),
            min_samples_leaf=rf_params.get("min_samples_leaf", 1),
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=seed, n_estimators=gb_params.get("n_estimators", 100),
            learning_rate=gb_params.get("learning_rate", 0.1),
            max_depth=gb_params.get("max_depth", 3),
        ),
    }


class SimpleMLPRegressor(nn.Module):
    """
    A small feed-forward regressor: Input -> Dense -> ReLU -> Dense ->
    ReLU -> Output (single continuous value).

    Same architectural philosophy as Problem 1's SimpleMLPClassifier -
    two hidden layers, nothing fancier. The only real difference is
    the output: one number (predicted Output Power), not class scores,
    and no Dropout by default (regression targets here are less prone
    to the kind of severe class-imbalance overfitting Dropout helps
    with in Problem 1 - Dropout is still available via `dropout > 0`
    if a later experiment wants it).
    """

    def __init__(self, input_dim: int, hidden_size: int = 32, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch, 1) -> (batch,), matches y's shape


def predict_values(model: SimpleMLPRegressor, X, device: str):
    """
    Run a trained SimpleMLPRegressor on X and return predictions as a
    plain numpy array.

    Parameters
    ----------
    model : SimpleMLPRegressor
    X : array-like
    device : str
        "cuda" or "cpu" - see src/utils.py's get_device().

    Returns
    -------
    numpy.ndarray
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float32).to(device)
        predictions = model(X_tensor)
    return predictions.cpu().numpy()
