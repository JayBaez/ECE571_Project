"""
models.py (Problem 5 — Transfer Learning)

The primary transfer-learning model: a small MLP with the exact
architecture requested (Section 12): Input -> Dense(128) -> ReLU ->
Dropout -> Dense(64) -> ReLU -> Dense(1).

Two-stage workflow this class is built for:
    1. PRETRAIN on Davis (fresh weights, normal learning rate).
    2. FINE-TUNE on a small Amherst sample, continuing from the
       pretrained weights (not re-initialized), usually with a
       smaller learning rate - see run_experiments.py.

`freeze_first_layer()` / `unfreeze_all()` support the freezing
ablation (Section 15-16, 34): freezing the first Dense(128) layer
during fine-tuning tests whether the GENERAL weather/irradiance
representation it likely learned on Davis is worth keeping fixed,
while only the later, more city-specific layers adapt to Amherst.
"""

import numpy as np
import torch
import torch.nn as nn


class TransferMLP(nn.Module):
    """
    Dense(128) -> ReLU -> Dropout -> Dense(64) -> ReLU -> Dense(1).

    Deliberately the exact size requested in the Phase 8 instructions,
    not re-derived from Problem 2's smaller MLP (hidden_size=32) -
    transfer learning specifically wants enough capacity in the first
    layer to hold a genuinely reusable weather/irradiance
    representation, which is also why THIS layer (not the second) is
    the one tested for freezing.
    """

    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        x = self.relu2(self.layer2(x))
        return self.output_layer(x).squeeze(-1)

    def freeze_first_layer(self) -> None:
        """
        Freeze layer1's weights (the 128-unit layer closest to the
        input) so fine-tuning only updates layer2 and the output layer
        - tests whether the general weather->representation mapping
        learned on Davis transfers as-is, with only the later,
        city-specific mapping needing to adapt.
        """
        for param in self.layer1.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Make every parameter trainable again - the default state, and what "full fine-tuning" uses."""
        for param in self.parameters():
            param.requires_grad = True


def predict_values(model: TransferMLP, X, device: str) -> np.ndarray:
    """
    Run a trained TransferMLP on X and return predictions as a plain
    numpy array. Same pattern as Problem 2's models.predict_values().

    Parameters
    ----------
    model : TransferMLP
    X : array-like
    device : str

    Returns
    -------
    numpy.ndarray
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.as_tensor(np.asarray(X, dtype=float).copy(), dtype=torch.float32).to(device)
        predictions = model(X_tensor)
    return predictions.cpu().numpy()
