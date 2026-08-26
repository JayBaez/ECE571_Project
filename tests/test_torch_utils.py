"""
Tests for src/torch_utils.py, using a tiny synthetic dataset and a
trivial one-layer model - the point is to prove the training loop
mechanics work (batching, early stopping, checkpointing), not to
train anything meaningful.
"""

import numpy as np
import torch
import torch.nn as nn

from src import torch_utils, utils


def _tiny_linear_problem(n_samples=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 3)).astype(np.float32)
    true_weights = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    y = (X @ true_weights + rng.normal(scale=0.01, size=n_samples)).astype(np.float32).reshape(-1, 1)
    return X, y


def test_make_dataloader_produces_correct_batch_shapes():
    X, y = _tiny_linear_problem(n_samples=20)
    loader = torch_utils.make_dataloader(X, y, batch_size=4, shuffle=False)
    first_batch_X, first_batch_y = next(iter(loader))
    assert first_batch_X.shape == (4, 3)
    assert first_batch_y.shape == (4, 1)


def test_train_torch_model_reduces_loss_on_a_trivial_problem(tmp_path):
    utils.set_seed(42)
    X, y = _tiny_linear_problem(n_samples=200)

    train_loader = torch_utils.make_dataloader(X[:160], y[:160], batch_size=16, shuffle=True)
    val_loader = torch_utils.make_dataloader(X[160:], y[160:], batch_size=16, shuffle=False)

    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    history = torch_utils.train_torch_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device="cpu",
        max_epochs=30,
        patience=10,
        checkpoint_path=str(tmp_path / "model.pt"),
        verbose=False,
    )

    assert len(history["epoch"]) > 0
    assert history["val_loss"][-1] < history["val_loss"][0]  # loss should have decreased
    assert (tmp_path / "model.pt").exists()


def test_train_torch_model_early_stopping_triggers(tmp_path):
    # A model that CANNOT improve (frozen weights, zero learning rate)
    # should trigger early stopping quickly rather than running all
    # max_epochs.
    utils.set_seed(0)
    X, y = _tiny_linear_problem(n_samples=40)
    train_loader = torch_utils.make_dataloader(X, y, batch_size=8, shuffle=True)
    val_loader = torch_utils.make_dataloader(X, y, batch_size=8, shuffle=False)

    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 -> weights never change
    loss_fn = nn.MSELoss()

    history = torch_utils.train_torch_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device="cpu",
        max_epochs=50,
        patience=3,
        verbose=False,
    )

    assert len(history["epoch"]) <= 6  # 1 improving epoch + 3 patience + a little slack, well under 50
