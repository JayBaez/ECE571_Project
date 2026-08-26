"""
torch_utils.py

Generic PyTorch training infrastructure that any future neural network
(MLP, LSTM, GRU, autoencoder - see course_context/ML_METHOD_MAP.md) can
reuse, without this file needing to know anything about what those
models look like.

WHY THERE'S NO "MODEL INTERFACE" CLASS HERE: classical models
(scikit-learn's LinearRegression, RandomForestClassifier, etc.) already
share a common `.fit(X, y)` / `.predict(X)` interface for free - there
is nothing to wrap. PyTorch's `nn.Module` doesn't give you training for
free the way scikit-learn's `.fit()` does, so THIS file's job is to
provide that missing piece (a training loop, early stopping,
checkpointing) generically, for ANY `nn.Module` a future problem phase
defines - not to invent a new "Model" class hierarchy on top of
PyTorch's own.

This module works even if a specific model (LSTM, autoencoder, ...)
doesn't exist yet - see the framework demo (scripts/framework_demo.py)
for it being exercised with a tiny throwaway model.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dataloader(X, y, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    """
    Turn feature/target arrays into a PyTorch DataLoader.

    IMPORTANT - `shuffle` here means something different from the
    "never shuffle chronological data" rule elsewhere in this project:
    this shuffles the ORDER MINI-BATCHES ARE DRAWN IN during training,
    from data that is ALREADY correctly and chronologically split into
    train/test. Shuffling mini-batch order within an already-correct
    training set is standard, expected practice for stochastic
    gradient descent (course_context/COURSE_CONTEXT.md, Week03b) and
    does NOT leak test data or future information - it's not the same
    thing as shuffling before the train/test split itself. In this
    framework: use `shuffle=True` for a training DataLoader,
    `shuffle=False` for validation/test DataLoaders (there's no need
    to shuffle when you're just measuring performance, not learning
    from gradients).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    batch_size : int
    shuffle : bool
        See the note above.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_torch_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    loss_fn,
    device: str,
    max_epochs: int = 100,
    patience: int = 10,
    checkpoint_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Train any PyTorch model (any `nn.Module`) with early stopping and
    optional checkpointing.

    ML concept - early stopping: rather than training for a fixed
    number of epochs regardless of whether the model is still
    improving, this watches the validation loss and stops once it
    hasn't improved for `patience` epochs in a row - a simple way to
    avoid overfitting without needing to guess the "right" number of
    epochs in advance (relates to course_context/COURSE_CONTEXT.md,
    Week11 regularization / early stopping).

    Checkpointing: whenever validation loss reaches a new best value,
    the model's weights are saved to `checkpoint_path` (if given). At
    the end of training, the BEST weights (not necessarily the last
    epoch's weights) are loaded back into `model` before returning -
    so the returned model is always the best one seen, even if
    training continued a bit past that point before early-stopping
    triggered.

    Parameters
    ----------
    model : torch.nn.Module
        Any PyTorch model - this function doesn't need to know its
        architecture.
    train_loader, val_loader : torch.utils.data.DataLoader
        From make_dataloader(). `train_loader` should have
        `shuffle=True`, `val_loader` should have `shuffle=False`.
    optimizer : torch.optim.Optimizer
        e.g. torch.optim.Adam(model.parameters(), lr=1e-3).
    loss_fn : callable
        e.g. torch.nn.MSELoss() for regression, torch.nn.CrossEntropyLoss()
        for classification.
    device : str
        "cuda" or "cpu" - see utils.get_device().
    max_epochs : int
        Upper limit on training epochs.
    patience : int
        Stop if validation loss hasn't improved for this many epochs
        in a row.
    checkpoint_path : str, optional
        Where to save the best model's weights (via
        `torch.save(model.state_dict(), checkpoint_path)`). If None,
        no checkpoint file is written, but the best weights are still
        restored into `model` in memory before returning.
    verbose : bool
        If True (default), print one line per epoch, e.g.
        "Epoch 10/100  Train Loss: 0.1234  Val Loss: 0.1456".

    Returns
    -------
    dict
        {"epoch": [...], "train_loss": [...], "val_loss": [...]} -
        one entry per epoch actually run (may be less than max_epochs
        if early stopping triggered). Save this with
        experiment_runner.save_training_log().
    """
    model = model.to(device)
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                val_losses.append(loss_fn(predictions, y_batch).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"Epoch {epoch}/{max_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history
