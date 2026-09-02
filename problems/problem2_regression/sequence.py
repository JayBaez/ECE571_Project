"""
sequence.py (Problem 2 — Supervised Regression, sequence sub-task)

Builds K=12 sliding-window sequences and trains a GRU to forecast the
next Output Power reading, per the project spec's sequence-forecasting
setup (course_context/TEACHER_EXPECTATIONS.md, Problem 2): use the
previous 12 steps (~6 hours at 30-minute sampling) to predict the next
one.

WHY GRU, NOT LSTM: both are taught (course_context/COURSE_CONTEXT.md,
Week15) and either would be a reasonable choice. GRU has fewer gates/
parameters than LSTM and trains a bit faster, and this project's
K=12-step windows don't obviously need LSTM's extra machinery for
very-long-range dependencies. A reasonable, explainable pick given the
instruction not to implement every possible architecture.

LEAKAGE PREVENTION (the reason this file is organized the way it is):
  1. Split the raw city data chronologically FIRST (src/splitting.py).
  2. Fit the preprocessor on the TRAINING split only.
  3. Apply it to train and test separately.
  4. Build sliding windows from the train-scaled data and the
     test-scaled data INDEPENDENTLY - no window is ever built using
     rows from both splits. This means the first K rows of the test
     period can't form a window (there's no earlier TEST-period data
     to fill it) - a small, deliberate sacrifice of a handful of test
     predictions, in exchange for a design that's simple to verify:
     "test windows only ever contain test rows" is a much easier
     invariant to check than "test windows may reach back into
     training data, but only in a way that's still fine because...".
"""

import numpy as np
import torch
import torch.nn as nn

K_STEPS = 12  # ~6 hours of history at 30-minute sampling

# Every primary feature PLUS the previous Output Power readings
# themselves. Including Output Power here is legitimate ONLY because
# every value inside a window is strictly earlier than the target
# being predicted (the step right after the window) - see
# course_context/LEAKAGE_MAP.md, Problem 2.
SEQUENCE_TARGET_COLUMN = "Output Power"


def build_sequences(feature_df, target_series, feature_columns: list, k: int = K_STEPS) -> tuple:
    """
    Slide a length-k window across one (already chronologically
    sorted) city split and build training examples: window of k
    consecutive rows -> the target value at the very next row.

    IMPORTANT - two different "Output Power" roles, kept deliberately
    separate: `feature_df` should be the SCALED/preprocessed data (so
    Output Power as a lag feature is on the same scale as every other
    input), but `target_series` should be the RAW, unscaled Output
    Power column (in kW) - so the model's prediction target, and every
    RMSE/MAE computed against it, stays in interpretable physical
    units. `feature_df` and `target_series` must come from the same
    original rows in the same order (e.g. a preprocessed DataFrame and
    the raw DataFrame it was derived from, column-transforms only, no
    row reordering).

    Parameters
    ----------
    feature_df : pandas.DataFrame
        A SINGLE split (train OR test) of one city's data, already
        preprocessed (scaled/encoded) - never pass combined train+test
        data here, and never call this before splitting.
    target_series : pandas.Series
        The RAW (unscaled) Output Power column, same rows/order as
        `feature_df`.
    feature_columns : list of str
        Which columns go into each window - typically
        features.get_feature_columns()'s primary set plus the (scaled)
        "Output Power" column itself as a lag feature.
    k : int
        Window length. Default 12, per the project spec.

    Returns
    -------
    (X, y) : tuple of numpy.ndarray
        X : shape (n_samples, k, n_features)
        y : shape (n_samples,) - the RAW Output Power (kW) value
            immediately after each window.
    """
    feature_array = feature_df[feature_columns].values.astype(np.float32)
    target_array = target_series.values.astype(np.float32)

    n_rows = len(feature_df)
    X, y = [], []
    for i in range(k - 1, n_rows - 1):  # window covers rows [i-k+1, i], target is row i+1
        X.append(feature_array[i - k + 1: i + 1])
        y.append(target_array[i + 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def persistence_baseline_predictions(target_series, k: int = K_STEPS) -> np.ndarray:
    """
    Section 29's "simple sequence baseline": predict the next Output
    Power reading as just equal to the MOST RECENT observed reading
    (the last row of the window) - i.e. "assume nothing changes in the
    next 30 minutes." This is the standard sanity-check baseline for
    any forecasting model: if the GRU can't beat this, it isn't
    learning anything the trivial "no change" guess didn't already
    capture.

    Aligned with build_sequences()'s y array - call this on the SAME
    (raw, unscaled) target_series and k used there.

    Parameters
    ----------
    target_series : pandas.Series
        RAW (unscaled) Output Power column.
    k : int

    Returns
    -------
    numpy.ndarray
        shape (n_samples,), same length/order as build_sequences()'s y.
    """
    target_array = target_series.values.astype(np.float32)
    n_rows = len(target_array)
    return np.array([target_array[i] for i in range(k - 1, n_rows - 1)], dtype=np.float32)


class SimpleGRURegressor(nn.Module):
    """
    A small GRU-based sequence regressor: GRU -> final hidden state ->
    Dense -> single Output Power prediction.

    Deliberately simple: one (or a couple, if tuned) GRU layer(s), no
    attention, no bidirectionality - a small architecture consistent
    with the "do not run an enormous architecture" instruction and the
    project owner's stated background.
    """

    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, final_hidden = self.gru(x)
        last_layer_hidden = final_hidden[-1]  # (batch, hidden_size) - top layer's final hidden state
        return self.output_layer(last_layer_hidden).squeeze(-1)


def predict_sequence_values(model: SimpleGRURegressor, X, device: str) -> np.ndarray:
    """
    Run a trained SimpleGRURegressor on a batch of windows and return
    predictions as a plain numpy array.

    Parameters
    ----------
    model : SimpleGRURegressor
    X : array-like, shape (n_samples, k, n_features)
    device : str

    Returns
    -------
    numpy.ndarray
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(device)
        predictions = model(X_tensor)
    return predictions.cpu().numpy()
