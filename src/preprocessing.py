"""
preprocessing.py

Reusable helpers for cleaning data and preparing it for a model:
identifying column types, handling missing values, scaling numeric
columns, encoding categorical columns, separating features from the
target, and (optionally) scaling the target itself for cross-city work.

KEY DESIGN RULE (this is the most important thing in this file):
Every "fit" function takes ONLY a training DataFrame. Every "apply"
function takes an already-fitted scaler/encoder and a DataFrame to
transform. There is no function that fits and transforms the full
dataset in one step. This is intentional: it makes it structurally
awkward to accidentally fit a scaler on test data, which would leak
information from the test set into training (see
course_context/EXPERIMENT_PLAN.md, "Major risks" sections, and
course_context/TEACHER_EXPECTATIONS.md, "no test-set leakage").

Correct usage pattern:
    scaler = fit_scaler(train_df, columns)
    train_scaled = apply_scaler(train_df, scaler, columns)
    test_scaled = apply_scaler(test_df, scaler, columns)   # same scaler!

Why plain functions instead of scikit-learn's Pipeline/ColumnTransformer:
those are good, standard tools, but they hide the fit/transform steps
inside an object you configure once and call `.fit_transform()` on -
which makes it easy to explain WHAT they do but harder to see WHEN
each step happens. Since the #1 goal here is making leakage
structurally hard to introduce by accident, explicit fit_X()/apply_X()
function pairs make each step's timing (train-only fit, then apply to
both) visible in the calling code itself.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that are categorical codes, not real numbers, based on the
# Phase 0 dataset inspection (course_context/DATASET_PROFILE.md).
# "Cloud Type" is an NSRDB category code (0=Clear, 1=Probably Clear,
# etc.) - averaging or scaling it like a number would be meaningless.
DEFAULT_CATEGORICAL_COLUMNS = ["Cloud Type"]

# Columns that identify a row in time rather than describing weather -
# usually excluded from "numeric features to scale" even though they
# are technically numbers, because they're handled separately by
# feature_engineering.py (e.g. turned into sin/cos time features).
TIME_COLUMNS = ["Year", "Month", "Day", "Hour", "Minute"]


# ---------------------------------------------------------------------------
# Column identification
# ---------------------------------------------------------------------------


def get_numeric_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Return the names of numeric columns in `df`, excluding known
    categorical columns, time columns, and anything in `exclude`.

    Parameters
    ----------
    df : pandas.DataFrame
    exclude : list of str, optional
        Extra columns to exclude (e.g. the target column - you almost
        never want to scale the thing you're predicting the same way
        as the inputs).

    Returns
    -------
    list of str
    """
    exclude = set(exclude or [])
    exclude |= set(DEFAULT_CATEGORICAL_COLUMNS) | set(TIME_COLUMNS)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def get_categorical_columns(df: pd.DataFrame, categorical_cols: list = None) -> list:
    """
    Return the names of categorical columns present in `df`.

    Parameters
    ----------
    df : pandas.DataFrame
    categorical_cols : list of str, optional
        Override the default list (DEFAULT_CATEGORICAL_COLUMNS) if a
        particular problem needs a different set.

    Returns
    -------
    list of str
        Only columns that both are in the categorical list AND
        actually exist in `df`.
    """
    candidates = categorical_cols if categorical_cols is not None else DEFAULT_CATEGORICAL_COLUMNS
    return [c for c in candidates if c in df.columns]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Count missing (NaN) values per column.

    Returns
    -------
    pandas.Series
        Column name -> number of missing values, sorted descending.
        Columns with zero missing values are still included.
    """
    return df.isna().sum().sort_values(ascending=False)


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "drop", columns: list = None
) -> pd.DataFrame:
    """
    Handle missing values in `df`. Returns a NEW DataFrame - the
    original is never modified in place, so it's always safe to
    inspect what changed.

    Parameters
    ----------
    df : pandas.DataFrame
    strategy : str
        "drop" - remove rows that have a missing value in any of
            `columns` (or any column, if `columns` is None).
        "interpolate" - fill missing values using linear
            interpolation. Only sensible for numeric, time-ordered
            columns (e.g. filling a few missing Output Power readings
            using their neighbors) - make sure `df` is already sorted
            chronologically before using this strategy.
    columns : list of str, optional
        Which columns to check/fill. Defaults to all columns.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with missing values handled.

    Raises
    ------
    ValueError
        If `strategy` is not "drop" or "interpolate".
    """
    df = df.copy()
    cols = columns if columns is not None else df.columns.tolist()

    if strategy == "drop":
        return df.dropna(subset=cols)
    elif strategy == "interpolate":
        df[cols] = df[cols].interpolate(method="linear", limit_direction="both")
        return df
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use 'drop' or 'interpolate'."
        )


# ---------------------------------------------------------------------------
# Feature / target separation
# ---------------------------------------------------------------------------


def select_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Select an explicit set of columns. This is a trivial function, but
    it exists so "which columns did we actually use" is always a
    single, visible, loggable step rather than something buried inside
    a bigger function.

    Parameters
    ----------
    df : pandas.DataFrame
    columns : list of str

    Returns
    -------
    pandas.DataFrame
        A copy of `df` containing only `columns`, in that order.
    """
    return df[columns].copy()


def prepare_xy(df: pd.DataFrame, target_column: str, exclude_columns: list = None) -> tuple:
    """
    Split a DataFrame into X (features) and y (target), and make the
    excluded columns explicit.

    ML concept / leakage prevention: this is the single place where
    "what is a feature and what is the answer" gets decided. Passing
    `exclude_columns` explicitly (e.g. excluding "GHI" and
    "Clearsky GHI" when the target is the sky-condition label derived
    from them - see course_context/TEACHER_EXPECTATIONS.md, Problem 1)
    makes leakage a visible, deliberate choice instead of an accident.

    Parameters
    ----------
    df : pandas.DataFrame
    target_column : str
        The column to predict, e.g. "Output Power" or "Sky_Condition".
    exclude_columns : list of str, optional
        Extra columns to leave out of X entirely (not used as
        features) - e.g. leakage-risk columns, or identifier columns
        that shouldn't be fed to a model. `target_column` is always
        excluded automatically, so it doesn't need to be repeated here.

    Returns
    -------
    (X, y, feature_columns) : tuple
        X : pandas.DataFrame - the feature columns.
        y : pandas.Series - the target column.
        feature_columns : list of str - exactly which columns ended up
            in X, so this can be logged/saved alongside an experiment's
            results for reproducibility.
    """
    exclude = set(exclude_columns or [])
    exclude.add(target_column)

    feature_columns = [c for c in df.columns if c not in exclude]
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y, feature_columns


# ---------------------------------------------------------------------------
# Numeric scaling
# ---------------------------------------------------------------------------


def fit_scaler(train_df: pd.DataFrame, columns: list) -> StandardScaler:
    """
    Fit a StandardScaler (subtract mean, divide by standard deviation)
    on TRAINING data only.

    ML concept: many models (e.g. linear regression with gradient
    descent, MLPs, kNN) work better and train faster when input
    features are on a similar scale - this was explicitly taught in
    course_context/COURSE_CONTEXT.md (Week03b: "inputs should always
    be normalized"). The scaler must be fit only on training data,
    otherwise information about the test set's distribution leaks into
    training.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data ONLY. Never pass test or combined data here.
    columns : list of str
        Numeric columns to scale.

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        A fitted scaler. Use apply_scaler() to transform any DataFrame
        (train or test) with it.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[columns])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, columns: list) -> pd.DataFrame:
    """
    Apply an already-fitted scaler to `df`. Safe to call on train,
    validation, or test data - the scaler itself doesn't change.

    Parameters
    ----------
    df : pandas.DataFrame
    scaler : StandardScaler
        A scaler previously returned by fit_scaler().
    columns : list of str
        Must be the same columns the scaler was fit on.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with `columns` replaced by their scaled values.
    """
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def fit_encoder(train_df: pd.DataFrame, columns: list) -> OneHotEncoder:
    """
    Fit a one-hot encoder for categorical columns (e.g. "Cloud Type")
    on TRAINING data only.

    ML concept: "Cloud Type" is a category code (0=Clear, 1=Probably
    Clear, ...), not a real number - treating it as a number would
    imply, e.g., that "Fog" (2) is "twice as much" as "Probably Clear"
    (1), which is meaningless. One-hot encoding turns it into several
    0/1 columns instead, one per category.

    `handle_unknown="ignore"` means that if the test set contains a
    category value never seen in training (e.g. a rare cloud type that
    only appears in the test period), it's encoded as all-zeros
    instead of crashing.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data ONLY.
    columns : list of str
        Categorical columns to encode.

    Returns
    -------
    sklearn.preprocessing.OneHotEncoder
        A fitted encoder. Use apply_encoder() to transform any
        DataFrame with it.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[columns])
    return encoder


def apply_encoder(df: pd.DataFrame, encoder: OneHotEncoder, columns: list) -> pd.DataFrame:
    """
    Apply an already-fitted one-hot encoder to `df`.

    Parameters
    ----------
    df : pandas.DataFrame
    encoder : OneHotEncoder
        An encoder previously returned by fit_encoder().
    columns : list of str
        Must be the same columns the encoder was fit on.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with `columns` removed and replaced by new
        0/1 columns, one per category (named e.g. "Cloud Type_0.0",
        "Cloud Type_1.0", ...).
    """
    encoded_array = encoder.transform(df[columns])
    encoded_columns = encoder.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)

    df = df.copy().drop(columns=columns)
    return pd.concat([df, encoded_df], axis=1)


# ---------------------------------------------------------------------------
# Bundled preprocessor (scaler + encoder together)
# ---------------------------------------------------------------------------


def fit_preprocessor(train_df: pd.DataFrame, numeric_columns: list, categorical_columns: list) -> dict:
    """
    Fit a scaler and an encoder together on TRAINING data, and bundle
    them into one plain dict so they can be passed around and saved as
    a single object representing "how to preprocess this experiment's
    data."

    This is intentionally just a dict, not a custom class - a class
    here would only be adding a name to something that's really just
    "two fitted objects plus which columns they apply to," which a
    dict already expresses clearly.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data ONLY.
    numeric_columns : list of str
        Columns to scale (see get_numeric_columns()).
    categorical_columns : list of str
        Columns to one-hot encode (see get_categorical_columns()).

    Returns
    -------
    dict
        {"scaler": StandardScaler, "encoder": OneHotEncoder,
         "numeric_columns": [...], "categorical_columns": [...]}
        Pass this whole dict to apply_preprocessor().
    """
    return {
        "scaler": fit_scaler(train_df, numeric_columns) if numeric_columns else None,
        "encoder": fit_encoder(train_df, categorical_columns) if categorical_columns else None,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def apply_preprocessor(df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    """
    Apply a preprocessor dict from fit_preprocessor() to any DataFrame
    (train, validation, or test).

    Parameters
    ----------
    df : pandas.DataFrame
    preprocessor : dict
        From fit_preprocessor().

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with numeric columns scaled and categorical
        columns one-hot encoded.
    """
    result = df.copy()
    if preprocessor["scaler"] is not None:
        result = apply_scaler(result, preprocessor["scaler"], preprocessor["numeric_columns"])
    if preprocessor["encoder"] is not None:
        result = apply_encoder(result, preprocessor["encoder"], preprocessor["categorical_columns"])
    return result


# ---------------------------------------------------------------------------
# Target scaling (optional - only for cross-city / multi-city experiments)
# ---------------------------------------------------------------------------


def fit_target_scaler(train_target: pd.Series) -> StandardScaler:
    """
    Fit a StandardScaler for the TARGET column (e.g. Output Power) on
    TRAINING data only.

    WHEN TO USE THIS: only when combining multiple cities' targets in
    one training run (e.g. a cross-city or transfer-learning
    experiment), because Output Power has very different physical
    scales per city - Davis's plant is roughly 3-3.5x the capacity of
    Huron/Santa Barbara/La Jolla (see course_context/DATASET_PROFILE.md).
    Without this, a model trained on combined cities would be
    dominated by whichever city has the largest raw numbers.

    WHEN NOT TO USE THIS: for a same-city experiment, don't bother -
    there's only one scale involved, so this adds complexity without
    benefit.

    IMPORTANT: if you scale the target for training, you MUST call
    inverse_transform_target() on the model's predictions (and on
    y_true, if it's also in scaled form) before computing RMSE/MAE/
    nRMSE - see evaluation.regression_metrics(). The project spec
    requires final regression metrics to be reported in the original
    kW scale, not the standardized scale
    (course_context/TEACHER_EXPECTATIONS.md, Problem 2).

    Parameters
    ----------
    train_target : pandas.Series
        The target column, TRAINING portion only.

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Fitted on a single column. Use apply_target_scaler() and
        inverse_transform_target() with it.
    """
    scaler = StandardScaler()
    scaler.fit(train_target.values.reshape(-1, 1))
    return scaler


def apply_target_scaler(target: pd.Series, scaler: StandardScaler) -> np.ndarray:
    """
    Scale a target Series using an already-fitted target scaler.

    Parameters
    ----------
    target : pandas.Series
    scaler : StandardScaler
        From fit_target_scaler().

    Returns
    -------
    numpy.ndarray
        1-D array of scaled values, same length as `target`.
    """
    return scaler.transform(target.values.reshape(-1, 1)).flatten()


def inverse_transform_target(scaled_values, scaler: StandardScaler) -> np.ndarray:
    """
    Convert scaled target values back to the original kW scale.

    Always call this on model predictions (and on y_true, if needed)
    before computing final regression metrics, if the target was
    scaled for training - see the warning in fit_target_scaler().

    Parameters
    ----------
    scaled_values : array-like
        Values in the scaled space (e.g. model predictions).
    scaler : StandardScaler
        The same scaler used to scale the target originally.

    Returns
    -------
    numpy.ndarray
        1-D array of values back in the original (kW) scale.
    """
    scaled_values = np.asarray(scaled_values).reshape(-1, 1)
    return scaler.inverse_transform(scaled_values).flatten()
