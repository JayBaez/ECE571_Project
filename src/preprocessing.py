"""
preprocessing.py

Reusable helpers for cleaning data and preparing it for a model:
identifying column types, handling missing values, scaling numeric
columns, and encoding categorical columns.

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
"""

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
