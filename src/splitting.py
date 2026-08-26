"""
splitting.py

Reusable functions for splitting data into train/test sets. This
project needs THREE genuinely different kinds of split, and mixing
them up is a real risk (see course_context/EXPERIMENT_PLAN.md, "Major
risks" sections), so each one gets its own clearly-named function
instead of one do-everything function:

1. chronological_split()   - for time series: earlier rows train,
                              later rows test. NO shuffling.
2. cross_city_split()      - one city's full data trains, a different
                              city's full data tests. Zero overlap by
                              construction, since they're different
                              sheets.
3. random_labeled_subset() / few_shot_sample() - reproducible random
                              sampling used ONLY for semi-supervised
                              labeled fractions (Problem 4) and
                              few-shot transfer learning (Problem 5).
                              This is a different concept from #1 and
                              #2: it's about how many labels you're
                              ALLOWED to see, not about time order or
                              city.
"""

import pandas as pd


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple:
    """
    Split a single city's time-series data into an earlier "train"
    portion and a later "test" portion. Never shuffles.

    ML concept: shuffling before splitting a time series is a classic
    leakage mistake - it lets the model "see the future" during
    training (e.g. training on a June reading, then testing on a May
    reading from the same year is testing on the model's own past).
    The project spec requires chronological splits for exactly this
    reason (course_context/TEACHER_EXPECTATIONS.md).

    Parameters
    ----------
    df : pandas.DataFrame
        A single city's data. Must contain Year, Month, Day, Hour,
        Minute columns (used to sort chronologically before splitting,
        as a safety net in case the rows weren't already in order).
    train_frac : float
        Fraction of rows to use for training, e.g. 0.8 for an 80/20
        split. Must be between 0 and 1 (exclusive).

    Returns
    -------
    (train_df, test_df, split_info) : tuple
        train_df, test_df : pandas.DataFrame
            train_df is the earlier `train_frac` portion, test_df is
            the remaining later portion.
        split_info : dict
            {"train_frac", "split_index", "n_train", "n_test",
             "last_train_timestamp", "first_test_timestamp"} -
            save this alongside an experiment's results so the exact
            split point can be reported/reproduced (the project spec
            requires split indices to be documented - see
            course_context/TEACHER_EXPECTATIONS.md).

    Raises
    ------
    ValueError
        If train_frac is not strictly between 0 and 1.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")

    time_cols = ["Year", "Month", "Day", "Hour", "Minute"]
    df_sorted = df.sort_values(time_cols).reset_index(drop=True)
    split_index = int(len(df_sorted) * train_frac)

    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()

    split_info = {
        "train_frac": train_frac,
        "split_index": split_index,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "last_train_timestamp": tuple(train_df[time_cols].iloc[-1]) if len(train_df) else None,
        "first_test_timestamp": tuple(test_df[time_cols].iloc[0]) if len(test_df) else None,
    }
    return train_df, test_df, split_info


def verify_no_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, key_columns: list = None) -> bool:
    """
    Check that two DataFrames share zero rows, based on a set of key
    columns (defaults to the timestamp columns). Useful as a sanity
    check right after any split - chronological, cross-city, or
    labeled/unlabeled - to confirm there's genuinely no leakage between
    the two halves.

    Parameters
    ----------
    df_a, df_b : pandas.DataFrame
    key_columns : list of str, optional
        Columns that together identify a row (default: Year, Month,
        Day, Hour, Minute).

    Returns
    -------
    bool
        True if there is no overlap, False if at least one matching
        row was found in both DataFrames.
    """
    key_columns = key_columns or ["Year", "Month", "Day", "Hour", "Minute"]
    keys_a = set(map(tuple, df_a[key_columns].values.tolist()))
    keys_b = set(map(tuple, df_b[key_columns].values.tolist()))
    return len(keys_a & keys_b) == 0


def cross_city_split(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_city: str = None,
    target_city: str = None,
) -> dict:
    """
    Package two different cities' data as a source (train) / target
    (test) pair for a cross-city or transfer-learning experiment.

    Because `source_df` and `target_df` come from different sheets
    (different cities), there is zero row overlap by construction -
    this function's job is mainly to keep the intent explicit (which
    city is training, which is testing) and catch an obvious mistake:
    accidentally passing the same city's data as both source and
    target.

    Parameters
    ----------
    source_df : pandas.DataFrame
        The data-rich city used for training (e.g. Davis).
    target_df : pandas.DataFrame
        The city used for testing/evaluation (e.g. Amherst, Huron,
        Santa Barbara, or La Jolla).
    source_city, target_city : str, optional
        Names to record for logging/results (e.g. "Davis", "Amherst").
        If not given, this function tries to read them from
        `df.attrs["city"]` (set automatically by data_loader.load_sheet()).

    Returns
    -------
    dict
        {"train": source_df, "test": target_df,
         "source_city": ..., "target_city": ...}

    Raises
    ------
    ValueError
        If source_city and target_city resolve to the same city - a
        cross-city split by definition needs two different cities.
    """
    source_city = source_city or source_df.attrs.get("city")
    target_city = target_df.attrs.get("city") if target_city is None else target_city

    if source_city is not None and source_city == target_city:
        raise ValueError(
            f"source_city and target_city are both '{source_city}' - "
            f"a cross-city split needs two different cities."
        )

    return {
        "train": source_df,
        "test": target_df,
        "source_city": source_city,
        "target_city": target_city,
    }


def random_labeled_subset(df: pd.DataFrame, label_fraction: float, seed: int) -> tuple:
    """
    Randomly split a (training-portion) DataFrame into a "labeled"
    subset and an "unlabeled" subset, for semi-supervised learning
    (Problem 4).

    IMPORTANT: only call this on data that is already inside your
    training split (i.e. call chronological_split() first, then call
    this on the resulting train_df). Never call this before splitting
    off the test set - see course_context/EXPERIMENT_PLAN.md,
    Problem 4 "Major risks".

    Parameters
    ----------
    df : pandas.DataFrame
        The TRAINING portion only.
    label_fraction : float
        Fraction of rows to treat as "labeled", e.g. 0.1 for 10%.
        Must be between 0 and 1 (exclusive).
    seed : int
        Random seed, for reproducible sampling.

    Returns
    -------
    (labeled_df, unlabeled_df) : tuple of pandas.DataFrame
    """
    if not (0.0 < label_fraction < 1.0):
        raise ValueError(f"label_fraction must be between 0 and 1, got {label_fraction}")

    labeled_df = df.sample(frac=label_fraction, random_state=seed)
    unlabeled_df = df.drop(index=labeled_df.index)
    return labeled_df, unlabeled_df


def few_shot_sample(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """
    Randomly sample exactly k rows from `df`, for few-shot transfer
    learning (Problem 5: fine-tuning on a small target-city sample).

    Parameters
    ----------
    df : pandas.DataFrame
        The target city's data to sample from (typically its training
        portion only).
    k : int
        Number of rows to sample, e.g. 10, 50, or 100 per the project
        spec (course_context/TEACHER_EXPECTATIONS.md, Problem 5).
    seed : int
        Random seed, for reproducible sampling.

    Returns
    -------
    pandas.DataFrame
        k randomly sampled rows.

    Raises
    ------
    ValueError
        If k is larger than the number of rows available.
    """
    if k > len(df):
        raise ValueError(
            f"Requested k={k} samples, but only {len(df)} rows are available."
        )
    return df.sample(n=k, random_state=seed)
