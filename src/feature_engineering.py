"""
feature_engineering.py

Reusable functions for building new features from the raw dataset
columns. Each function adds ONE feature group and is safe to call
independently, so problem-specific code can enable/disable groups as
needed (e.g. "I want time features but not lag features").

None of these functions modify the input DataFrame in place - they
all return a new DataFrame with extra column(s) added.

LEAKAGE WARNING - read before using add_clear_sky_index():
The Clear-Sky Index (k = GHI / Clearsky GHI) is literally the
definition of the Problem 1 sky-condition label (see
course_context/TEACHER_EXPECTATIONS.md, Problem 1). That means:
- It is SAFE to use as an input feature for Problem 2 (regression) or
  Problem 3 (dimension reduction), where it's just another weather
  descriptor.
- It is NOT SAFE to use as an input feature when the target is the
  sky-condition class derived from it - that would be using the
  answer to predict the answer. For that specific task, use the other
  weather variables instead (this project's spec is explicit about
  this - GHI and Clearsky GHI themselves must also be excluded as
  features for that task).

LEAKAGE WARNING - read before using add_lag_features():
Lag features (e.g. Output Power from 1-2 steps ago) must be computed
BEFORE any train/test split, on a single city's data sorted
chronologically. If you split first and add lags after, or if you mix
multiple cities together before computing lags, a lag value could
accidentally include information from across a split boundary or from
the wrong city. See course_context/EXPERIMENT_PLAN.md (Problem 2,
"Major risks") for more detail.
"""

import numpy as np
import pandas as pd


def add_clear_sky_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the Clear-Sky Index: k = GHI / Clearsky GHI.

    This measures how much of the theoretically-possible sunlight
    (under a perfectly clear sky) actually arrived. k close to 1 means
    clear skies; k close to 0 means heavy cloud cover. It is the exact
    quantity the project spec uses to define the sky-condition label
    (Clear: k >= 0.85, Partly cloudy: 0.4 <= k < 0.85, Overcast: k < 0.4
    - see course_context/TEACHER_EXPECTATIONS.md).

    See the LEAKAGE WARNING at the top of this file before using this
    as a classifier input feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain "GHI" and "Clearsky GHI" columns.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with a new "Clear_Sky_Index" column. Rows where
        "Clearsky GHI" is 0 (e.g. nighttime, not expected in this
        daytime-only dataset - see course_context/DATASET_PROFILE.md)
        get NaN instead of a division error.
    """
    df = df.copy()
    df["Clear_Sky_Index"] = np.where(
        df["Clearsky GHI"] > 0,
        df["GHI"] / df["Clearsky GHI"],
        np.nan,
    )
    return df


def add_time_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sine/cosine encodings of time-of-day, month, and day-of-year.

    ML concept: Hour 23 and Hour 0 are only 1 hour apart, but as plain
    numbers they look 23 apart - a model would wrongly think they're
    very different. Sine/cosine encoding fixes this by mapping cyclical
    values onto a circle, so "close in time" always means "close in
    the encoded features" too. Same idea for Month (December and
    January are 1 month apart, not 11) and day-of-year (Dec 31 and
    Jan 1 are 1 day apart, not 364).

    Month and day-of-year both capture "time of year," just at
    different resolutions - day-of-year is finer-grained and mostly
    makes Month_sin/Month_cos redundant, but Month is included too
    since a simpler model might prefer the coarser signal.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain "Year", "Month", "Day", "Hour", "Minute" columns.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with six new columns: "Hour_sin", "Hour_cos",
        "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos".
    """
    df = df.copy()

    fractional_hour = df["Hour"] + df["Minute"] / 60.0
    df["Hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24.0)

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    timestamps = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]))
    day_of_year = timestamps.dt.dayofyear
    # Use 366 so leap-year day-of-year values still map correctly onto
    # a full circle.
    df["DayOfYear_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)

    return df


def add_lag_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
    """
    Add lagged versions of `column` (e.g. Output Power at 1 and 2 steps
    ago), for a SINGLE city's data already sorted chronologically.

    Read the LEAKAGE WARNING at the top of this file before using this
    across a train/test split or across multiple cities.

    Parameters
    ----------
    df : pandas.DataFrame
        A single city's data, already sorted chronologically (e.g. by
        Year, Month, Day, Hour, Minute).
    column : str
        Column to lag, e.g. "Output Power".
    lags : list of int
        Which lags to create, e.g. [1, 2] for t-1 and t-2.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with new columns named e.g. "Output Power_lag1",
        "Output Power_lag2". The first `max(lags)` rows will have NaN
        in these new columns, since there's no earlier data to look
        back to.
    """
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


# Maps a feature-group name to the function that builds it, so callers
# can enable/disable groups by name instead of calling each function
# individually. Lag features are deliberately NOT included here, since
# they need a `column` and `lags` argument and the chronological-order
# safety warning above - call add_lag_features() directly when needed.
FEATURE_GROUPS = {
    "clear_sky_index": add_clear_sky_index,
    "time_cyclical": add_time_cyclical_features,
}


def add_feature_groups(df: pd.DataFrame, groups: list) -> pd.DataFrame:
    """
    Apply several feature groups at once, by name.

    Parameters
    ----------
    df : pandas.DataFrame
    groups : list of str
        Names from FEATURE_GROUPS, e.g. ["clear_sky_index", "time_cyclical"].

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with all requested feature groups added.

    Raises
    ------
    ValueError
        If a requested group name isn't in FEATURE_GROUPS.
    """
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {unknown}. "
            f"Available groups: {list(FEATURE_GROUPS.keys())}"
        )

    for group_name in groups:
        df = FEATURE_GROUPS[group_name](df)
    return df
