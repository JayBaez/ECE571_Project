"""
targets.py (Problem 1 — Supervised Classification)

Builds the two classification targets for Problem 1:

- Task A: Sky-condition (Clear / Partly Cloudy / Overcast), from the
  Clear-Sky Index. The thresholds are FIXED CONSTANTS from the project
  spec, not statistics learned from data - so there's no "fit on train
  only" concern here, unlike a scaler or the Task B terciles below.
  Applying a fixed rule to a value doesn't leak anything, no matter
  which rows it's applied to.

- Task B: Generation-regime (Low / Medium / High), from per-city
  Output Power terciles. Terciles ARE a statistic learned from data
  (the boundary values themselves depend on the data they're computed
  from), so these MUST be fit on the training portion only, then
  applied to test - exactly like a scaler. See
  fit_tercile_boundaries() / apply_tercile_labels() below.
"""

import numpy as np
import pandas as pd

from src.feature_engineering import add_clear_sky_index

SKY_CONDITION_CLASSES = ["Overcast", "Partly Cloudy", "Clear"]
GENERATION_REGIME_CLASSES = ["Low", "Medium", "High"]


def make_sky_condition_labels(df: pd.DataFrame) -> pd.Series:
    """
    Task A target: bin the Clear-Sky Index into Clear / Partly Cloudy
    / Overcast, using the exact thresholds from the project spec
    (course_context/TEACHER_EXPECTATIONS.md, Problem 1):
        Clear:         k >= 0.85
        Partly Cloudy: 0.4 <= k < 0.85
        Overcast:      k < 0.4

    These thresholds are used exactly as specified - not adjusted,
    even though course_context/EDA_REPORT.md's Clear-Sky Index section
    already checked they produce reasonable (non-degenerate) class
    sizes on the real data (72.0% / 18.7% / 9.3%).

    EDGE CASE (documented, not silently handled): Clear-Sky Index can
    exceed 1.0 in a small number of rows (up to ~1.5x in the verified
    data - course_context/EDA_REPORT.md, Section 14) due to brief
    instrument spikes. The spec's stated Clear range is "0.85-1.0,"
    but says nothing about what happens above 1.0. Decision: values
    above 1.0 are still labeled "Clear" (k >= 0.85 is treated as an
    open-ended lower bound, not a strict 0.85-1.0 window) since they
    represent even brighter-than-clear-sky conditions, not a
    different physical regime. This is a documented judgment call, not
    a change to the spec's thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain "GHI" and "Clearsky GHI".

    Returns
    -------
    pandas.Series
        Categorical labels: "Clear", "Partly Cloudy", or "Overcast".
        Rows where Clear-Sky Index is NaN (Clearsky GHI == 0 - not
        expected in this daytime-only dataset, see
        course_context/DATASET_PROFILE.md) get NaN, not a fabricated
        label - such rows must be dropped before training/evaluation,
        not silently assigned a class.
    """
    with_index = add_clear_sky_index(df)
    k = with_index["Clear_Sky_Index"]

    labels = pd.Series(np.select(
        [k >= 0.85, (k >= 0.4) & (k < 0.85), k < 0.4],
        ["Clear", "Partly Cloudy", "Overcast"],
        default=None,
    ), index=df.index, name="Sky_Condition")
    # np.select's default=None produces the string "None" for NaN rows
    # instead of an actual NaN - fix that explicitly rather than let it
    # silently become a spurious fourth class.
    labels = labels.where(k.notna(), other=np.nan)
    return labels


def fit_tercile_boundaries(train_output_power: pd.Series) -> tuple:
    """
    Compute the Low/Medium/High tercile boundaries for ONE city, using
    TRAINING data only.

    ML concept / leakage prevention: this is exactly analogous to
    fitting a StandardScaler - the boundary VALUES themselves are a
    statistic of the data they're computed from. Computing them from
    the full dataset (including test) and then evaluating on that same
    test set would leak test-set information into how the labels
    themselves are defined - see
    course_context/TEACHER_EXPECTATIONS.md, Problem 1's "MUST BE PER
    CITY" note, and course_context/EDA_REPORT.md Section 19 for why
    per-city (not global) terciles are also required.

    Parameters
    ----------
    train_output_power : pandas.Series
        One city's Output Power column, TRAINING portion only.

    Returns
    -------
    (low_boundary, high_boundary) : tuple of float
        Rows <= low_boundary -> "Low"
        low_boundary < rows <= high_boundary -> "Medium"
        rows > high_boundary -> "High"
    """
    clean = train_output_power.dropna()
    low_boundary = clean.quantile(1 / 3)
    high_boundary = clean.quantile(2 / 3)
    return float(low_boundary), float(high_boundary)


def apply_tercile_labels(output_power: pd.Series, low_boundary: float, high_boundary: float) -> pd.Series:
    """
    Apply already-fitted tercile boundaries to any Output Power Series
    (train or test) - analogous to preprocessing.apply_scaler().

    Parameters
    ----------
    output_power : pandas.Series
    low_boundary, high_boundary : float
        From fit_tercile_boundaries() - MUST come from that city's
        TRAINING data.

    Returns
    -------
    pandas.Series
        Categorical labels: "Low", "Medium", or "High". Missing
        Output Power values produce NaN, not a fabricated label.
    """
    labels = pd.Series(np.select(
        [output_power <= low_boundary, output_power > high_boundary],
        ["Low", "High"],
        default="Medium",
    ), index=output_power.index, name="Generation_Regime")
    labels = labels.where(output_power.notna(), other=np.nan)
    return labels


def report_class_distribution(labels: pd.Series, label_name: str) -> pd.DataFrame:
    """
    Small helper: return class counts/percentages as a DataFrame, and
    print them - used to check (and document) whether a test-set
    tercile split actually landed close to 33/33/33, since applying
    TRAIN-fitted boundaries to a chronologically different TEST period
    is not guaranteed to reproduce an exact one-third split (e.g. if
    the test period is disproportionately summer, "High" may be
    over-represented) - see Problem 1's Section 6 requirement to
    document this.

    Parameters
    ----------
    labels : pandas.Series
    label_name : str
        Used only in the printed message, e.g. "Davis train sky-condition".

    Returns
    -------
    pandas.DataFrame
        columns: class, count, pct
    """
    counts = labels.value_counts(dropna=True)
    pct = (100 * counts / counts.sum()).round(2)
    table = pd.DataFrame({"class": counts.index, "count": counts.values, "pct": pct.values})
    print(f"{label_name} class distribution:")
    print(table.to_string(index=False))
    return table
