"""
cleaning.py

Reusable functions for detecting and handling data-quality issues:
missing values, physically-invalid readings, and duplicate rows.

This is deliberately separate from preprocessing.py: cleaning.py is
about "is this data OK, and if not, what do we do about it, and what
happened" - it always reports what it found and what it changed.
preprocessing.py is about the fit-on-train/apply-to-test transforms
(scaling, encoding) that come after the data is already clean.

Design rule: cleaning never silently deletes or changes anything -
every function that modifies data also returns a report describing
exactly what happened, with real numbers from the actual DataFrame it
was given (see course_context/DATASET_PROFILE.md for the real
numbers found during Phase 0 - e.g. Amherst's 4 missing Output Power
rows, and the 2012-03-22 anomaly).
"""

import pandas as pd

from src.preprocessing import check_missing_values, handle_missing_values


def detect_invalid_numeric_values(df: pd.DataFrame) -> dict:
    """
    Check for physically-impossible values in the irradiance columns,
    based on the checks used during the Phase 0 dataset inspection
    (course_context/DATASET_PROFILE.md).

    This function only DETECTS and reports issues - it doesn't remove
    or fix anything. Deciding what to do about a flagged anomaly (e.g.
    the 2012-03-22 zero-Output-Power day) is a problem-specific
    decision, not something the cleaning step should silently resolve.

    Checks performed:
    - Negative irradiance values (DHI, DNI, GHI, and their Clearsky
      counterparts) - irradiance can't be negative.
    - GHI exceeding 1.2x Clearsky GHI - GHI shouldn't meaningfully
      exceed the theoretical clear-sky maximum.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        {"negative_irradiance": {column: count, ...},
         "ghi_exceeds_clearsky": count}
        Counts are 0 where no issue was found - this dict always has
        the same shape, so it's easy to check programmatically.
    """
    irradiance_columns = [
        c
        for c in ["DHI", "DNI", "GHI", "Clearsky DHI", "Clearsky DNI", "Clearsky GHI"]
        if c in df.columns
    ]
    negative_counts = {c: int((df[c] < 0).sum()) for c in irradiance_columns}

    ghi_exceeds_clearsky = 0
    if "GHI" in df.columns and "Clearsky GHI" in df.columns:
        ghi_exceeds_clearsky = int((df["GHI"] > df["Clearsky GHI"] * 1.2).sum())

    return {
        "negative_irradiance": negative_counts,
        "ghi_exceeds_clearsky": ghi_exceeds_clearsky,
    }


def detect_duplicate_rows(df: pd.DataFrame, subset: list = None) -> int:
    """
    Count exact duplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame
    subset : list of str, optional
        Only consider these columns when checking for duplicates
        (e.g. just the timestamp columns). Defaults to all columns.

    Returns
    -------
    int
        Number of duplicate rows found (not counting the first
        occurrence of each).
    """
    return int(df.duplicated(subset=subset).sum())


def clean_sheet(
    df: pd.DataFrame,
    target_column: str = "Output Power",
    missing_strategy: str = "interpolate",
    drop_duplicates: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    Run the standard cleaning pass on one city's sheet: detect and
    report missing values and duplicates, handle missing target
    values, and return a report of exactly what happened.

    This does NOT do anything to detected "invalid numeric values"
    (see detect_invalid_numeric_values()) - those are physically
    suspicious but not missing, so silently altering them here would
    be an undocumented modeling decision. Report and decide on those
    at the problem level instead.

    Parameters
    ----------
    df : pandas.DataFrame
        A single city's raw sheet (e.g. from data_loader.load_city()).
    target_column : str
        Column whose missing values should be handled (default
        "Output Power", the project's regression target - see
        course_context/DATASET_PROFILE.md for the known ~4 missing
        rows in the Amherst sheet).
    missing_strategy : str
        "interpolate" (default) or "drop" - passed to
        preprocessing.handle_missing_values() for `target_column` only.
        Other columns are left as-is (in this dataset, only
        `target_column` has missing values - see
        course_context/DATASET_PROFILE.md - but this function doesn't
        assume that will always be true; see the report's
        "missing_other_columns" field).
    drop_duplicates : bool
        If True, drop exact duplicate rows (keeping the first
        occurrence). Default False, since Phase 0 found zero
        duplicates in this dataset - only enable this if you have a
        specific reason to.
    verbose : bool
        If True (default), print a human-readable report.

    Returns
    -------
    (cleaned_df, report) : tuple of (pandas.DataFrame, dict)
        report contains: "rows_before", "missing_target",
        "missing_other_columns" (dict, non-target columns with any
        missing values), "duplicate_rows", "rows_after".
    """
    rows_before = len(df)
    missing_counts = check_missing_values(df)
    missing_target = int(missing_counts.get(target_column, 0))
    missing_other_columns = {
        col: int(count)
        for col, count in missing_counts.items()
        if col != target_column and count > 0
    }
    duplicate_rows = detect_duplicate_rows(df)

    cleaned_df = df.copy()

    if missing_target > 0:
        if missing_strategy == "interpolate":
            # Interpolation only makes physical sense in chronological
            # order (filling a gap using its actual time-neighbors) -
            # sort defensively even though loaded sheets are already
            # in order (see course_context/DATASET_PROFILE.md).
            cleaned_df = cleaned_df.sort_values(
                ["Year", "Month", "Day", "Hour", "Minute"]
            ).reset_index(drop=True)
        cleaned_df = handle_missing_values(
            cleaned_df, strategy=missing_strategy, columns=[target_column]
        )

    if drop_duplicates and duplicate_rows > 0:
        cleaned_df = cleaned_df.drop_duplicates()

    rows_after = len(cleaned_df)

    report = {
        "rows_before": rows_before,
        "missing_target": missing_target,
        "missing_other_columns": missing_other_columns,
        "duplicate_rows": duplicate_rows,
        "rows_after": rows_after,
    }

    if verbose:
        _print_cleaning_report(report, target_column, missing_strategy)

    return cleaned_df, report


def _print_cleaning_report(report: dict, target_column: str, missing_strategy: str) -> None:
    """Print a short, human-readable summary of a clean_sheet() report."""
    print(f"Rows before cleaning: {report['rows_before']}")
    print(f"Missing {target_column}: {report['missing_target']} ({missing_strategy})")
    if report["missing_other_columns"]:
        print(f"Missing values in other columns: {report['missing_other_columns']}")
    if report["duplicate_rows"]:
        print(f"Duplicate rows found: {report['duplicate_rows']}")
    print(f"Rows remaining: {report['rows_after']}")
