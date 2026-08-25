"""
data_loader.py

Reusable functions for loading the project's Excel dataset
("Further Consolidated Data, HnL.xlsx") into pandas DataFrames.

Design goals (see course_context/DATASET_PROFILE.md for the full
dataset investigation this module is based on):
- Never silently modify the data - functions here only READ and
  return DataFrames, they don't clean, scale, or engineer features.
  (Cleaning lives in preprocessing.py, features live in
  feature_engineering.py.)
- Fail with a clear error message rather than a confusing one, so a
  missing file or a typo'd sheet name is obvious immediately.
- Don't hard-code fragile assumptions (e.g. exact row counts) - only
  check for what the rest of the framework actually depends on.
"""

import os
import re

import pandas as pd

# Default location of the raw Excel workbook, relative to the project
# root. Can always be overridden by passing an explicit `path`.
DEFAULT_DATA_PATH = os.path.join(
    "course", "Further Consolidated Data, HnL.xlsx"
)

# Columns every sheet is expected to have, based on the Phase 0 dataset
# inspection (course_context/DATASET_PROFILE.md). We check for these
# because later modules (splitting, feature engineering) depend on
# them existing - not because we expect the data to change, but so a
# mistake (e.g. loading the wrong file) is caught immediately instead
# of failing confusingly three steps later.
REQUIRED_COLUMNS = [
    "Year", "Month", "Day", "Hour", "Minute",
    "GHI", "Clearsky GHI", "Cloud Type", "Output Power",
]

# Maps the abbreviated city spelling used in sheet names to a clean,
# full city name. Built from the exact 9 sheet names found during
# Phase 0 (course_context/DATASET_PROFILE.md) - if a sheet name doesn't
# match any known pattern, parse_sheet_name() falls back gracefully
# instead of crashing (see below).
_CITY_NAME_MAP = {
    "amhst": "Amherst",
    "davis": "Davis",
    "huron": "Huron",
    "snt.barb": "Santa Barbara",
    "lajolla": "La Jolla",
}


def _resolve_path(path: str = None) -> str:
    """Return the path to use, falling back to the default if none given."""
    return path if path is not None else DEFAULT_DATA_PATH


def list_sheets(path: str = None) -> list:
    """
    List every sheet name in the Excel workbook, without loading the
    (large) data itself.

    Parameters
    ----------
    path : str, optional
        Path to the .xlsx file. Defaults to DEFAULT_DATA_PATH.

    Returns
    -------
    list of str
        Sheet names exactly as they appear in the workbook.

    Raises
    ------
    FileNotFoundError
        If the Excel file doesn't exist at the given path.
    """
    path = _resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find the dataset at '{path}'. "
            f"Check that the file exists and the path is correct."
        )
    workbook = pd.ExcelFile(path)
    return workbook.sheet_names


def parse_sheet_name(sheet_name: str) -> dict:
    """
    Extract city and year-range information from a sheet name, e.g.
    "Davis 5hr-daily '11-'16" -> {"city": "Davis", "year_start": 2011,
    "year_end": 2016}.

    This is a best-effort parser, not a strict requirement: if a sheet
    name doesn't match the expected pattern, this returns the raw
    sheet name as the city and None for the years, rather than
    raising an error. That way, adding a new sheet with an unexpected
    name later won't break the whole loader.

    Parameters
    ----------
    sheet_name : str
        A sheet name as returned by list_sheets().

    Returns
    -------
    dict
        {"city": str, "year_start": int or None, "year_end": int or None}
    """
    prefix = sheet_name.split()[0].lower()
    city = _CITY_NAME_MAP.get(prefix, sheet_name)

    # Sheet names encode years like "'11-'16" - two 2-digit years.
    years = re.findall(r"'(\d{2})", sheet_name)
    if len(years) == 2:
        # Assume 20xx since this dataset only covers 2011-2020.
        year_start = 2000 + int(years[0])
        year_end = 2000 + int(years[1])
    else:
        year_start, year_end = None, None

    return {"city": city, "year_start": year_start, "year_end": year_end}


def load_sheet(sheet_name: str, path: str = None) -> pd.DataFrame:
    """
    Load a single sheet from the Excel workbook as a pandas DataFrame.

    This function does NOT clean, scale, or modify the data in any
    way - it returns exactly what's in the spreadsheet. Cleaning
    happens later, explicitly, in preprocessing.py.

    Parameters
    ----------
    sheet_name : str
        Exact sheet name, e.g. "Davis 5hr-daily '11-'16". Use
        list_sheets() to see valid names.
    path : str, optional
        Path to the .xlsx file. Defaults to DEFAULT_DATA_PATH.

    Returns
    -------
    pandas.DataFrame
        The raw sheet contents, with city/year metadata attached in
        `df.attrs` (this is just informational metadata, not a change
        to the actual data values).

    Raises
    ------
    FileNotFoundError
        If the Excel file doesn't exist.
    ValueError
        If the requested sheet name doesn't exist in the workbook, or
        if the sheet is missing columns the rest of the framework
        depends on.
    """
    path = _resolve_path(path)
    available_sheets = list_sheets(path)

    if sheet_name not in available_sheets:
        raise ValueError(
            f"Sheet '{sheet_name}' was not found in '{path}'.\n"
            f"Available sheets are:\n  " + "\n  ".join(available_sheets)
        )

    df = pd.read_excel(path, sheet_name=sheet_name)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing expected column(s): "
            f"{missing_columns}. Found columns: {list(df.columns)}"
        )

    metadata = parse_sheet_name(sheet_name)
    df.attrs["sheet_name"] = sheet_name
    df.attrs["city"] = metadata["city"]
    df.attrs["year_start"] = metadata["year_start"]
    df.attrs["year_end"] = metadata["year_end"]

    return df


def load_multiple_sheets(sheet_names: list, path: str = None) -> dict:
    """
    Load several sheets at once.

    Parameters
    ----------
    sheet_names : list of str
        Sheet names to load. Use list_sheets() to see valid names.
    path : str, optional
        Path to the .xlsx file. Defaults to DEFAULT_DATA_PATH.

    Returns
    -------
    dict
        {sheet_name: DataFrame}, one entry per requested sheet.
        Sheets are kept separate (not concatenated) because different
        cities have very different Output Power scales (see
        course_context/DATASET_PROFILE.md) - combining them is a
        problem-specific decision, not something the loader should
        decide silently.
    """
    return {name: load_sheet(name, path=path) for name in sheet_names}
