"""
Tests for src/data_loader.py.

Uses the real dataset for the "happy path" tests (this IS the loader's
job - reading the real file), and a small temporary .xlsx file for the
error-path tests, so those don't depend on the real file happening to
be well-formed or broken in a particular way.
"""

import os

import pandas as pd
import pytest

from src import data_loader


def test_list_sheets_finds_nine_sheets():
    sheets = data_loader.list_sheets()
    assert len(sheets) == 9


def test_list_sheets_missing_file_raises_filenotfounderror():
    with pytest.raises(FileNotFoundError):
        data_loader.list_sheets("this/path/does/not/exist.xlsx")


def test_load_sheet_returns_nonempty_dataframe():
    sheets = data_loader.list_sheets()
    df = data_loader.load_sheet(sheets[0])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_sheet_unknown_sheet_name_raises_valueerror():
    with pytest.raises(ValueError):
        data_loader.load_sheet("This Sheet Does Not Exist")


def test_parse_sheet_name_extracts_city_and_years():
    meta = data_loader.parse_sheet_name("Davis 5hr-daily '11-'16")
    assert meta == {"city": "Davis", "year_start": 2011, "year_end": 2016}


def test_parse_sheet_name_handles_unknown_pattern_gracefully():
    meta = data_loader.parse_sheet_name("Some Totally Unexpected Sheet Name")
    assert meta["city"] == "Some Totally Unexpected Sheet Name"
    assert meta["year_start"] is None
    assert meta["year_end"] is None


def test_get_sheet_name_long_vs_short_for_a_city_with_both():
    long_name = data_loader.get_sheet_name("Davis", years="long")
    short_name = data_loader.get_sheet_name("Davis", years="short")
    assert "11" in long_name and "16" in long_name
    assert "14" in short_name and "16" in short_name
    assert long_name != short_name


def test_get_sheet_name_amherst_only_has_one_option():
    # Amherst has only one sheet, so "long" and "short" must resolve
    # to the same sheet instead of crashing or silently picking wrong.
    long_name = data_loader.get_sheet_name("Amherst", years="long")
    short_name = data_loader.get_sheet_name("Amherst", years="short")
    assert long_name == short_name


def test_get_sheet_name_unknown_city_raises_valueerror():
    with pytest.raises(ValueError):
        data_loader.get_sheet_name("Nowhere")


def test_load_city_matches_load_sheet():
    via_city = data_loader.load_city("Davis", years="long")
    via_sheet = data_loader.load_sheet(data_loader.get_sheet_name("Davis", years="long"))
    assert via_city.shape == via_sheet.shape


def test_load_sheet_missing_required_column_raises_valueerror(tmp_path):
    # Build a tiny, deliberately broken workbook missing "Output Power".
    broken_path = os.path.join(tmp_path, "broken.xlsx")
    broken_df = pd.DataFrame(
        {
            "Year": [2020],
            "Month": [1],
            "Day": [1],
            "Hour": [10],
            "Minute": [0],
            "GHI": [500],
            "Clearsky GHI": [600],
            "Cloud Type": [0],
            # "Output Power" is intentionally missing.
        }
    )
    broken_df.to_excel(broken_path, sheet_name="Broken Sheet", index=False)

    with pytest.raises(ValueError):
        data_loader.load_sheet("Broken Sheet", path=broken_path)


def test_load_multiple_sheets_returns_one_df_per_sheet():
    sheets = data_loader.list_sheets()[:2]
    result = data_loader.load_multiple_sheets(sheets)
    assert set(result.keys()) == set(sheets)
    for df in result.values():
        assert len(df) > 0
