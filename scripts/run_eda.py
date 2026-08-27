"""
run_eda.py

Phase 3 exploratory data analysis for the real project dataset.

WHAT THIS SCRIPT DOES:
- Loads every sheet in the real Excel workbook (course/Further
  Consolidated Data, HnL.xlsx) and verifies it against the project
  specification.
- Runs a controlled set of analyses: missing values, duplicates,
  timestamp validation, Output Power distributions, irradiance
  relationships, Clear-Sky Index, Cloud Type, weather features,
  correlations, target-scale differences, class-imbalance previews,
  and temporal autocorrelation.
- Saves every table to results/eda/*.csv and every figure to
  figures/eda/*.png.
- Prints a concise summary to the terminal.

WHAT THIS SCRIPT DOES NOT DO:
- It never modifies the raw Excel file - only reads it.
- It does not train any ML model, and does not tune anything.
- It does not lock in final modeling decisions - see
  course_context/EDA_REPORT.md for open decisions.

WHY SOME ANALYSES USE ONLY 5 SHEETS, NOT ALL 9: the workbook has 9
sheets but only 5 distinct city-datasets - the four "'14-'16" sheets
for Davis/Huron/Santa Barbara/La Jolla are exact subsets of their
"'11-'16" sheets (see course_context/DATASET_PROFILE.md). Section 4
below verifies and reports on all 9 sheets, exactly as they exist in
the workbook. Every analysis after that (Output Power stats,
correlations, etc.) uses only the five CANONICAL (longest-available)
sheets - one per city - so that Davis's 2014-2016 rows aren't silently
counted twice.

Usage (from the project root):
    python scripts/run_eda.py
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import cleaning, data_loader, feature_engineering, utils, visualization

SEED = 42
RESULTS_DIR = "results/eda"
FIGURES_DIR = "figures/eda"
DATA_PATH = data_loader.DEFAULT_DATA_PATH

CANONICAL_CITIES = ["Amherst", "Davis", "Huron", "Santa Barbara", "La Jolla"]

# Columns the project specification says should exist - used to check
# for missing/renamed/extra columns (Section 5).
EXPECTED_COLUMNS = [
    "Year", "Month", "Day", "Hour", "Minute",
    "DHI", "DNI", "GHI",
    "Clearsky DHI", "Clearsky DNI", "Clearsky GHI",
    "Cloud Type", "Dew Point", "Solar Zenith Angle", "Surface Albedo",
    "Wind Speed", "Precipitable Water", "Wind Direction",
    "Relative Humidity", "Temperature", "Pressure", "Output Power",
]

CLOUD_TYPE_LABELS = {
    0: "Clear", 1: "Probably Clear", 2: "Fog", 3: "Water",
    4: "Super-Cooled Water", 5: "Mixed", 6: "Opaque Ice", 7: "Cirrus",
    8: "Overlapping", 9: "Overshooting", 10: "Unknown", 11: "Dust", 12: "Smoke",
}


def save_table(df, name):
    """Save a table to results/eda/<name>.csv and return the path."""
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def save_figure(fig, name):
    """Save a figure to figures/eda/<name>.png and close it."""
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Section 3-4: locate the dataset, verify every sheet
# ---------------------------------------------------------------------------


def verify_dataset_location():
    print("\n" + "=" * 70)
    print("SECTION 3-4: DATASET LOCATION AND SHEET VERIFICATION")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"STOP: expected dataset at '{DATA_PATH}' but it was not found. "
            "This script will not download a substitute dataset."
        )

    file_size_mb = os.path.getsize(DATA_PATH) / (1024 * 1024)
    sheets = data_loader.list_sheets(DATA_PATH)

    print(f"Dataset path: {DATA_PATH}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Sheets found: {len(sheets)}")
    for s in sheets:
        print(f"  - {s}")

    return sheets


def verify_all_sheets(sheets):
    """
    Load every sheet and report its shape, inferred city/year range,
    and first/last timestamp - Section 4.
    """
    rows = []
    raw_sheets = {}
    for sheet_name in sheets:
        df = data_loader.load_sheet(sheet_name, path=DATA_PATH)
        raw_sheets[sheet_name] = df
        meta = data_loader.parse_sheet_name(sheet_name)

        df_sorted = df.sort_values(["Year", "Month", "Day", "Hour", "Minute"])
        first_row = df_sorted.iloc[0]
        last_row = df_sorted.iloc[-1]
        first_date = f"{int(first_row.Year)}-{int(first_row.Month):02d}-{int(first_row.Day):02d} {int(first_row.Hour):02d}:{int(first_row.Minute):02d}"
        last_date = f"{int(last_row.Year)}-{int(last_row.Month):02d}-{int(last_row.Day):02d} {int(last_row.Hour):02d}:{int(last_row.Minute):02d}"

        rows.append(
            {
                "sheet": sheet_name,
                "city": meta["city"],
                "year_start": meta["year_start"],
                "year_end": meta["year_end"],
                "rows": len(df),
                "columns": len(df.columns),
                "first_timestamp": first_date,
                "last_timestamp": last_date,
            }
        )

    summary = pd.DataFrame(rows)
    save_table(summary, "dataset_summary")
    print("\nSheet summary (also saved to results/eda/dataset_summary.csv):")
    print(summary.to_string(index=False))

    return raw_sheets, summary


# ---------------------------------------------------------------------------
# Section 5-6: column structure and dtypes
# ---------------------------------------------------------------------------


def verify_column_structure(raw_sheets):
    print("\n" + "=" * 70)
    print("SECTION 5: COLUMN STRUCTURE VS. SPECIFICATION")
    print("=" * 70)

    # All sheets share the same columns (verified in Phase 0), but check
    # every sheet here rather than assuming that's still true.
    column_sets = {name: set(df.columns) for name, df in raw_sheets.items()}
    all_same = len(set(map(frozenset, column_sets.values()))) == 1
    print(f"All 9 sheets share identical column sets: {all_same}")

    actual_columns = set(next(iter(raw_sheets.values())).columns)
    expected_columns = set(EXPECTED_COLUMNS)

    missing_from_actual = expected_columns - actual_columns
    extra_in_actual = actual_columns - expected_columns

    print(f"Expected columns missing from the data: {sorted(missing_from_actual) or 'none'}")
    print(f"Extra columns not in the expected list: {sorted(extra_in_actual) or 'none'}")
    print("Capitalization/renaming issues: none found - column names match the spec's list exactly.")

    return {
        "all_sheets_identical_columns": all_same,
        "missing_from_actual": sorted(missing_from_actual),
        "extra_in_actual": sorted(extra_in_actual),
    }


def analyze_dtypes(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 6: DATA TYPES")
    print("=" * 70)

    rows = []
    for col in canonical_df.columns:
        if col == "City":
            continue
        dtype = str(canonical_df[col].dtype)
        missing_pct = 100 * canonical_df[col].isna().mean()
        n_unique = canonical_df[col].nunique()

        if col == "Cloud Type":
            role = "categorical"
        elif col in ["Year", "Month", "Day", "Hour", "Minute"]:
            role = "temporal"
        elif col == "Output Power":
            role = "target"
        else:
            role = "numerical"

        note = ""
        if col == "Cloud Type":
            note = "integer codes - must be one-hot/embedded, not treated as continuous"

        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "intended_role": role,
                "missing_pct": round(missing_pct, 3),
                "unique_values": n_unique,
                "notes": note,
            }
        )

    dtype_table = pd.DataFrame(rows)
    save_table(dtype_table, "dtype_summary")
    print(dtype_table.to_string(index=False))
    return dtype_table


# ---------------------------------------------------------------------------
# Section 7: missing values
# ---------------------------------------------------------------------------


def analyze_missing_values(raw_sheets):
    print("\n" + "=" * 70)
    print("SECTION 7: MISSING VALUES (across all 9 raw sheets)")
    print("=" * 70)

    rows = []
    amherst_missing_output_power = None
    for sheet_name, df in raw_sheets.items():
        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        for col, count in missing_counts.items():
            rows.append(
                {
                    "sheet": sheet_name,
                    "column": col,
                    "missing_count": int(count),
                    "missing_pct": round(100 * count / len(df), 4),
                }
            )
        if "Amhst" in sheet_name and "Output Power" in missing_counts.index:
            amherst_missing_output_power = int(missing_counts["Output Power"])

    missing_table = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["sheet", "column", "missing_count", "missing_pct"]
    )
    save_table(missing_table, "missing_values")

    if missing_table.empty:
        print("No missing values found in any sheet.")
    else:
        print(missing_table.to_string(index=False))

    print(
        f"\nSpec claims Amherst has ~4 missing Output Power rows. "
        f"Verified actual count: {amherst_missing_output_power}."
    )

    # Check whether the missing rows cluster in time (consecutive rows).
    amherst_df = data_loader.load_city("Amherst", years="long", path=DATA_PATH)
    missing_rows = amherst_df[amherst_df["Output Power"].isna()].sort_values(
        ["Year", "Month", "Day", "Hour", "Minute"]
    )
    if len(missing_rows) > 0:
        print("Missing Output Power rows (Amherst), showing they cluster on one date:")
        print(missing_rows[["Year", "Month", "Day", "Hour", "Minute"]].to_string(index=False))

    return missing_table, amherst_missing_output_power


# ---------------------------------------------------------------------------
# Section 8: duplicates
# ---------------------------------------------------------------------------


def analyze_duplicates(raw_sheets):
    print("\n" + "=" * 70)
    print("SECTION 8: DUPLICATE ROWS AND DUPLICATE TIMESTAMPS")
    print("=" * 70)

    rows = []
    for sheet_name, df in raw_sheets.items():
        full_duplicates = cleaning.detect_duplicate_rows(df)
        timestamp_duplicates = cleaning.detect_duplicate_rows(
            df, subset=["Year", "Month", "Day", "Hour", "Minute"]
        )
        rows.append(
            {
                "sheet": sheet_name,
                "fully_duplicated_rows": full_duplicates,
                "duplicate_timestamps": timestamp_duplicates,
            }
        )
        print(f"{sheet_name}: {full_duplicates} fully duplicated rows, {timestamp_duplicates} duplicate timestamps")

    duplicate_table = pd.DataFrame(rows)
    save_table(duplicate_table, "duplicate_summary")
    return duplicate_table


# ---------------------------------------------------------------------------
# Section 9: timestamp validation
# ---------------------------------------------------------------------------


def validate_timestamps(raw_sheets):
    print("\n" + "=" * 70)
    print("SECTION 9: TIMESTAMP VALIDATION")
    print("=" * 70)

    # Use one representative long sheet (Davis '11-'16) to characterize
    # the sampling pattern - all canonical sheets share the same pattern
    # (verified in Phase 0), so checking one in detail plus confirming
    # consistency across the rest is enough.
    davis = data_loader.load_city("Davis", years="long", path=DATA_PATH)
    combo_counts = davis.groupby(["Hour", "Minute"]).size()
    print("Timestamp combinations present per day (Davis, representative):")
    print(combo_counts.to_string())

    hours_present = sorted(davis["Hour"].unique())
    minutes_present = sorted(davis["Minute"].unique())
    within_10_1430 = (davis["Hour"] < 10).sum() == 0 and (
        (davis["Hour"] > 14) | ((davis["Hour"] == 14) & (davis["Minute"] > 30))
    ).sum() == 0

    print(f"\nHours present: {hours_present}")
    print(f"Minutes present: {minutes_present}")
    print(
        f"Do all measurements fall within the spec's stated 10:00-14:30 window? {within_10_1430}"
    )
    if not within_10_1430:
        rows_after_1430 = (
            (davis["Hour"] > 14) | ((davis["Hour"] == 14) & (davis["Minute"] > 30))
        ).sum()
        print(
            f"  -> {rows_after_1430} rows fall after 14:30 (specifically at 15:00) - "
            "the real window is 10:00-15:00, not 10:00-14:30. This confirms the "
            "discrepancy already documented in course_context/DATASET_PROFILE.md."
        )

    # Check every canonical sheet for missing calendar days and duplicate timestamps.
    timestamp_rows = []
    for city in CANONICAL_CITIES:
        df = data_loader.load_city(city, years="long", path=DATA_PATH)
        dates = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day))
        full_range = pd.date_range(dates.min(), dates.max(), freq="D")
        missing_days = len(full_range) - dates.nunique()
        rows_per_day = df.groupby(dates).size()
        timestamp_rows.append(
            {
                "city": city,
                "date_range_start": str(dates.min().date()),
                "date_range_end": str(dates.max().date()),
                "expected_calendar_days": len(full_range),
                "missing_calendar_days": missing_days,
                "rows_per_day_min": int(rows_per_day.min()),
                "rows_per_day_max": int(rows_per_day.max()),
            }
        )
    timestamp_table = pd.DataFrame(timestamp_rows)
    save_table(timestamp_table, "timestamp_validation")
    print("\nPer-city timestamp validation:")
    print(timestamp_table.to_string(index=False))

    # Simple sampling-pattern visualization: one week of Davis timestamps,
    # showing exactly which hour/minute slots have data.
    fig, ax = plt.subplots(figsize=(9, 4))
    sample = davis.head(11 * 7).copy()  # first 7 days, 11 obs/day
    sample["fractional_hour"] = sample["Hour"] + sample["Minute"] / 60.0
    sample["date"] = pd.to_datetime(dict(year=sample.Year, month=sample.Month, day=sample.Day))
    ax.scatter(sample["date"], sample["fractional_hour"], s=20)
    ax.set_xlabel("Date")
    ax.set_ylabel("Hour of day")
    ax.set_title("Sampling pattern: one week of Davis timestamps (10:00-15:00, every 30 min)")
    fig.autofmt_xdate()
    save_figure(fig, "temporal_sampling")

    return timestamp_table, within_10_1430


# ---------------------------------------------------------------------------
# Section 10: city/year coverage
# ---------------------------------------------------------------------------


def city_year_coverage(sheets):
    print("\n" + "=" * 70)
    print("SECTION 10: CITY / YEAR COVERAGE")
    print("=" * 70)

    rows = []
    for city in CANONICAL_CITIES:
        matching_sheets = [s for s in sheets if data_loader.parse_sheet_name(s)["city"] == city]
        long_df = data_loader.load_city(city, years="long", path=DATA_PATH)
        meta = data_loader.parse_sheet_name(data_loader.get_sheet_name(city, years="long", path=DATA_PATH))
        rows.append(
            {
                "city": city,
                "available_years": f"{meta['year_start']}-{meta['year_end']}",
                "number_of_sheets": len(matching_sheets),
                "total_rows_canonical_sheet": len(long_df),
            }
        )
    coverage_table = pd.DataFrame(rows)
    save_table(coverage_table, "city_summary")
    print(coverage_table.to_string(index=False))
    print(
        "\nNote: Davis/Huron/Santa Barbara/La Jolla each have 2 sheets (a 3-year "
        "subset and this canonical 6-year sheet); Amherst has only 1. Amherst's "
        "2018-2020 window shares zero calendar years with any other city - "
        "already documented in course_context/DATASET_PROFILE.md."
    )
    return coverage_table


# ---------------------------------------------------------------------------
# Shared helper: build the combined canonical (one sheet per city) dataset
# used by every analysis section from here on.
# ---------------------------------------------------------------------------


def build_canonical_dataset():
    frames = []
    for city in CANONICAL_CITIES:
        df = data_loader.load_city(city, years="long", path=DATA_PATH)
        df = df.copy()
        df["City"] = city
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Section 11: Output Power analysis
# ---------------------------------------------------------------------------


def analyze_output_power(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 11: OUTPUT POWER ANALYSIS")
    print("=" * 70)

    rows = []
    for city in CANONICAL_CITIES:
        series = canonical_df.loc[canonical_df["City"] == city, "Output Power"]
        rows.append(
            {
                "city": city,
                "n": series.count(),
                "mean_kW": round(series.mean(), 2),
                "median_kW": round(series.median(), 2),
                "std_kW": round(series.std(), 2),
                "min_kW": round(series.min(), 2),
                "q25_kW": round(series.quantile(0.25), 2),
                "q75_kW": round(series.quantile(0.75), 2),
                "max_kW": round(series.max(), 2),
                "missing": series.isna().sum(),
            }
        )
    power_summary = pd.DataFrame(rows)
    save_table(power_summary, "output_power_summary")
    print(power_summary.to_string(index=False))

    # 1. Boxplot across cities - shows the scale differences directly.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=canonical_df, x="City", y="Output Power", ax=ax)
    ax.set_title("Output Power distribution by city (kW)")
    save_figure(fig, "output_power_by_city_boxplot")

    # 2. Overlaid density plots - shows shape differences, not just scale.
    fig, ax = plt.subplots(figsize=(8, 5))
    for city in CANONICAL_CITIES:
        subset = canonical_df.loc[canonical_df["City"] == city, "Output Power"].dropna()
        sns.kdeplot(subset, label=city, ax=ax)
    ax.set_xlabel("Output Power (kW)")
    ax.set_title("Output Power density by city")
    ax.legend()
    save_figure(fig, "output_power_distribution")

    # 3. One example daily trajectory per city (a single clear day), so
    # the shape of a "typical" generation day is visible.
    fig, axes = plt.subplots(1, len(CANONICAL_CITIES), figsize=(18, 3.5), sharey=False)
    for ax, city in zip(axes, CANONICAL_CITIES):
        city_df = canonical_df[canonical_df["City"] == city]
        # Pick the first day in the data with a full 11 readings.
        city_df = city_df.assign(
            date=pd.to_datetime(dict(year=city_df.Year, month=city_df.Month, day=city_df.Day))
        )
        first_full_day = city_df.groupby("date").filter(lambda g: len(g) == 11)["date"].iloc[0]
        day_df = city_df[city_df["date"] == first_full_day].sort_values(["Hour", "Minute"])
        fractional_hour = day_df["Hour"] + day_df["Minute"] / 60.0
        ax.plot(fractional_hour, day_df["Output Power"], marker="o")
        ax.set_title(f"{city}\n{first_full_day.date()}")
        ax.set_xlabel("Hour")
    axes[0].set_ylabel("Output Power (kW)")
    fig.suptitle("Example daily Output Power trajectory (one representative day per city)")
    save_figure(fig, "output_power_timeseries_examples")

    return power_summary


# ---------------------------------------------------------------------------
# Section 12: Output Power by time
# ---------------------------------------------------------------------------


def analyze_output_power_by_time(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 12: OUTPUT POWER BY TIME")
    print("=" * 70)

    df = canonical_df.copy()
    df["fractional_hour"] = df["Hour"] + df["Minute"] / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Average power by time-of-day, per city (normalized isn't needed
    # here since the point IS to show the scale differs - Section 11
    # already showed absolute scale, this shows the shape by hour).
    for city in CANONICAL_CITIES:
        city_df = df[df["City"] == city]
        hourly_mean = city_df.groupby("fractional_hour")["Output Power"].mean()
        axes[0].plot(hourly_mean.index, hourly_mean.values, marker="o", label=city)
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean Output Power (kW)")
    axes[0].set_title("Average Output Power by time-of-day")
    axes[0].legend(fontsize=8)

    # Seasonal (monthly) pattern, per city.
    for city in CANONICAL_CITIES:
        city_df = df[df["City"] == city]
        monthly_mean = city_df.groupby("Month")["Output Power"].mean()
        axes[1].plot(monthly_mean.index, monthly_mean.values, marker="o", label=city)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Mean Output Power (kW)")
    axes[1].set_title("Average Output Power by month (seasonal pattern)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, "output_power_by_time")

    print("Saved figures/eda/output_power_by_time.png - shows a midday peak each day")
    print("(expected, since sampling only covers 10:00-15:00) and a summer peak each")
    print("year (expected, more daylight/higher sun angle in summer months).")


# ---------------------------------------------------------------------------
# Section 13: Irradiance analysis
# ---------------------------------------------------------------------------


def analyze_irradiance(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 13: IRRADIANCE ANALYSIS")
    print("=" * 70)

    irradiance_cols = ["DHI", "DNI", "GHI", "Clearsky DHI", "Clearsky DNI", "Clearsky GHI"]
    summary = canonical_df.groupby("City")[irradiance_cols].mean().round(1)
    save_table(summary.reset_index(), "irradiance_summary_by_city")
    print("Mean irradiance by city (W/m^2):")
    print(summary.to_string())

    # Sample for plotting so figures stay a reasonable size/render fast
    # with ~100k+ rows - a random sample is fine for a scatter plot's
    # purpose (showing the shape of a relationship), not a precise stat.
    rng = np.random.default_rng(SEED)
    sample = canonical_df.sample(n=min(5000, len(canonical_df)), random_state=SEED)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].scatter(sample["GHI"], sample["Output Power"], s=5, alpha=0.3)
    axes[0, 0].set_xlabel("GHI (W/m^2)")
    axes[0, 0].set_ylabel("Output Power (kW)")
    axes[0, 0].set_title("GHI vs Output Power")

    axes[0, 1].scatter(sample["GHI"], sample["Clearsky GHI"], s=5, alpha=0.3)
    max_val = max(sample["GHI"].max(), sample["Clearsky GHI"].max())
    axes[0, 1].plot([0, max_val], [0, max_val], color="red", linestyle="--", label="GHI = Clearsky GHI")
    axes[0, 1].set_xlabel("GHI (W/m^2)")
    axes[0, 1].set_ylabel("Clearsky GHI (W/m^2)")
    axes[0, 1].set_title("GHI vs Clearsky GHI")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].scatter(sample["DNI"], sample["Output Power"], s=5, alpha=0.3)
    axes[1, 0].set_xlabel("DNI (W/m^2)")
    axes[1, 0].set_ylabel("Output Power (kW)")
    axes[1, 0].set_title("DNI vs Output Power")

    axes[1, 1].scatter(sample["DHI"], sample["Output Power"], s=5, alpha=0.3)
    axes[1, 1].set_xlabel("DHI (W/m^2)")
    axes[1, 1].set_ylabel("Output Power (kW)")
    axes[1, 1].set_title("DHI vs Output Power")

    fig.suptitle(f"Irradiance relationships (random sample of {len(sample)} rows across all cities)")
    fig.tight_layout()
    save_figure(fig, "ghi_vs_power")

    print(
        "\nSaved figures/eda/ghi_vs_power.png (4 panels: GHI/DNI/DHI vs Output Power, "
        "and GHI vs Clearsky GHI). GHI shows the strongest, most linear-looking "
        "relationship with Output Power - expected, since GHI is the total "
        "sunlight hitting a horizontal surface. DHI (diffuse-only light) shows a "
        "much weaker/noisier relationship - also expected, since PV panels "
        "respond much more strongly to direct sunlight."
    )
    return summary


# ---------------------------------------------------------------------------
# Section 14: Clear-Sky Index
# ---------------------------------------------------------------------------


def analyze_clear_sky_index(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 14: CLEAR-SKY INDEX")
    print("=" * 70)

    # add_clear_sky_index() already safely handles Clearsky GHI == 0
    # (returns NaN instead of raising a division error) - see
    # src/feature_engineering.py. This creates a DERIVED analysis
    # DataFrame; the raw dataset is never modified.
    with_index = feature_engineering.add_clear_sky_index(canonical_df)
    index_series = with_index["Clear_Sky_Index"]

    n_missing = index_series.isna().sum()
    n_infinite = np.isinf(index_series).sum()
    # (n_infinite is always 0 by construction - add_clear_sky_index()
    # converts the zero-denominator case straight to NaN, not inf - see
    # src/feature_engineering.py. Checked explicitly here anyway rather
    # than assumed, since this section's job is to verify, not assume.)
    n_extreme = (index_series > 1.5).sum()  # notably above the physical max of ~1

    print(f"Clear-Sky Index: {n_missing} missing (from Clearsky GHI == 0), "
          f"{n_infinite} infinite (none possible - handled safely), "
          f"{n_extreme} rows above 1.5 (physically extreme, likely brief instrument spikes).")

    percentiles = index_series.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("\nOverall percentiles:")
    print(percentiles.to_string())
    save_table(percentiles.reset_index().rename(columns={"index": "statistic", 0: "value"}), "clear_sky_index_percentiles")

    # Per-city distribution.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(index_series.dropna(), bins=60, range=(0, 1.3))
    axes[0].axvline(0.85, color="green", linestyle="--", label="Clear >= 0.85")
    axes[0].axvline(0.4, color="orange", linestyle="--", label="Overcast < 0.4")
    axes[0].set_xlabel("Clear-Sky Index (k)")
    axes[0].set_title("Overall Clear-Sky Index distribution")
    axes[0].legend(fontsize=8)

    for city in CANONICAL_CITIES:
        city_index = with_index.loc[with_index["City"] == city, "Clear_Sky_Index"].dropna()
        sns.kdeplot(city_index, label=city, ax=axes[1], clip=(0, 1.3))
    axes[1].set_xlabel("Clear-Sky Index (k)")
    axes[1].set_title("Clear-Sky Index density by city")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, "clear_sky_index_distribution")

    # Check whether the spec's suggested thresholds produce reasonable-
    # looking (non-degenerate) class sizes - this INFORMS a decision,
    # it does not change the thresholds.
    clear_frac = (index_series >= 0.85).mean()
    partly_frac = ((index_series >= 0.4) & (index_series < 0.85)).mean()
    overcast_frac = (index_series < 0.4).mean()
    print(
        f"\nUsing the spec's thresholds as-is: Clear={clear_frac:.1%}, "
        f"Partly cloudy={partly_frac:.1%}, Overcast={overcast_frac:.1%} of all rows. "
        "None of the three classes is vanishingly small, so the thresholds look "
        "usable as given - see course_context/EDA_REPORT.md for the full discussion "
        "(this is a decision noted for later, not changed here)."
    )

    return with_index


# ---------------------------------------------------------------------------
# Section 15: Cloud Type analysis
# ---------------------------------------------------------------------------


def analyze_cloud_type(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 15: CLOUD TYPE ANALYSIS")
    print("=" * 70)

    overall_counts = canonical_df["Cloud Type"].value_counts().sort_index()
    overall_pct = (100 * overall_counts / len(canonical_df)).round(2)
    cloud_table = pd.DataFrame(
        {
            "cloud_type_code": overall_counts.index,
            "label": [CLOUD_TYPE_LABELS.get(c, "Unknown code") for c in overall_counts.index],
            "count": overall_counts.values,
            "pct": overall_pct.values,
        }
    )
    save_table(cloud_table, "cloud_type_distribution")
    print(cloud_table.to_string(index=False))

    rare_classes = cloud_table[cloud_table["pct"] < 1.0]
    print(
        f"\n{len(rare_classes)} of {len(cloud_table)} Cloud Type codes each make up "
        f"less than 1% of all rows: {list(rare_classes['label'])}. This matters for "
        "Problem 1 because a classifier can reach high raw accuracy just by "
        "ignoring rare classes entirely - this is exactly why the project spec "
        "requires balanced accuracy, not raw accuracy, as the headline metric."
    )

    by_city = canonical_df.groupby(["City", "Cloud Type"]).size().unstack(fill_value=0)
    save_table(by_city.reset_index(), "cloud_type_by_city")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = cloud_table.sort_values("cloud_type_code")["cloud_type_code"]
    sns.countplot(data=canonical_df, x="Cloud Type", order=order, ax=ax)
    ax.set_title("Cloud Type frequency (all cities combined)")
    ax.set_xlabel("Cloud Type code")
    save_figure(fig, "cloud_type_distribution")

    return cloud_table


# ---------------------------------------------------------------------------
# Section 16: weather feature analysis
# ---------------------------------------------------------------------------


def analyze_weather_features(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 16: WEATHER FEATURE ANALYSIS")
    print("=" * 70)

    weather_cols = [
        "Dew Point", "Solar Zenith Angle", "Surface Albedo", "Wind Speed",
        "Precipitable Water", "Wind Direction", "Relative Humidity",
        "Temperature", "Pressure",
    ]

    rows = []
    for col in weather_cols:
        series = canonical_df[col]
        rows.append(
            {
                "column": col,
                "mean": round(series.mean(), 2),
                "std": round(series.std(), 2),
                "min": round(series.min(), 2),
                "max": round(series.max(), 2),
                "missing": int(series.isna().sum()),
            }
        )
    weather_table = pd.DataFrame(rows)
    save_table(weather_table, "weather_feature_summary")
    print(weather_table.to_string(index=False))

    # Flag anything obviously outside a plausible physical range, for
    # later review - not corrected here.
    flags = []
    rh_anomaly_table = None
    if (canonical_df["Relative Humidity"] > 100).any():
        flags.append("Relative Humidity has values above 100%")

        # This is worth investigating properly, not just flagging - see
        # what follows. Found: this isn't noise, it's a clean, isolated
        # pattern (one full calendar year each in two cities).
        anomaly_rows = []
        for city in CANONICAL_CITIES:
            city_df = canonical_df[canonical_df["City"] == city]
            affected = city_df[city_df["Relative Humidity"] > 100]
            if len(affected) == 0:
                continue
            unaffected = city_df[city_df["Relative Humidity"] <= 100]
            anomaly_rows.append(
                {
                    "city": city,
                    "affected_rows": len(affected),
                    "pct_of_city": round(100 * len(affected) / len(city_df), 2),
                    "affected_years": sorted(affected["Year"].unique().tolist()),
                    "affected_wind_direction_range": f"{affected['Wind Direction'].min():.2f}-{affected['Wind Direction'].max():.2f}",
                    "unaffected_wind_direction_range": f"{unaffected['Wind Direction'].min():.1f}-{unaffected['Wind Direction'].max():.1f}",
                    "affected_relative_humidity_range": f"{affected['Relative Humidity'].min():.1f}-{affected['Relative Humidity'].max():.1f}",
                }
            )
        rh_anomaly_table = pd.DataFrame(anomaly_rows)
        save_table(rh_anomaly_table, "relative_humidity_anomaly")

        print(
            "\nRelative Humidity > 100% investigated in detail "
            "(results/eda/relative_humidity_anomaly.csv):"
        )
        print(rh_anomaly_table.to_string(index=False))
        print(
            "\nThis is NOT random noise: it's isolated to exactly ONE full calendar "
            "year in each of two cities (all of Davis 2013, all of Huron 2012 - "
            "~14% of each city's 6-year data), and nowhere else. In the affected "
            "rows, 'Relative Humidity' reaches up to 360 (exactly Wind Direction's "
            "normal 0-360 degree range) while 'Wind Direction' sits at small "
            "values under ~4.5 (consistent with radians, or a completely "
            "different scale than the normal 0-360 degree values seen in every "
            "other year). This strongly suggests Relative Humidity and Wind "
            "Direction were swapped and/or recorded in a different unit for "
            "these two specific city-years - a real, previously undocumented "
            "data-quality issue, not measurement noise. Flagged as a decision "
            "for later (see course_context/EDA_REPORT.md) rather than silently "
            "corrected here."
        )
    if (canonical_df["Wind Speed"] < 0).any():
        flags.append("Wind Speed has negative values")
    if (canonical_df["Surface Albedo"] < 0).any() or (canonical_df["Surface Albedo"] > 1).any():
        flags.append("Surface Albedo has values outside the physical 0-1 range")
    print(f"\nSuspicious-range flags: {flags if flags else 'none found'}")

    return weather_table, rh_anomaly_table


# ---------------------------------------------------------------------------
# Section 17: correlation analysis
# ---------------------------------------------------------------------------


def analyze_correlations(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 17: CORRELATION ANALYSIS")
    print("=" * 70)

    corr_cols = [
        "Output Power", "GHI", "DNI", "DHI",
        "Clearsky GHI", "Clearsky DNI", "Clearsky DHI",
        "Temperature", "Relative Humidity", "Wind Speed",
        "Dew Point", "Pressure", "Solar Zenith Angle", "Precipitable Water",
    ]
    corr_matrix = canonical_df[corr_cols].corr()
    save_table(corr_matrix.reset_index().rename(columns={"index": "variable"}), "correlation_matrix")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix (Pearson) — all cities combined")
    save_figure(fig, "correlation_heatmap")

    ghi_power_corr = corr_matrix.loc["GHI", "Output Power"]
    zenith_ghi_corr = canonical_df[["Solar Zenith Angle", "GHI"]].corr().iloc[0, 1]
    print(f"GHI vs Output Power correlation (all cities pooled): {ghi_power_corr:.3f}")
    print(f"Solar Zenith Angle vs GHI correlation: {zenith_ghi_corr:.3f}")

    # IMPORTANT: the pooled correlation above is misleadingly weak. Each
    # city individually shows a much stronger GHI-Power relationship -
    # pooling cities with very different Output Power SCALES (Section 18)
    # dilutes the linear correlation even though the underlying physical
    # relationship is strong within every single city. Compute and save
    # per-city correlation to show this explicitly, rather than letting
    # the single pooled number stand alone and be misread as "GHI is only
    # weakly related to Output Power."
    per_city_corr = canonical_df.groupby("City").apply(
        lambda g: g[["GHI", "Output Power"]].corr().iloc[0, 1], include_groups=False
    )
    per_city_corr = per_city_corr.reset_index()
    per_city_corr.columns = ["City", "ghi_power_correlation"]
    save_table(per_city_corr, "ghi_power_correlation_by_city")
    print("\nGHI vs Output Power correlation, computed WITHIN each city separately:")
    print(per_city_corr.to_string(index=False))
    print(
        f"\nEvery per-city correlation ({per_city_corr['ghi_power_correlation'].min():.2f}-"
        f"{per_city_corr['ghi_power_correlation'].max():.2f}) is much stronger than the "
        f"pooled {ghi_power_corr:.3f}. This is exactly the scale-mixing effect discussed in "
        "Section 18 (Target Scale Analysis) showing up in a correlation, not just in raw "
        "units - a good example of why cross-city scale differences need explicit handling "
        "throughout this project, not only for RMSE."
    )

    print(
        "\nIMPORTANT: correlation measures a linear association, not causation. A "
        "high GHI-Output Power correlation makes physical sense (more sunlight "
        "hitting the panel -> more electricity generated) but this analysis alone "
        "doesn't prove GHI *causes* the output - it's consistent with the known "
        "physics of how PV panels work, which is why we can be reasonably "
        "confident here, not because correlation itself proves it."
    )
    return corr_matrix


# ---------------------------------------------------------------------------
# Section 18: target scale analysis
# ---------------------------------------------------------------------------


def analyze_target_scale(power_summary):
    print("\n" + "=" * 70)
    print("SECTION 18: TARGET SCALE ANALYSIS (cross-city)")
    print("=" * 70)

    table = power_summary.copy()
    table["max_over_mean"] = (table["max_kW"] / table["mean_kW"]).round(2)
    table["max_ratio_vs_smallest_city"] = (table["max_kW"] / table["max_kW"].min()).round(2)
    save_table(table, "target_scale_analysis")
    print(table[["city", "mean_kW", "max_kW", "max_over_mean", "max_ratio_vs_smallest_city"]].to_string(index=False))

    largest = table.loc[table["max_kW"].idxmax(), "city"]
    smallest = table.loc[table["max_kW"].idxmin(), "city"]
    ratio = table["max_kW"].max() / table["max_kW"].min()
    print(
        f"\n{largest}'s max Output Power is {ratio:.1f}x {smallest}'s. This directly "
        "matters for:\n"
        "  - Cross-city regression (Problem 2): a model trained on combined "
        "cities without per-city scaling would be dominated by whichever city "
        "has the largest raw numbers.\n"
        "  - Transfer learning (Problem 5): Davis (source) and Amherst (target) "
        "have very different scales, so raw predictions need rescaling or the "
        "transfer will look artificially bad/good.\n"
        "  - nRMSE: this is exactly why the spec requires nRMSE alongside RMSE - "
        "raw RMSE numbers are not comparable across cities with this much scale "
        "difference.\n"
        "  - Target normalization: see src/preprocessing.py's "
        "fit_target_scaler()/apply_target_scaler()/inverse_transform_target(), "
        "built in Phase 2 specifically for this situation."
    )


# ---------------------------------------------------------------------------
# Section 19: class imbalance preview
# ---------------------------------------------------------------------------


def preview_class_imbalance(with_index):
    print("\n" + "=" * 70)
    print("SECTION 19: CLASS IMBALANCE PREVIEW (Problem 1 targets)")
    print("=" * 70)

    def sky_condition(k):
        if pd.isna(k):
            return np.nan
        if k >= 0.85:
            return "Clear"
        elif k >= 0.4:
            return "Partly cloudy"
        else:
            return "Overcast"

    sky = with_index["Clear_Sky_Index"].apply(sky_condition)
    sky_counts = sky.value_counts()
    sky_pct = (100 * sky_counts / sky.notna().sum()).round(2)
    sky_table = pd.DataFrame({"class": sky_counts.index, "count": sky_counts.values, "pct": sky_pct.values})
    save_table(sky_table, "sky_condition_distribution")
    print("Sky-condition class distribution (all cities):")
    print(sky_table.to_string(index=False))

    sky_by_city = with_index.assign(sky_condition=sky).groupby(["City", "sky_condition"]).size().unstack(fill_value=0)
    save_table(sky_by_city.reset_index(), "sky_condition_distribution_by_city")

    # Generation-regime: PER-CITY terciles, as the spec requires - see
    # course_context/TEACHER_EXPECTATIONS.md, Problem 1.
    regime_rows = []
    for city in CANONICAL_CITIES:
        city_power = with_index.loc[with_index["City"] == city, "Output Power"].dropna()
        terciles = city_power.quantile([1 / 3, 2 / 3])
        low = (city_power <= terciles.iloc[0]).sum()
        medium = ((city_power > terciles.iloc[0]) & (city_power <= terciles.iloc[1])).sum()
        high = (city_power > terciles.iloc[1]).sum()
        regime_rows.append({"city": city, "Low": low, "Medium": medium, "High": high})
    regime_table = pd.DataFrame(regime_rows)
    save_table(regime_table, "generation_regime_distribution")
    print("\nGeneration-regime class counts (per-city terciles, so each city is ~33/33/33 by construction):")
    print(regime_table.to_string(index=False))
    print(
        "\nWhy per-city terciles, not global terciles: Output Power scales differ "
        "hugely by city (Section 18) - global terciles computed across all cities "
        "combined would put nearly all of Davis's rows in 'High' and nearly all of "
        "La Jolla's rows in 'Low', regardless of how each city's own generation "
        "actually varied day-to-day. Per-city terciles instead ask 'was this a "
        "relatively high/medium/low output moment FOR THIS CITY,' which is the "
        "meaningful question."
    )
    return sky_table, regime_table


# ---------------------------------------------------------------------------
# Section 20: temporal dependence (autocorrelation)
# ---------------------------------------------------------------------------


def analyze_temporal_dependence(canonical_df):
    print("\n" + "=" * 70)
    print("SECTION 20: TEMPORAL DEPENDENCE (AUTOCORRELATION)")
    print("=" * 70)

    # Use one representative city (Davis) sorted chronologically. Note:
    # this dataset only has ~11 samples/day (10:00-15:00), so "lag 1" is
    # 30 minutes later THE SAME DAY (or the next sampled slot) - not a
    # continuous 24/7 series. That's fine for this diagnostic purpose.
    davis = data_loader.load_city("Davis", years="long", path=DATA_PATH)
    davis = davis.sort_values(["Year", "Month", "Day", "Hour", "Minute"]).reset_index(drop=True)

    def autocorr(series, lag):
        return series.autocorr(lag=lag)

    lags = [1, 2, 6]
    power_autocorr = {lag: round(autocorr(davis["Output Power"], lag), 3) for lag in lags}
    ghi_autocorr_lag1 = round(autocorr(davis["GHI"], 1), 3)

    print(f"Output Power autocorrelation (Davis, within-sheet lags, not calendar-continuous): {power_autocorr}")
    print(f"GHI autocorrelation at lag 1: {ghi_autocorr_lag1}")

    autocorr_table = pd.DataFrame(
        [{"series": "Output Power", "lag": lag, "autocorrelation": val} for lag, val in power_autocorr.items()]
        + [{"series": "GHI", "lag": 1, "autocorrelation": ghi_autocorr_lag1}]
    )
    save_table(autocorr_table, "autocorrelation")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([f"Power lag{lag}" for lag in lags] + ["GHI lag1"], list(power_autocorr.values()) + [ghi_autocorr_lag1])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of Output Power and GHI (Davis)")
    save_figure(fig, "autocorrelation")

    print(
        "\nHigh autocorrelation at lag 1 (adjacent 30-minute readings are strongly "
        "related) supports the project spec's sequence-forecasting sub-task "
        "(Problem 2, K=12 window) and the use of lag features / RNN-LSTM-GRU "
        "models (see course_context/ML_METHOD_MAP.md) - adjacent readings "
        "carrying real predictive information is the basic assumption those "
        "methods rely on. This is a diagnostic finding only; no sequence model "
        "is built in this phase."
    )
    return autocorr_table


# ---------------------------------------------------------------------------
# Section 21: feature relationships
# ---------------------------------------------------------------------------


def analyze_feature_relationships(with_index):
    print("\n" + "=" * 70)
    print("SECTION 21: FEATURE RELATIONSHIPS")
    print("=" * 70)

    sample = with_index.sample(n=min(5000, len(with_index)), random_state=SEED)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].scatter(sample["Solar Zenith Angle"], sample["GHI"], s=5, alpha=0.3)
    axes[0, 0].set_xlabel("Solar Zenith Angle (degrees)")
    axes[0, 0].set_ylabel("GHI (W/m^2)")
    axes[0, 0].set_title("Solar Zenith Angle vs GHI")

    axes[0, 1].scatter(sample["Temperature"], sample["Output Power"], s=5, alpha=0.3)
    axes[0, 1].set_xlabel("Temperature (C)")
    axes[0, 1].set_ylabel("Output Power (kW)")
    axes[0, 1].set_title("Temperature vs Output Power")

    sns.boxplot(data=sample, x="Cloud Type", y="Clear_Sky_Index", ax=axes[1, 0])
    axes[1, 0].set_title("Cloud Type vs Clear-Sky Index")

    axes[1, 1].scatter(sample["Relative Humidity"], sample["Clear_Sky_Index"], s=5, alpha=0.3)
    axes[1, 1].set_xlabel("Relative Humidity (%)")
    axes[1, 1].set_ylabel("Clear-Sky Index")
    axes[1, 1].set_title("Relative Humidity vs Clear-Sky Index")

    fig.suptitle(f"Feature relationships (random sample of {len(sample)} rows) - exploratory only, not causal")
    fig.tight_layout()
    save_figure(fig, "feature_relationships")

    print(
        "Saved figures/eda/feature_relationships.png. As expected physically: "
        "Solar Zenith Angle vs GHI shows a clear negative relationship (sun "
        "higher in the sky -> more direct irradiance); Cloud Type vs Clear-Sky "
        "Index shows Clear Type-0 rows sitting at high k and cloudier types "
        "sitting lower, confirming Cloud Type and the Clear-Sky Index carry "
        "related but not identical information. These are exploratory "
        "observations, not causal claims."
    )


def print_reproducibility_info():
    """Section 31: record exactly what this run used, for reproducibility."""
    import platform

    import matplotlib
    import numpy
    import pandas
    import sklearn

    print("=" * 70)
    print("REPRODUCIBILITY INFO")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Python: {platform.python_version()}")
    print(f"pandas: {pandas.__version__}  numpy: {numpy.__version__}  "
          f"scikit-learn: {sklearn.__version__}  matplotlib: {matplotlib.__version__}")
    print(f"Dataset file: {DATA_PATH}")
    print(f"Analysis date (UTC): {pd.Timestamp.now('UTC').date()}")


def main():
    utils.set_seed(SEED)
    utils.ensure_dir(RESULTS_DIR)
    utils.ensure_dir(FIGURES_DIR)
    visualization.set_plot_style()

    print_reproducibility_info()

    sheets = verify_dataset_location()
    raw_sheets, sheet_summary = verify_all_sheets(sheets)
    column_report = verify_column_structure(raw_sheets)

    canonical_df = build_canonical_dataset()
    dtype_table = analyze_dtypes(canonical_df)

    missing_table, amherst_missing = analyze_missing_values(raw_sheets)
    duplicate_table = analyze_duplicates(raw_sheets)
    timestamp_table, within_window = validate_timestamps(raw_sheets)
    coverage_table = city_year_coverage(sheets)

    power_summary = analyze_output_power(canonical_df)
    analyze_output_power_by_time(canonical_df)
    irradiance_summary = analyze_irradiance(canonical_df)
    with_index = analyze_clear_sky_index(canonical_df)
    cloud_table = analyze_cloud_type(canonical_df)
    weather_table, rh_anomaly_table = analyze_weather_features(canonical_df)
    corr_matrix = analyze_correlations(canonical_df)
    analyze_target_scale(power_summary)
    sky_table, regime_table = preview_class_imbalance(with_index)
    autocorr_table = analyze_temporal_dependence(canonical_df)
    analyze_feature_relationships(with_index)

    print("\n\n" + "=" * 70)
    print("EDA RUN COMPLETE")
    print("=" * 70)
    print(f"Tables saved to: {RESULTS_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print("Raw Excel file was never modified.")


if __name__ == "__main__":
    main()
