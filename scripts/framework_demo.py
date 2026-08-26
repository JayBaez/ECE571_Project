"""
framework_demo.py

============================================================
 THIS IS A FRAMEWORK TEST / DEMO. IT IS NOT A PROJECT RESULT.
============================================================

This script proves that the Phase 2 framework works end-to-end:

    config -> synthetic data -> clean -> features -> split
    -> preprocess -> train (plain Linear Regression) -> evaluate
    -> save results/figures

It uses a small SYNTHETIC dataset generated in this script (NOT the
real project Excel file), and a deliberately simple model (Linear
Regression), because its only job is to prove the plumbing works -
not to produce a meaningful number. Every result this script produces
is saved under "framework_demo", never under "problem1".."problem5", so
it can never be confused with (or accidentally picked up by a
leaderboard query for) a real Problem 1-5 result.

Usage (from the project root):
    python scripts/framework_demo.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import cleaning, evaluation, experiment_runner, feature_engineering, preprocessing, splitting, utils, visualization

DEMO_PROBLEM_NAME = "framework_demo"
DEMO_SEED = 42


def make_synthetic_dataset(n_days: int = 60, seed: int = DEMO_SEED) -> pd.DataFrame:
    """
    Build a small, fully synthetic dataset that LOOKS like a
    simplified version of the real project data (same kind of
    columns: timestamps, GHI/Clearsky GHI, weather, Cloud Type,
    Output Power) but contains no real project data whatsoever -
    every value here is randomly generated.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    hours = [10, 12, 14]

    rows = []
    for date in dates:
        seasonal_factor = np.sin(date.dayofyear / 365 * np.pi)  # rough summer/winter pattern
        clearsky_ghi = 500 + 300 * seasonal_factor
        for hour in hours:
            cloud_factor = rng.uniform(0.5, 1.0)
            ghi = clearsky_ghi * cloud_factor
            temperature = 15 + 10 * seasonal_factor + rng.normal(0, 2)
            wind_speed = rng.uniform(0, 5)
            cloud_type = int(rng.choice([0, 1, 2, 3]))
            output_power = 0.2 * ghi + rng.normal(0, 5)  # simple, noisy linear-ish relationship

            rows.append(
                {
                    "Year": date.year,
                    "Month": date.month,
                    "Day": date.day,
                    "Hour": hour,
                    "Minute": 0,
                    "GHI": ghi,
                    "Clearsky GHI": clearsky_ghi,
                    "Temperature": temperature,
                    "Wind Speed": wind_speed,
                    "Cloud Type": cloud_type,
                    "Output Power": output_power,
                }
            )

    df = pd.DataFrame(rows)
    # Inject one missing value on purpose, so the demo actually
    # exercises the cleaning step instead of trivially having nothing
    # to clean.
    df.loc[5, "Output Power"] = np.nan
    return df


def main():
    print("=" * 70)
    print(" FRAMEWORK DEMO - NOT A PROJECT RESULT")
    print(" (synthetic data, a trivial Linear Regression model -")
    print("  this only proves the pipeline works end-to-end)")
    print("=" * 70)

    utils.set_seed(DEMO_SEED)
    total_steps = 6

    utils.log_step(1, total_steps, "Loading data (synthetic, in-memory)...")
    df = make_synthetic_dataset()
    print(f"    Generated {len(df)} synthetic rows.")

    utils.log_step(2, total_steps, "Cleaning data...")
    cleaned_df, cleaning_report = cleaning.clean_sheet(
        df, target_column="Output Power", missing_strategy="interpolate", verbose=True
    )

    utils.log_step(3, total_steps, "Building features...")
    featured_df = feature_engineering.add_feature_groups(
        cleaned_df, groups=["clear_sky_index", "time_cyclical"]
    )
    print(f"    Columns after feature engineering: {list(featured_df.columns)}")

    utils.log_step(4, total_steps, "Splitting data (chronological 80/20)...")
    train_df, test_df, split_info = splitting.chronological_split(featured_df, train_frac=0.8)
    print(f"    Train rows: {split_info['n_train']}, Test rows: {split_info['n_test']}")
    assert splitting.verify_no_overlap(train_df, test_df), "demo split leaked rows - this should never happen"

    utils.log_step(5, total_steps, "Preprocessing and training model...")
    numeric_columns = preprocessing.get_numeric_columns(train_df, exclude=["Output Power"])
    categorical_columns = preprocessing.get_categorical_columns(train_df)

    preprocessor = preprocessing.fit_preprocessor(train_df, numeric_columns, categorical_columns)
    train_processed = preprocessing.apply_preprocessor(train_df, preprocessor)
    test_processed = preprocessing.apply_preprocessor(test_df, preprocessor)

    X_train, y_train, feature_columns = preprocessing.prepare_xy(train_processed, target_column="Output Power")
    X_test, y_test, _ = preprocessing.prepare_xy(test_processed, target_column="Output Power")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    utils.log_step(6, total_steps, "Evaluating and saving results...")
    metrics = evaluation.regression_metrics(y_test, y_pred)
    print(f"    Demo metrics (synthetic data, not meaningful): {metrics}")

    experiment_id = experiment_runner.generate_experiment_id(
        problem=DEMO_PROBLEM_NAME, model="linear_regression", dataset="synthetic", seed=DEMO_SEED
    )
    experiment_dir = experiment_runner.create_experiment_dir(DEMO_PROBLEM_NAME, experiment_id)

    experiment_runner.save_metrics_json(metrics, experiment_dir)
    experiment_runner.save_predictions_csv(y_test, y_pred, experiment_dir)
    experiment_runner.save_experiment_config(
        {
            "problem": DEMO_PROBLEM_NAME,
            "model": "linear_regression",
            "seed": DEMO_SEED,
            "feature_columns": feature_columns,
            "cleaning_report": cleaning_report,
            "split_info": {k: v for k, v in split_info.items()},
        },
        experiment_dir,
    )

    figure_path = visualization.build_figure_path(DEMO_PROBLEM_NAME, experiment_id, "prediction_vs_truth.png")
    visualization.plot_prediction_vs_truth(
        y_test, y_pred, save_path=figure_path, title="FRAMEWORK DEMO - NOT A PROJECT RESULT"
    )

    experiment_runner.save_result(
        {
            "experiment_id": experiment_id,
            "timestamp": pd.Timestamp.now("UTC").isoformat(),
            "problem": DEMO_PROBLEM_NAME,
            "model": "linear_regression",
            "dataset": "synthetic",
            "source_city": "",
            "target_city": "",
            "seed": DEMO_SEED,
            "parameters": "{}",
            "metric": "rmse",
            "score": metrics["rmse"],
            "runtime_seconds": "",
            "notes": "FRAMEWORK DEMO ONLY - synthetic data, not a real project result.",
        }
    )

    print(f"\n    Saved artifacts to: {experiment_dir}")
    print(f"    Saved figure to: {figure_path}")
    print("\n" + "=" * 70)
    print(" FRAMEWORK DEMO COMPLETE - again, NOT a project result.")
    print(" This only confirms the pipeline runs end-to-end correctly.")
    print("=" * 70)


if __name__ == "__main__":
    main()
