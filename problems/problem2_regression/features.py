"""
features.py (Problem 2 — Supervised Regression)

Defines which columns are allowed as features for predicting Output
Power. Unlike Problem 1's sky-condition task, there's no leakage-risk
ablation needed here: Output Power isn't derived from a fixed rule
applied to GHI/DHI/DNI/etc. the way the sky-condition label was, so
every irradiance and weather column is a legitimate predictor - see
course_context/LEAKAGE_MAP.md, Problem 2.

The only column that must never appear as a feature is the current-
timestep Output Power itself (that's the target). Lagged Output Power
(t-1, t-2, ...) is legitimate ONLY for the sequence-forecasting
sub-task, where the setup explicitly assumes the previous K steps are
available at forecast time - see sequence.py.
"""

TARGET_COLUMN = "Output Power"

CATEGORICAL_COLUMNS = ["Cloud Type"]

WEATHER_COLUMNS = [
    "Dew Point", "Surface Albedo", "Wind Speed", "Precipitable Water",
    "Wind Direction", "Relative Humidity", "Temperature", "Pressure",
]
IRRADIANCE_COLUMNS = [
    "GHI", "DNI", "DHI", "Clearsky GHI", "Clearsky DNI", "Clearsky DHI",
    "Solar Zenith Angle",
]
TIME_COLUMNS = ["Hour_sin", "Hour_cos", "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"]

# The primary, non-sequence feature set: every legitimate weather and
# irradiance column, Cloud Type (one-hot encoded), plus cyclical time
# features. This is what every same-city, cross-city, and 3yr-vs-6yr
# experiment uses.
PRIMARY_FEATURE_COLUMNS = WEATHER_COLUMNS + IRRADIANCE_COLUMNS + CATEGORICAL_COLUMNS + TIME_COLUMNS


def get_feature_columns() -> list:
    """
    Return the primary feature-column list for the non-sequence
    regression experiments (same-city, cross-city, 3yr-vs-6yr).

    Returns
    -------
    list of str
    """
    return list(PRIMARY_FEATURE_COLUMNS)
