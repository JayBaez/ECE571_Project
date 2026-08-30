"""
features.py (Problem 1 — Supervised Classification)

Defines which columns are allowed as features for each task, and the
feature groups used for the ablation study (Section 22 of the Phase 4
instructions). Every exclusion here is cross-referenced against
course_context/LEAKAGE_MAP.md - nothing here is invented independently
of that document.

Feature-engineering columns (Clear_Sky_Index, Hour_sin/cos,
Month_sin/cos, DayOfYear_sin/cos) are added by
src/feature_engineering.py before these lists are used to select
columns - see run_experiments.py.
"""

# ---------------------------------------------------------------------------
# Task A: Sky-condition
# ---------------------------------------------------------------------------

# MUST NOT use - these directly define the label (k = GHI / Clearsky GHI).
SKY_CONDITION_LABEL_COLUMNS = ["GHI", "Clearsky GHI", "Clear_Sky_Index"]

# RESOLVED decision (course_context/LEAKAGE_MAP.md, TEACHER_EXPECTATIONS.md):
# excluded from the PRIMARY model because DHI+DNI+Solar Zenith Angle can
# approximately reconstruct GHI (GHI ~= DNI*cos(zenith) + DHI), and
# Solar Zenith Angle alone correlates with GHI at -0.74 (verified,
# course_context/EDA_REPORT.md).
SKY_CONDITION_SECONDARY_LEAKAGE_RISK_COLUMNS = ["DHI", "DNI", "Solar Zenith Angle"]

# Columns safe to use for sky-condition in the PRIMARY model.
SKY_CONDITION_SAFE_WEATHER_COLUMNS = [
    "Cloud Type", "Dew Point", "Surface Albedo", "Wind Speed",
    "Precipitable Water", "Wind Direction", "Relative Humidity",
    "Temperature", "Pressure",
]
SKY_CONDITION_TIME_COLUMNS = ["Hour_sin", "Hour_cos", "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"]

# ---------------------------------------------------------------------------
# Task B: Generation-regime
# ---------------------------------------------------------------------------

# MUST NOT use - current Output Power is literally the label.
GENERATION_REGIME_LABEL_COLUMNS = ["Output Power"]

# No leakage restriction on GHI/Clearsky GHI here - Output Power isn't
# derived from a fixed rule applied to them (unlike Task A), it's the
# actual physical generation reading, so irradiance IS a legitimate
# predictive feature.
GENERATION_REGIME_SAFE_COLUMNS = [
    "GHI", "DNI", "DHI", "Clearsky GHI", "Clearsky DNI", "Clearsky DHI",
    "Cloud Type", "Dew Point", "Solar Zenith Angle", "Surface Albedo",
    "Wind Speed", "Precipitable Water", "Wind Direction",
    "Relative Humidity", "Temperature", "Pressure",
]
GENERATION_REGIME_TIME_COLUMNS = SKY_CONDITION_TIME_COLUMNS

CATEGORICAL_COLUMNS = ["Cloud Type"]


def get_feature_columns(task: str, ablation_group: str = "full", include_leakage_risk: bool = False) -> list:
    """
    Return the list of feature-engineered column names to use for one
    task and one ablation group.

    Parameters
    ----------
    task : str
        "sky_condition" or "generation_regime".
    ablation_group : str
        "time_only" - only the cyclical time features.
        "weather_only" - only the non-irradiance weather columns
            (Temperature, Humidity, Wind, Dew Point, Pressure,
            Albedo, Precipitable Water) - no Cloud Type, no irradiance.
        "irradiance_weather" - weather columns + irradiance columns
            (for generation_regime; for sky_condition this is the same
            as "full" minus time features, since sky_condition's
            "irradiance" columns ARE the leakage-risk ones).
        "full" - every safe column (the primary, non-ablation feature
            set actually used for the main results).
    include_leakage_risk : bool
        Only meaningful for task="sky_condition". If True, adds back
        `DHI`/`DNI`/`Solar Zenith Angle` - used ONLY for the explicitly
        labeled secondary leakage-demonstration ablation, never for
        the primary reported model.

    Returns
    -------
    list of str

    Raises
    ------
    ValueError
        For an unknown task or ablation_group.
    """
    if task == "sky_condition":
        weather = list(SKY_CONDITION_SAFE_WEATHER_COLUMNS)
        time_cols = list(SKY_CONDITION_TIME_COLUMNS)
        leakage_risk = list(SKY_CONDITION_SECONDARY_LEAKAGE_RISK_COLUMNS) if include_leakage_risk else []

        if ablation_group == "time_only":
            return time_cols
        elif ablation_group == "weather_only":
            # For sky-condition, "weather_only" excludes Cloud Type too
            # (Cloud Type is itself a cloud/sky observation, closer to
            # irradiance-family information than plain weather).
            return [c for c in weather if c != "Cloud Type"]
        elif ablation_group == "irradiance_weather":
            return weather + leakage_risk
        elif ablation_group == "full":
            return weather + time_cols + leakage_risk
        else:
            raise ValueError(f"Unknown ablation_group '{ablation_group}'")

    elif task == "generation_regime":
        weather = [c for c in GENERATION_REGIME_SAFE_COLUMNS if c not in
                   ["GHI", "DNI", "DHI", "Clearsky GHI", "Clearsky DNI", "Clearsky DHI"]]
        irradiance = ["GHI", "DNI", "DHI", "Clearsky GHI", "Clearsky DNI", "Clearsky DHI"]
        time_cols = list(GENERATION_REGIME_TIME_COLUMNS)

        if ablation_group == "time_only":
            return time_cols
        elif ablation_group == "weather_only":
            return [c for c in weather if c != "Cloud Type"]
        elif ablation_group == "irradiance_weather":
            return weather + irradiance
        elif ablation_group == "full":
            return weather + irradiance + time_cols
        else:
            raise ValueError(f"Unknown ablation_group '{ablation_group}'")

    else:
        raise ValueError(f"Unknown task '{task}'. Use 'sky_condition' or 'generation_regime'.")
