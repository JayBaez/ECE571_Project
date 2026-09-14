"""
features.py (Problem 3 — Dimension Reduction)

Defines the SINGLE, unified feature set that PCA and the autoencoder
are fit on. This feature set is then reused for BOTH downstream tasks
(Problem 1's sky-condition classifier and Problem 2's Output Power
regressor), so the central comparison table (Section 26) can show one
"PCA-2" row with both a classification and a regression column, rather
than two different "PCA-2"s built from two different inputs.

WHY THIS FEATURE SET, SPECIFICALLY (an important, deliberate decision -
not just reusing whatever was easiest): it's Problem 1's sky-condition
LEAKAGE-SAFE feature set (course_context/PROBLEM1_REPORT.md,
course_context/LEAKAGE_MAP.md) - Cloud Type, weather columns, and
cyclical time features, with NO irradiance-family columns (GHI, DNI,
DHI, Clearsky GHI/DNI/DHI, Solar Zenith Angle).

This matters for a reason that goes beyond Problem 1 alone: if PCA/the
autoencoder were fit on a feature set THAT INCLUDES GHI etc., the
resulting components would likely capture mostly GHI-driven variance
(GHI dominates this dataset's variance - course_context/EDA_REPORT.md
correlation section), and using those components as classifier inputs
would functionally leak the sky-condition label through the back door,
the same way Problem 1's DHI/DNI/Solar-Zenith-Angle ablation proved
directly (balanced accuracy jumped from ~0.74-0.81 to ~0.98 when those
columns were included). Reusing the leakage-safe set keeps Problem 3
consistent with that established, empirically-justified policy,
applied to BOTH downstream tasks for one fair, unified representation.

TRADEOFF (documented explicitly, not hidden): Problem 3's regression
"raw" baseline is therefore NOT the same as Problem 2's headline
Davis result (which legitimately used the fuller feature set,
including irradiance - Output Power has no leakage restriction on
irradiance features). Problem 3's raw regression baseline will be
weaker, because it deliberately excludes irradiance features too, for
this one fair, shared comparison. See course_context/PROBLEM3_REPORT.md
for the full discussion.
"""

from problems.problem1_classification.features import (
    CATEGORICAL_COLUMNS,
    SKY_CONDITION_SAFE_WEATHER_COLUMNS,
    SKY_CONDITION_TIME_COLUMNS,
)

# The unified, leakage-safe feature set used for PCA/autoencoder input
# AND for the "raw" baseline in both downstream comparisons.
UNIFIED_FEATURE_COLUMNS = list(SKY_CONDITION_SAFE_WEATHER_COLUMNS) + list(SKY_CONDITION_TIME_COLUMNS)


def get_feature_columns() -> list:
    """
    Return the unified (pre-encoding) feature-column list used
    throughout Problem 3.

    Returns
    -------
    list of str
    """
    return list(UNIFIED_FEATURE_COLUMNS)
