# My Project Cheat Sheet

Read this right before presenting.

**PROJECT GOAL**
Predict solar (photovoltaic) power output from weather data, across 5
cities, using 5 different ML paradigms — and honestly report which
ones actually help.

**DATASET**
5 cities (Davis, Huron, Santa Barbara, La Jolla, Amherst), 30-min
readings, mostly 2011-2016 (Amherst: 2018-2020), 22 columns
(irradiance, weather, Cloud Type, Output Power). Davis's plant is
~2.7-3.5x the average output of the smaller cities — a scale
difference, not a weather difference.

**BEST MODEL / RESULTS**
Gradient boosting, Davis same-city regression: **RMSE=15.17 kW,
R²=0.953** — the strongest result in the whole project.

**PROBLEM 1 (Classification):**
Sky-condition & generation-regime from weather. Best: logistic
regression, 0.772 balanced accuracy. GHI excluded — it defines the
label.

**PROBLEM 2 (Regression — core task):**
Output Power in kW. Best: gradient boosting, RMSE=15.17 kW. Cross-city
zero-shot failed (scale mismatch); GRU sequence model beat persistence.

**PROBLEM 3 (Dimension Reduction):**
PCA + autoencoder at d=2/5/10. Raw features beat every compressed
version, every time. Traced to Cloud Type compressing poorly.

**PROBLEM 4 (Semi-Supervised Learning):**
10%/30%/50% labels + pseudo-labeling. Did NOT help — pseudo-labels
were 100% accurate but redundant (all the easy "Clear" class).

**PROBLEM 5 (Transfer Learning):**
Davis → Amherst. Transfer beat few-shot at every k, +21.4% at k=10.
Found and fixed a real negative-transfer bug along the way (bad
learning rate).

**BIGGEST FINDING:**
The Problem 5 learning-rate story — same setup, severe failure (140.9
kW) or clear win (27.2 kW) depending only on the fine-tuning learning
rate. Found it, diagnosed it, fixed it.

**BIGGEST LIMITATION:**
Deliberately small hyperparameter searches throughout — found
repeatedly that tuning barely moved results, but never confirmed with
a larger search.

**WHY CHRONOLOGICAL SPLIT:**
It's time-series data — random shuffling would let the model train on
future data and get tested on the past, making every metric
meaningless for real forecasting.

**WHY RMSE + MAE:**
RMSE penalizes big misses more (squares errors first); MAE is the
plain average error size. Reporting both gives a fuller picture than
either alone.

**WHY MULTIPLE SEEDS:**
To check results aren't a fluke of one random initialization — every
result is mean ± std across 3 seeds (42, 123, 2026).

**WHY CROSS-CITY:**
To test whether a model trained on one city's weather-power
relationship generalizes anywhere else — turns out it does, but only
once you account for the output-scale difference.

**WHY TRANSFER LEARNING:**
New or small solar installations rarely have years of data. Transfer
learning tests whether a data-rich city (Davis) can help a data-poor
one (Amherst) — it can, especially with very few target samples.

**WHY SEMI-SUPERVISED LEARNING:**
Labeling data is expensive; unlabeled weather readings are cheap.
Tests whether a model can use unlabeled data to make up for having few
labels — here, it couldn't, and I explained precisely why.

**WHY DIMENSION REDUCTION:**
Tests whether the feature space can be compressed without losing
useful information — useful in general for efficiency/visualization,
but here it turned out to hurt, which is itself a real finding.

**AI ASSISTANCE:**
Used throughout for code generation, debugging, and drafting the
report/presentation — every result was actually run and verified, and
every methodological decision (target definitions, leakage rules,
which experiments to run) was mine. Full disclosure in
`report/FINAL_REPORT.md`, Section 15.
