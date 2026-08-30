# Problem 1 Report — Supervised Classification

Internal technical record for Problem 1. Every number below comes from
an actual executed run of `problems/problem1_classification/
run_experiments.py` against the real dataset — nothing is estimated.
Full results: `results/problem1/problem1_results.csv` (198 rows).
Reproducibility: seeds 42/123/2026, Python 3.12.3, pandas 3.0.2,
scikit-learn 1.8.0, torch 2.13.0, run 2026-08-30.

---

## 1. Objective

Predict two 3-class targets from meteorological/irradiance features:
sky-condition (Clear/Partly Cloudy/Overcast) and generation-regime
(Low/Medium/High Output Power), for Davis and Amherst, using a
reasonable classical → deep-learning model progression.

## 2. Dataset used

Davis (`'11-'16`, 24,112 rows) and Amherst (`'18-'20`, 12,056 rows,
12,052 after dropping 4 rows with missing Output Power — see Section
8). Both loaded via `src/data_loader.py`'s `load_city(city, years="long")`.

## 3. Classification tasks

**Task A (sky-condition):** `k = GHI / Clearsky GHI` → Clear (k≥0.85) /
Partly Cloudy (0.4≤k<0.85) / Overcast (k<0.4). Thresholds used exactly
as specified. **Documented edge case:** k can exceed 1.0 (verified,
`EDA_REPORT.md`); such rows are labeled Clear (open-ended upper bound),
not a separate class — a judgment call, not a threshold change.

**Task B (generation-regime):** per-city Output Power terciles,
**fit on training data only, applied to test** (see Section 6).

## 4. Target construction

`problems/problem1_classification/targets.py`. Sky-condition uses a
fixed threshold rule (no data-dependent fitting, so labeling before or
after the split is equivalent — done before, for simplicity).
Generation-regime terciles ARE data-dependent, so they're fit strictly
after the chronological split, on the training portion only.

## 5. Feature selection

`problems/problem1_classification/features.py`. **Task A primary
model:** Cloud Type, Dew Point, Surface Albedo, Wind Speed,
Precipitable Water, Wind Direction, Relative Humidity, Temperature,
Pressure, + 6 cyclical time features (23 columns after one-hot
encoding Cloud Type for Davis, 24 for Amherst — different category
counts per city). **Task B:** all of the above **plus** GHI/DNI/DHI/
Clearsky-family columns (legitimate for this task — Output Power isn't
derived from a fixed rule applied to them) — 30/31 columns.

## 6. Leakage prevention

- **Task A:** `GHI`/`Clearsky GHI` excluded (define the label).
  **Resolved decision, empirically confirmed this phase (Section 18):**
  `DHI`/`DNI`/`Solar Zenith Angle` also excluded from the primary
  model — a secondary ablation including them pushed balanced accuracy
  from 0.74–0.81 to **0.98–0.99**, dramatically confirming the leakage
  risk was real, not theoretical.
- **Task B:** current `Output Power` excluded (it's the label). No lag
  features used (not part of this task's setup per
  `course_context/LEAKAGE_MAP.md`).
- **Tercile fitting:** train-only, verified — see Section 6's result
  below for what happened when applied to test.
- **Test-distribution check (as required):** Davis's test-set
  generation-regime distribution stayed close to even; **Amherst's
  test set skewed to Medium 38.1% / High 34.6% / Low 27.3%** instead
  of 33/33/33 — train-fit boundaries don't perfectly generalize to a
  chronologically later, seasonally different test period. Documented,
  not corrected (correcting it would itself be a leak).

## 7. Train/test split

Chronological 80/20 per city via `src/splitting.py`'s
`chronological_split()`. Davis: 19,289 train / 4,823 test. Amherst
(Task B): 9,641 train / 2,411 test (4 fewer than Task A due to the
dropped missing-target rows). Split indices/timestamps are returned by
`chronological_split()` and available in each run's saved config.

## 8. Preprocessing

`StandardScaler` (numeric) and `OneHotEncoder` (Cloud Type), both fit
on training data only, via `src/preprocessing.py`'s
`fit_preprocessor()`/`apply_preprocessor()` (Phase 2 framework, reused
as-is). **Missing values:** Task B's 4 missing Amherst `Output Power`
rows were **dropped, not interpolated** — interpolating a value and
then treating it as a real class label would fabricate ground truth
for a classifier, unlike for a regression target. Task A needed no
missing-value handling (`GHI`/`Clearsky GHI` have zero missing values,
verified `DATASET_PROFILE.md`). Raw Excel file confirmed unmodified
throughout (checked via `git status` after every run).

## 9. Models tested

Majority-class baseline, Logistic Regression, Decision Tree, Random
Forest, Gradient Boosting, and a small MLP (2 hidden layers, ReLU,
dropout, `CrossEntropyLoss`, early stopping) — `problems/
problem1_classification/models.py`. Plus tuned variants of Random
Forest/Gradient Boosting/MLP, and class-weighted variants of Logistic
Regression/Decision Tree/Random Forest (the three scikit-learn models
that accept `class_weight` directly — Gradient Boosting doesn't, and
wasn't force-fit with `sample_weight` to keep this simple, documented
as a deliberate scope limit).

## 10. Hyperparameter search

Small, fixed 4-candidate grid per model type (Random Forest, Gradient
Boosting, MLP), evaluated on a **chronological inner-validation split**
(last 20% of training data), never the real test set. Full search
space and results: `results/problem1/hyperparameter_search_results.json`.

**Honest finding: tuning barely moved the needle.** Every tuned-vs-
untuned delta was within ±0.008 balanced accuracy, and half were
slightly *negative* — e.g. sky-condition/Amherst MLP: untuned 0.816 →
tuned 0.809. The tested grids didn't find meaningful improvement over
sensible defaults; this is reported as-is rather than only showing the
tuned numbers.

## 11. Random seeds

42, 123, 2026 (`src/utils.py`'s `DEFAULT_SEEDS`) for every randomized
model; the hyperparameter search itself used a single seed (42) for
the search process, per the framework's design (search determinism
isn't the thing multi-seed reporting is protecting against — see
`course_context/PROJECT_STATUS.md`'s Phase 2 summary).

## 12. Evaluation metrics

Balanced accuracy (headline, per spec), accuracy, macro-precision,
macro-recall, macro-F1, confusion matrices, per-class precision/
recall/F1/support — all via `src/evaluation.py` (Phase 2 framework).

## 13. Results

**Mean balanced accuracy across 3 seeds, full feature set:**

| Task | City | Best model | Balanced acc. | Macro F1 |
|---|---|---|---|---|
| Sky-condition | Davis | logistic_regression_balanced_weight | 0.772 | 0.720 |
| Sky-condition | Amherst | mlp | 0.816 | 0.815 |
| Generation-regime | Davis | mlp_tuned | 0.946 | 0.946 |
| Generation-regime | Amherst | random_forest | 0.758 | 0.756 |

Full model-by-model table: `results/problem1/problem1_results.csv`.
**No single model type won everywhere** — a real, reportable finding
(Section 20).

**Class weighting effect** (mean, full feature set): mostly small,
mixed-sign (e.g. sky-condition/Davis logistic regression +0.003,
decision tree **-0.019**; generation-regime results were flat or
slightly negative everywhere). Worth testing, not worth assuming.

## 14. Best model

See Section 13's table — differs by (task, city); see Section 20 for
discussion of why.

## 15. Confusion matrices

`figures/problem1/problem1_sky_condition_confusion_matrix_{Davis,Amherst}.png`,
`figures/problem1/problem1_generation_regime_confusion_matrix_{Davis,Amherst}.png`.

## 16. Per-class metrics

**Sky-condition, Davis** (logistic_regression_balanced_weight):
Overcast P=0.53/R=0.85/F1=0.65, Partly Cloudy P=0.56/R=0.57/F1=0.57,
Clear P=1.00/R=0.90/F1=0.95.

**Sky-condition, Amherst** (mlp): Overcast P=0.84/R=0.59/F1=0.70,
Partly Cloudy P=0.67/R=0.89/F1=0.76, Clear P=1.00/R=0.97/F1=0.98.

**Generation-regime, Davis** (mlp_tuned): Low P=0.97/R=0.96/F1=0.97,
Medium P=0.92/R=0.91/F1=0.91, High P=0.96/R=0.97/F1=0.96.

**Generation-regime, Amherst** (random_forest): Low P=0.88/R=0.77/
F1=0.82, Medium P=0.72/R=0.60/F1=0.66, High P=0.70/R=0.90/F1=0.78.

**Hardest class:** Partly Cloudy (Davis sky-condition) and Overcast
(Amherst sky-condition) for Task A; Medium for Task B in both cities —
the *middle* class is consistently hardest, which makes sense: it's
defined by two boundaries instead of one, so it accumulates confusion
from both neighboring classes.

## 17. Error analysis

Not just "the model made mistakes" — investigated why:

- **Sky-condition, Davis:** 813/4823 test rows misclassified (16.9%).
  Mean Clear-Sky Index on errors (0.707) sits well below the overall
  mean (0.845) — errors skew toward cloudier conditions, where the
  weather-proxy features are inherently less distinctive. **35.2% of
  errors fall within 0.05 of a decision threshold** (0.4 or 0.85) —
  over a third of mistakes are genuinely borderline cases, not wild
  misses.
- **Sky-condition, Amherst:** 356/2412 (14.8%) misclassified; 25.3%
  near a threshold boundary — same pattern, smaller magnitude.
- **Generation-regime, Davis:** only 244/4823 (5.1%) misclassified —
  much lower error rate than Task A, consistent with Task B's much
  higher accuracy overall (irradiance features are directly, strongly
  predictive of Output Power).
- **Generation-regime, Amherst:** 607/2411 (25.2%) misclassified —
  notably higher than Davis, consistent with Amherst's skewed test-set
  class distribution (Section 6) making the "Medium"/"High" boundary
  harder to hit consistently on unseen, later data.

## 18. Feature ablation

Random Forest, single consistent model across all groups (see Section
20 for why one model was used):

| Task | City | time_only | weather_only | irradiance_weather | full | full+leakage_risk |
|---|---|---|---|---|---|---|
| Sky-condition | Davis | 0.356 | 0.613 | 0.742 | 0.743 | **0.985** |
| Sky-condition | Amherst | 0.360 | 0.561 | 0.803 | 0.812 | **0.982** |
| Generation-regime | Davis | 0.826 | 0.881 | 0.939 | 0.945 | — |
| Generation-regime | Amherst | 0.494 | 0.680 | 0.755 | 0.758 | — |

Two clear findings: (1) **time alone is nearly useless for
sky-condition** (0.36, barely above the 0.33 majority baseline) but
**meaningfully useful for generation-regime** (0.49–0.83, since
time-of-day/season correlates with typical output levels); (2) the
leakage-risk ablation's jump to ~0.98 (Section 6) is by far the
largest effect in this entire study.

## 19. Feature importance

Permutation importance (implemented directly — `compute_
permutation_importance()` — rather than via scikit-learn's estimator-
wrapper machinery, so the identical method works for both scikit-learn
models and the PyTorch MLP; see the function's docstring).

**Sky-condition:** `Cloud Type` (specifically the "Clear" category)
dominates for both cities (Davis 0.248, Amherst 0.266) — makes sense,
it's the most direct available cloud observation once irradiance is
excluded. `Temperature`/`Dew Point` are secondary.

**Generation-regime:** `GHI` dominates (Davis 0.277, Amherst 0.063),
with `Solar Zenith Angle` and `Clearsky GHI` also prominent for Davis —
consistent with the strong GHI-Output Power physical relationship
already confirmed in `EDA_REPORT.md`.

**Feature importance is not causal importance** — it measures how much
a *specific fitted model* relies on a feature, not what physically
drives PV output.

## 20. Discussion

**No single model type wins across all four (task, city) combos** —
logistic regression won Davis sky-condition, MLP won Amherst
sky-condition and Davis generation-regime, Random Forest won Amherst
generation-regime. This is itself informative: for the harder,
information-poor Task A, a well-regularized linear model or a small
MLP is competitive with (sometimes better than) tree ensembles; for
Task B, where the strongest features (irradiance) are directly
available, ensemble methods and the MLP both do very well, with
smaller gaps between model types.

**Davis notably outperforms Amherst on generation-regime** (0.946 vs.
0.758). Contributing factors: Davis has ~2x Amherst's training data
(19,289 vs. 9,641 rows), and Amherst's chronologically-later test
period has a skewed regime-class distribution (Section 6) that the
train-fit tercile boundaries don't perfectly anticipate — both
plausible explanations, not disentangled further in this phase.

**Ablation used one model (Random Forest) throughout**, not the
per-combo "best" model, so ablation groups are directly comparable to
each other — using a different model per combo would confound "does
this feature group help" with "which model handles this feature group
better."

## 21. Limitations

- Hyperparameter grids were small and fixed (4 candidates); a larger
  or randomized search might do better, though Section 10's finding
  suggests there isn't much room within this search space.
- Class weighting was only tested for 3 of 6 model types (documented
  scope decision, Section 9).
- Error analysis and feature importance used a single seed (42) for
  reproducible, inspectable figures — not averaged across seeds like
  the headline metrics.
- The Davis/Amherst performance gap (Task B) has plausible but
  untested explanations (data volume vs. test-period skew) — not
  isolated experimentally this phase.

## 22. Reproducibility information

Seeds: 42, 123, 2026. Python 3.12.3; pandas 3.0.2; numpy 2.4.4;
scikit-learn 1.8.0; torch 2.13.0 (CPU used in this sandbox — no GPU
present here; will use CUDA automatically via `src/utils.py`'s
`get_device()` on the project owner's RTX 2070). Dataset: `course/
Further Consolidated Data, HnL.xlsx`. Full run history: `results/
problem1/problem1_results.csv` (198 rows), `results/experiment_
history.csv`. Hyperparameter search results: `results/problem1/
hyperparameter_search_results.json`. Best models/preprocessors:
`results/problem1/models/`. Note on execution: this phase's
experiments were run as several separate script invocations (per
task/city combo) rather than one continuous run, purely to stay within
this sandbox's per-command execution time limit — every stage is
still a real, deterministic, independently-reproducible run; nothing
about the results depends on this batching.
