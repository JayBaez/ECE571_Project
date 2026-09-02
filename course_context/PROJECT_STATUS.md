# Project Status

**Last updated:** Problem 2 completion (supervised regression).
**Future AI agents: update this file as the project progresses. Keep
entries short — status + one-line note, not a log of everything done.**

```
Phase 0 — Course material & spec analysis: COMPLETE
Phase 1 — Repository construction:          COMPLETE
Phase 2 — ML framework:                     COMPLETE
Phase 3 — Dataset validation:                COMPLETE
Problem 1 (Classification):                 COMPLETE
Problem 2 (Regression):                     COMPLETE
Problem 3 (Dimension Reduction):            NOT STARTED
Problem 4 (Semi-Supervised Learning):       NOT STARTED
Problem 5 (Transfer Learning):              NOT STARTED
Final optimization:                         NOT STARTED
Report:                                     NOT STARTED
Video presentation:                         NOT STARTED
```

## Problem 2 summary

- Built `problems/problem2_regression/` (features.py, models.py,
  sequence.py, run_experiments.py) reusing the Phase 2 framework
  throughout, including target scaling functions built in Phase 2
  specifically for this kind of need (`fit_target_scaler()` etc.).
  Added R² to `src/evaluation.py`'s `regression_metrics()` (additive,
  non-breaking - existing tests unaffected, one new test added).
- Ran same-city (Davis, Amherst), cross-city zero-shot (Davis→Huron/
  Santa Barbara/La Jolla), 3yr-vs-6yr (Davis), and K=12 sequence
  (GRU) experiments — 108 real experiment rows in `results/problem2/
  problem2_results.csv`.
- **Cross-city zero-shot showed severe raw-scale failure (R² as low as
  -72), reported honestly rather than hidden** — then a labeled
  diagnostic proved the failure was almost entirely a scale mismatch,
  not a pattern-recognition failure (R² 0.55-0.84 once rescaled).
- **Caught and fixed a real methodology bug during development:** the
  first GRU hyperparameter search implementation scored candidates
  against the real test set, violating the project's own rule. Fixed
  before any tuned result was recorded.
- Saved best models for Davis, Amherst, cross-city, and the sequence
  model (`results/problem2/models/`), all verified loadable.
- **No fabricated numbers** — every result in
  `course_context/PROBLEM2_REPORT.md` traces to an actual run.

## Problem 1 summary

- Built `problems/problem1_classification/` (targets.py, features.py,
  models.py, run_experiments.py) reusing the Phase 2 framework
  throughout — no framework code needed rebuilding, only one small,
  well-justified extension (`torch_utils.make_dataloader()` gained a
  `y_dtype` parameter to support classification's Long-typed labels,
  alongside regression's existing Float default).
- Ran the full model progression (majority baseline → logistic
  regression → decision tree → random forest → gradient boosting →
  MLP) for both tasks (sky-condition, generation-regime), both cities
  (Davis, Amherst), at 3 seeds each — 198 real experiment rows in
  `results/problem1/problem1_results.csv`.
- Ran a small hyperparameter search (chronological inner validation),
  a class-weighting comparison, and a feature ablation study —
  **found and confirmed empirically** that the Solar Zenith Angle/
  DHI/DNI leakage risk flagged in Phase 3 is real: including those
  features pushed sky-condition accuracy from ~0.74–0.81 to ~0.98.
- Saved confusion matrices, per-class metrics, error analysis, and
  permutation-importance figures for the actual best model of each
  (task, city) combo — plus the models/preprocessors themselves
  (`results/problem1/models/`), all verified loadable.
- **No fabricated numbers** — every result in
  `course_context/PROBLEM1_REPORT.md` traces to an actual run.

## Phase 3 summary

- Built `scripts/run_eda.py`: a reusable, re-runnable EDA script
  covering all 34 requested analysis areas — sheet/column/dtype
  verification, missing values, duplicates, timestamp validation,
  Output Power distributions, irradiance relationships, Clear-Sky
  Index, Cloud Type, weather features, correlations, target-scale
  analysis, class-imbalance preview, and autocorrelation. Never
  modifies the raw Excel file (verified: file checksum unchanged
  after every run).
- Saved 19 tables to `results/eda/*.csv` and 11 figures to
  `figures/eda/*.png`.
- **Found one previously-undocumented, genuine data-quality issue:**
  `Relative Humidity`/`Wind Direction` appear swapped and/or
  mis-unitted for all of Davis 2013 and all of Huron 2012 (~14% of
  each city's data) — see `course_context/EDA_REPORT.md` and
  `DATASET_PROFILE.md`.
- Found and fixed a real internal contradiction in
  `DATASET_PROFILE.md` left over from Phase 0/1: its "Recommended
  defaults" section still said to skip the redundant `'14-'16` sheets,
  directly contradicting the correction earlier in the same file
  (which says both sheets are needed for the spec's 3yr-vs-6yr
  ablation). Fixed and now internally consistent.
- Created `course_context/LEAKAGE_MAP.md` (per-problem feature safety
  rules) and `course_context/EDA_REPORT.md` (full technical findings).
- Updated `DATASET_PROFILE.md` throughout with explicit
  [SPECIFIED]/[VERIFIED] labels per section, per this phase's
  instruction not to let an assumption pass as a fact.
- **No ML models were trained. No problem-specific code was written.
  No modeling decisions (final classifier, regressor, SSL algorithm,
  transfer strategy) were locked in.**

## Phase 2 summary

- Extended the Phase 1 framework into a full reusable ML
  experimentation system: 2 new modules (`cleaning.py`, `torch_utils.py`),
  substantial additions to `preprocessing.py` (target/feature
  separation, bundled preprocessor, city-specific target scaling),
  `splitting.py` (reproducible split metadata, overlap verification),
  `evaluation.py` (multi-seed aggregation), `experiment_runner.py`
  (full artifact system: metrics/config/predictions/training-log
  saving, plus a leaderboard query function).
- Built a 79-test suite (`tests/`) covering every module, using
  synthetic data for speed except where testing against a real,
  known Phase 0 finding was more meaningful (e.g. the Amherst
  4-missing-row regression test). **All 79 tests pass.**
- Built and ran a framework demonstration
  (`scripts/framework_demo.py`) proving the full pipeline works
  end-to-end on synthetic data with a trivial Linear Regression model.
  **This demo is clearly not a project result** - its outputs live
  under `results/framework_demo/` and `figures/framework_demo/`,
  never under `problem1`-`problem5`.
- Found and fixed one real bug during this phase: adding a new column
  (`parameters`) to the results-history schema without a safety check
  would have silently misaligned every future row against the old
  header. Added a schema-mismatch guard to `save_result()` (and a
  test for it) so this can't happen silently again.
- **No ML models were trained on real project data. No problem-specific
  code was written. `results/experiment_history.csv` contains exactly
  one row: the labeled framework demo (not a real result).**

## Phase 1 summary

- Built the repository foundation: `src/` (8 reusable modules),
  `configs/`, `problems/` (5 placeholder folders), `results/` (schema
  only, no data), `models/`, `figures/`, `logs/`, `data/`,
  `requirements.txt`, `.gitignore`, `README.md`, `scripts/check_setup.py`.
- All 8 `src/` modules were smoke-tested against the real dataset
  (loading, splitting, scaling, encoding, feature engineering,
  metrics, plotting, config loading, result logging).
- **No ML models were trained. No problem-specific code was written.
  `results/experiment_history.csv` and `results/leaderboard.csv`
  contained headers only — zero real rows.**

## Phase 0 summary

- Read all 19 course files (18 PDF + 1 pptx) and the real project spec
  document → `COURSE_CONTEXT.md`, `TEACHER_EXPECTATIONS.md`.
- Inspected the actual Excel dataset (9 sheets, 5 cities) → 
  `DATASET_PROFILE.md`. Found one genuine anomaly (2012-03-22 zero-output
  across 4 cities) and one spec-vs-data discrepancy (stated sampling
  window vs. actual).
- Mapped course concepts to project needs, separating taught / useful-
  but-uncovered / unnecessary → `ML_METHOD_MAP.md`.
- Built a non-executed, per-problem experiment plan → `EXPERIMENT_PLAN.md`.
- **No code had been written. No models had been trained. No metrics
  existed yet anywhere in the project.**

## Known open items

None currently — the three items previously listed here (Problem 4's
SSL algorithm, the sky-condition classifier's Solar Zenith Angle/DHI/
DNI feature question, and the Davis-2013/Huron-2012 Relative
Humidity/Wind Direction finding) were all resolved based on the
project owner's direct guidance. See `TEACHER_EXPECTATIONS.md`,
`EXPERIMENT_PLAN.md`, `LEAKAGE_MAP.md`, and `YOUR_PROJECT_NOTES.md`
for the resolutions.

## Context on repository history

Before this Phase 0 work began, git history showed an earlier,
now-deleted implementation attempt (a `README.md`, an `ece571/` package
with `problem1.py`–`problem5.py`, and `results/` with computed metrics
and figures), removed by a commit titled "wipe." Per instruction, this
Phase 0 work treated the project as a fresh start and did not reuse or
verify those old results. If any of that old code/results should be
recovered from git history for reference in a later phase, that's a
decision for the project owner, not something a future agent should do
unprompted.
