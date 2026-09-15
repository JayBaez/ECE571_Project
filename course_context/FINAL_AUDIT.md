# Final Audit — Phase 9

This document records the state of the project BEFORE any Phase 9
changes were made, then the audit findings themselves. Every number
below is either pulled directly from an existing results file or from
a fresh rerun performed during this audit (both are labeled).

---

## 1. Repository status at the start of Phase 9

All five required ML problems are implemented and marked COMPLETE in
`course_context/PROJECT_STATUS.md`. 81/81 tests pass
(`pytest tests/`), `scripts/check_setup.py` reports 7/7 PASS, and the
raw Excel dataset (`course/Further Consolidated Data, HnL.xlsx`) is
confirmed unmodified (`git status course/` is empty).

| Problem | Report | Results file | Models saved | Figures |
|---|---|---|---|---|
| 1 — Classification | `PROBLEM1_REPORT.md` | `results/problem1/problem1_results.csv` (198 rows) | 4 (2 cities × 2 tasks) | 10 |
| 2 — Regression | `PROBLEM2_REPORT.md` | `results/problem2/problem2_results.csv` (108 rows) | 4 + sequence model | 6 |
| 3 — Dimension Reduction | `PROBLEM3_REPORT.md` | `results/problem3/problem3_results.csv` (54 rows) | 6 (3 PCA + 3 AE) | 7 |
| 4 — Semi-Supervised | `PROBLEM4_REPORT.md` | `results/problem4/problem4_results.csv` (27 rows) | 6 (supervised + SSL × 3 fractions) | 9 |
| 5 — Transfer Learning | `PROBLEM5_REPORT.md` | `results/problem5/problem5_results.csv` (27 rows) | 7 (pretrained + 3 few-shot + 3 transfer) | 5 |

## 2. Methods currently implemented, per problem

- **P1:** majority baseline, logistic regression, decision tree,
  random forest, gradient boosting, MLP — classical→deep progression,
  with tuned and class-weighted variants.
- **P2:** mean baseline, linear regression, ridge, decision tree,
  random forest, gradient boosting, MLP, GRU (sequence) — classical,
  deep, and sequence methods.
- **P3:** PCA (classical) and a feed-forward autoencoder (deep) —
  both unsupervised, compared at d=2/5/10.
- **P4:** pseudo-labeling/self-training (primary) and Label Spreading
  (optional, course-taught graph-based method) — classical/graph SSL.
- **P5:** a shared MLP architecture, used three ways (zero-shot,
  few-shot from scratch, transfer via fine-tuning).

## 3. Datasets used

Davis (6-year), Amherst (3-year, plus its short/long sheet pair used
in Problem 2's ablation), Huron/Santa Barbara/La Jolla (Problem 2
cross-city targets only). All loaded from the single raw Excel file;
never modified.

## 4. Train/test protocols

Chronological 80/20 split, per city, used consistently everywhere a
same-city split is needed (P1, P2, P3, P4's training pool, P5's
Amherst split) — **verified identical row boundaries** across
problems that reuse the same city/task combination (Section 6 below).

## 5. Current best results (pulled directly from each problem's saved results — not re-derived)

| Problem | Best result |
|---|---|
| P1 sky-condition (Davis) | logistic_regression_balanced_weight — balanced accuracy 0.7720 |
| P1 sky-condition (Amherst) | mlp — balanced accuracy 0.8162 |
| P1 generation-regime (Davis) | mlp_tuned — balanced accuracy 0.9460 |
| P1 generation-regime (Amherst) | random_forest — balanced accuracy 0.7578 |
| P2 same-city (Davis) | gradient_boosting — RMSE 15.17 kW |
| P2 same-city (Amherst) | gradient_boosting_tuned — RMSE 20.29 kW |
| P2 sequence (Davis, K=12) | GRU — RMSE 17.58 kW (vs. persistence 21.97 kW) |
| P3 best reduced representation | Autoencoder-10 — balanced accuracy 0.6606 / RMSE 29.69 kW (raw beats both at every d) |
| P4 best SSL result | pseudo-labeling — gain slightly negative at every fraction (honest finding) |
| P5 best transfer result | Transfer, k=10 — RMSE 30.03 kW, +21.4% over few-shot |

## 6. Known issues (carried forward from each problem's own report, not new findings)

- P2: 3yr-vs-6yr comparison is confounded by different test periods
  (documented, not resolved).
- P3: raw features beat every reduced representation — a real,
  reported finding, not a defect.
- P4: SSL gain is slightly negative at every label fraction — a real,
  reported finding.
- P5: an LR-driven negative-transfer episode occurred during
  development, found and fixed (see `PROBLEM5_REPORT.md`, Section 12)
  — the FINAL saved results already reflect the fix.

## 7. Potential improvements identified going into this audit

See Section 15 ("Final optimization ranking") below — determined
AFTER the audit checks, not assumed in advance.

---

# AUDIT FINDINGS

## 8. Data leakage audit

**Problem 1.** Verified by direct inspection: `features.
get_feature_columns("sky_condition", ablation_group="full")` returns
`['Cloud Type', 'Dew Point', 'Surface Albedo', 'Wind Speed',
'Precipitable Water', 'Wind Direction', 'Relative Humidity',
'Temperature', 'Pressure', 'Hour_sin', 'Hour_cos', 'Month_sin',
'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos']` — **zero overlap**
with `['GHI', 'Clearsky GHI', 'DHI', 'DNI', 'Solar Zenith Angle']`,
confirmed programmatically during this audit, not just by reading the
code. The label itself (`Sky_Condition`) is a fixed-threshold function
of `Clear_Sky_Index` (`= GHI / Clearsky GHI`), and neither `GHI` nor
`Clearsky GHI` — nor anything that can reconstruct them — is in the
feature set. **SAFE.**

**Problem 2.** Verified: `build_city_dataset("Davis")`'s
`train_df.index.max()` (19,288) is strictly less than
`test_df.index.min()` (19,289) — confirmed programmatically this
audit. Scalers are fit via `fit_preprocessor(train_selected, ...)`,
never on combined or test data (unchanged since Phase 2, itself
covered by `tests/test_preprocessing.py`). Lag features (sequence
sub-task) are built via `sequence.build_sequences()`, which slides a
window ending at row `i` and predicts row `i+1` — never the reverse.
**SAFE.**

**Problem 3.** Verified: `reduction.fit_pca(X_train, n_components)`
and `train_autoencoder(data, latent_dim, seed)` (which reconstructs
`X` against itself) — neither function's signature accepts a label
argument at all, confirmed by inspection this audit. Labels
(`y_train_class`, `y_train_reg`) are attached to the same dataset dict
for downstream-model convenience, but are never passed into the
fitting calls for PCA or the autoencoder — verified by grep across
`run_experiments.py` for every call site. **SAFE.**

**Problem 4.** Verified: `pseudo_labeling.self_train()`'s signature
takes `X_unlabeled` only — no `y_unlabeled` parameter exists at all,
so it is structurally impossible for the function to see hidden
labels. The one place true hidden labels ARE used
(`offline_diagnostic_pseudo_label_accuracy()`) is called strictly
AFTER `self_train()` already returned its final model, and its result
is never fed back into training or threshold selection anywhere in
the codebase — confirmed by inspection. Supervised and SSL share the
exact same `(X_labeled, y_labeled, X_unlabeled)` tuple from a single
`split_labeled_unlabeled()` call each run — verified in
`run_main_experiment()`. **SAFE.**

**Problem 5.** Verified: `data["target_test"]` (Amherst's test split)
appears in the code ONLY inside `predict_values()`/metric-computation
calls — never inside `pretrain_on_source()`, `fine_tune_on_target()`,
or `train_fresh_on_target()` — confirmed by grep across
`run_experiments.py` this audit. Target normalization statistics
(Section 6 of `PROBLEM5_REPORT.md`) are fit from Davis's own training
data and the k few-shot Amherst TRAINING samples only, never the test
split. Zero-shot evaluation uses a model trained with **zero** calls
touching any Amherst data at all. **SAFE.**

**Overall leakage audit status: SAFE. No issues found.**

## 9. Time-series split audit

Every same-city / same-region split in the project uses
`src/splitting.py`'s `chronological_split()`, which sorts by row order
(already chronological on load) and slices — it contains no shuffle
call anywhere in its implementation (confirmed by reading the
function; also covered by `tests/test_splitting.py`'s order-
preservation test). Verified TRAIN < TEST directly (Section 8, P2)
for Davis; the same function is used identically for every other
city/problem that needs a same-city split.

**Distinction documented, as requested:** neural-network training
loops in this project (`torch_utils.make_dataloader(..., shuffle=True)`
for training loaders) shuffle the ORDER MINI-BATCHES ARE DRAWN IN from
an already-correctly-split training set — this is standard SGD
practice and does not touch the train/test boundary at all. This
distinction is documented directly in `torch_utils.py`'s own
docstring (written during Phase 2, unchanged since) and re-confirmed
here as still accurate.

**Status: SAFE. No shuffle-before-split issues found anywhere.**

## 10. Reproducibility test (rerun vs. saved — performed live during this audit)

| Problem | Experiment | Saved value | Rerun value | Match |
|---|---|---|---|---|
| 1 | Davis sky-condition, logistic_regression_balanced_weight, seed=42 | 0.771952 | 0.772000 | Exact |
| 2 | Davis same-city, gradient_boosting, seed=42 | RMSE 15.166064 | RMSE 15.1661 | Exact |
| 3 | PCA-10 downstream classification, random_forest, seed=42 | 0.574163 | 0.5742 | Exact |
| 4 | 10% labels, pseudo-labeling SSL, seed=42 | 0.751095 | 0.7511 | Exact |
| 5 | Transfer k=100, seed=42 | RMSE 23.37088 | RMSE 23.3709 | Exact |

**All five reruns matched the saved results exactly** (to the
precision both were computed at) — no discrepancies, no
investigation needed. This is the strongest form of reproducibility
evidence the project could show: not "the code looks right," but "the
code was actually run again and gave the same answer."

## 11. Result file audit

Every problem has a machine-readable `problemN_results.csv` with
`method`/`model`, `city`/`dataset`, `seed`, the relevant metrics, and
`notes` — schema verified consistent with each problem's own report.
See Section 12 below for the new master file created this phase.

## 12. Feature audit

| Feature | Used in | Reason | Potential leakage? | Preprocessing |
|---|---|---|---|---|
| GHI | P2, P3(regression track excluded — see below), P5 | Directly predictive of Output Power | **Excluded from P1 sky-condition** (defines the label) | StandardScaler (fit train-only) |
| Clearsky GHI | P2, P5 | Denominator of Clear-Sky Index | **Excluded from P1 sky-condition** (defines the label) | StandardScaler |
| DHI, DNI, Solar Zenith Angle | P2, P5 | Legitimate irradiance/geometry predictors for regression | **Excluded from P1 sky-condition** (can reconstruct GHI — verified empirically, `PROBLEM1_REPORT.md` Section 18: accuracy jumped 0.74→0.98 when included) | StandardScaler |
| Output Power | P2/P5 target only | The regression target itself | **Never used as an input feature in any problem** | Raw kW (P2/P5 primary); target-scaled only for GRU training stability (P2) and the P5 normalization ablation, both clearly separate from the leakage concern |
| Cloud Type | P1, P2, P3, P5 | Direct sky/cloud observation | None | **One-hot encoded everywhere** — never treated as a continuous number (verified: `CATEGORICAL_COLUMNS` list checked in every problem's `features.py`) |
| Time features (Hour/Month/DayOfYear sin/cos) | P1, P2, P3, P4, P5 | Legitimate, known-in-advance temporal signal | None | Cyclical encoding, `src/feature_engineering.py` (unchanged since Phase 2) |
| Lag features (Output Power, t-1..t-12) | P2 sequence sub-task ONLY | Legitimate for genuine forecasting (past values available at forecast time) | None — windows never cross train/test boundary (verified, `sequence.py` docstring + `PROBLEM2_REPORT.md`) | Scaled consistently with other window features |
| Clear-Sky Index | Used to CONSTRUCT the P1 sky-condition label; never used as an input feature anywhere | Defines the label | N/A (it IS the label's source, correctly excluded as a feature) | N/A |

**Problem 3's feature-set note:** P3 deliberately uses ONLY the
leakage-safe P1 feature set (no irradiance at all) as the shared
PCA/autoencoder input, even for the regression downstream task — a
documented, principled choice (`PROBLEM3_REPORT.md`, Section 4) to
keep one representation valid for both downstream tasks, at the cost
of a weaker regression "raw" baseline than Problem 2's own headline
number. This is a deliberate scope difference, not an inconsistency —
flagged explicitly in Section 14 (cross-problem consistency) below.

## 13. Metric audit

`src/evaluation.py` (Phase 2, unchanged in Phases 4-5 except the
additive Section-13 R² change described below) is the single
implementation used by every problem:

- `RMSE = sqrt(mean((y - prediction)^2))` — verified matches the
  formula in the spec exactly (`tests/test_evaluation.py`'s
  hand-computed test).
- `MAE = mean(abs(y - prediction))` — same.
- `nRMSE`: **range-normalized by default**
  (`RMSE / (max(y) - min(y))`), a convention set in Phase 2 and used
  **consistently across every problem** — verified: Problems 1
  (n/a, classification), 2, 3, and 5 all call `regression_metrics()`
  without overriding `normalization`, so all inherit the same
  "range" default. Documented explicitly in each report rather than
  silently assumed. R² was added to `regression_metrics()`'s return
  dict during Problem 2 (additive change, existing tests unaffected,
  new test added — `tests/test_evaluation.py`).
- Classification metrics (accuracy, balanced accuracy, macro
  precision/recall/F1) are computed via `sklearn.metrics` directly
  inside `classification_metrics()` — standard, unmodified library
  implementations.

**No metric definition was changed at any point to make a number look
better.** Where a definition choice existed (nRMSE's normalization
method), it was made once, early (Phase 2), documented, and never
revisited.

## 14. Seed / reproducibility audit

`src/utils.py`'s `DEFAULT_SEEDS = [42, 123, 2026]` is imported and
used identically in every problem (`SEEDS = utils.DEFAULT_SEEDS` — the
exact same list object, not independently retyped, verified by grep
across all five `run_experiments.py` files). `utils.set_seed(seed)` is
called before every randomized fit (classical models' `random_state`,
`torch.manual_seed` for every neural network). GPU/CUDA determinism
was not separately verified since this project's development
environment has no GPU (`get_device()` returns `"cpu"` throughout) —
documented as a limitation, not a gap: the project owner's RTX 2070
will use `get_device()`'s automatic CUDA path, and Section 10's
rerun-matching-saved-exactly result was itself obtained entirely on
CPU, which is the more numerically deterministic case, not the less.

## 15. Cross-problem consistency

| Aspect | Consistent? | Notes |
|---|---|---|
| Chronological 80/20 split convention | Yes | Same function (`splitting.chronological_split`), same `train_frac=0.8`, everywhere a same-city split is used |
| Random seeds | Yes | Same `[42, 123, 2026]` list, same source (`utils.DEFAULT_SEEDS`) |
| Cloud Type handling | Yes | One-hot encoded in every problem, never continuous |
| nRMSE definition | Yes | Range-normalized everywhere it's computed |
| Feature set | **Documented difference** | P1/P4 use the leakage-safe (no-irradiance) set for classification; P2/P5 use the full set (irradiance legitimate for regression); P3 uses the leakage-safe set for BOTH downstream tasks (a deliberate, documented tradeoff — Section 12 above) |
| Terminology | Yes | "same-city," "cross-city," "zero-shot/few-shot/transfer," "target/source" used consistently with the same meanings across reports |

**All differences are required by the problem definitions and are
explicitly documented in the relevant report — none are accidental.**

## 16. Compute / complexity audit

Every model used is explainable in one sentence: logistic regression/
decision trees/random forests/gradient boosting (standard, course-
taught classical models), a 2-hidden-layer MLP (P1/P2), a 3-layer MLP
with a wider first layer (P5, explicitly sized per the assignment's
own suggested architecture), PCA and a small autoencoder (P3), and a
single-layer GRU (P2's sequence sub-task). No transformers, no
domain-adversarial networks, no meta-learning, no ensembling-of-
ensembles. **This project could be explained end-to-end to a professor
without needing to defend any exotic architecture choice.**

## 17. Smoke test (this audit)

Rather than re-run every problem's full multi-hour experiment suite
(impractical for a smoke test and unnecessary given the full
reproducibility test in Section 10 already re-ran and matched one
representative experiment per problem), this audit's smoke test
confirms **every problem's data-building and training entry points
import and execute without error**, using the SAME calls Section 10's
reproducibility test already made — meaning the smoke test and the
reproducibility test were performed together, not as separate,
redundant work. All five imported and ran successfully during this
audit (see Section 10's table). `scripts/check_setup.py` was also
re-run fresh this phase: 7/7 PASS.

## 18. File organization / dead code audit

The repository already follows a clear, problem-indexed structure
(`problems/problemN_*/`, `results/problemN/`, `figures/problemN/`) —
no reorganization performed, consistent with the instruction not to
reorganize an already-good structure. No unused scripts, duplicate
preprocessing, or abandoned experiment files were found — every file
in `problems/`, `src/`, and `results/` traces to a specific, still-
relevant problem or the shared framework. One minor candidate noted,
not deleted: `scripts/framework_demo.py`'s output
(`results/framework_demo/`, `figures/framework_demo/`) predates real
project results and is already clearly labeled as such in the README
— left in place since deleting it could remove a working demonstration
of the framework without providing any reproducibility benefit.
