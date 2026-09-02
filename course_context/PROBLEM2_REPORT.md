# Problem 2 Report — Supervised Regression

Internal technical record for Problem 2, the project's core task.
Every number below comes from an actual executed run of `problems/
problem2_regression/run_experiments.py` — nothing is estimated. Full
results: `results/problem2/problem2_results.csv` (108 rows).
Reproducibility: seeds 42/123/2026, Python 3.12.3, pandas 3.0.2,
scikit-learn 1.8.0, torch 2.13.0, run 2026-09-01.

---

## 1. Objective

Predict continuous Output Power (kW) from meteorological, irradiance,
and temporal features, across three settings: same-city, cross-city
zero-shot, and K=12 sequence forecasting.

## 2. Data

Davis and Amherst (same-city); Davis→Huron/Santa Barbara/La Jolla
(cross-city); Davis 3-year vs. 6-year sheets (ablation). All loaded via
`src/data_loader.py`. Missing `Output Power` rows dropped (not
interpolated) — same reasoning as `PROBLEM1_REPORT.md`, Section 8: a
regression target should never be a fabricated/interpolated value
treated as ground truth.

## 3. Same-city setup

Chronological 80/20 split per city (`src/splitting.py`). Davis:
19,289 train / 4,823 test. Amherst: 9,641 train / 2,411 test.

## 4. Cross-city setup

Source: Davis (6-year). Targets: Huron, Santa Barbara, La Jolla.
**Genuinely zero-shot**: the target cities' data is loaded, cleaned,
and feature-engineered identically, but the SOURCE's already-fitted
preprocessor is applied to it (never a newly-fit one) — no target-city
statistic is used to fit anything. Verified: target feature columns
match the source's column order exactly before prediction.

## 5. 3-year vs 6-year setup

Davis only (the only city with both sheet lengths).
`gradient_boosting`, same protocol, each sheet's own chronological
80/20 split (see Section 16 for why this isn't a perfectly clean
comparison).

## 6. Feature engineering

`Clear_Sky_Index` (safe here — unlike Problem 1, Output Power isn't
derived from a fixed rule applied to GHI, so no leakage risk),
`Hour_sin/cos`, `Month_sin/cos`, `DayOfYear_sin/cos` — reused directly
from `src/feature_engineering.py` (Phase 2). Primary feature set (30
columns after encoding, `problems/problem2_regression/features.py`):
all weather + irradiance + Cloud Type (encoded) + time features.

## 7. Target normalization

**The most important design decision in this phase — read carefully.**
Three roles, kept deliberately separate:

- **Same-city / cross-city (non-sequence) models:** raw kW target, no
  scaling at all. Simplest, and there's no cross-city leakage risk
  here since these are single-city experiments.
- **Cross-city zero-shot (Section 22):** also raw kW — **this is the
  only genuinely zero-shot-honest choice.** Any scheme that rescales
  predictions using a target city's own mean/max would need
  information a real zero-shot deployment wouldn't have at prediction
  time. nRMSE is still computed per target city (using that city's own
  test-period range) because nRMSE here is a **post-hoc evaluation
  metric** — interpreting how bad a raw-kW error is relative to that
  city's own scale — never used to generate the prediction or fit
  anything. A separate, explicitly-labeled **diagnostic** (not a real
  zero-shot method) rescales predictions by the target city's own mean
  ÷ Davis's training mean, purely to check whether the failure is
  *scale* mismatch vs. genuinely poor pattern-matching (Section 15).
- **Sequence model (GRU):** target IS scaled during training — but for
  a completely different, non-leakage reason: raw kW targets (up to
  ~260) made the GRU converge far too slowly (RMSE still ~101 after 10
  epochs; see Section 25). The scaler is fit on Davis's own training
  data only (no cross-city concern — single city), and every reported
  metric is computed after inverse-transforming predictions back to
  kW. This is standard NN training practice, not a workaround for
  Section 7's cross-city concern.

**Rejected approach:** predicting a physically-normalized target (e.g.
power ÷ city capacity) for zero-shot. Not implemented as the *reported*
method because "city capacity" isn't independently known in this
dataset — the only available proxy (historical mean/max) is target-city
information a true zero-shot system wouldn't have before deployment.
Used only in the labeled diagnostic above, never the headline result.

## 8. Leakage prevention

Current-timestep `Output Power` excluded from every non-sequence
experiment. Sequence model: `Output Power` used as a **lag** feature
inside each window, legitimate because every value in a window is
strictly earlier than the target (the step right after the window) —
verified via `sequence.build_sequences()`'s window construction
(target = row `i+1`, window = rows `[i-11, i]`). Preprocessing fit on
training data only throughout (`src/preprocessing.py`, Phase 2,
unchanged).

## 9. Classical models

Mean baseline, Linear Regression, Ridge, Decision Tree, Random Forest,
Gradient Boosting (`problems/problem2_regression/models.py`). Random
Forest's default `n_estimators` reduced from 200→100 partway through
development after timing a single fit at 41s vs. 20s — a deliberate,
documented efficiency change with negligible quality impact (RMSE
15.56→15.59 on the same fit), given how many times this model runs
across the full experiment suite.

## 10. Neural-network model

`SimpleMLPRegressor`: Dense→ReLU→Dense→ReLU→Output, MSE loss, Adam,
early stopping on a chronological inner-validation split (last 20% of
training data).

## 11. Sequence model

`SimpleGRURegressor`: single GRU layer → final hidden state → Dense →
1 output. **Why GRU, not LSTM:** both taught (`COURSE_CONTEXT.md`,
Week15); GRU has fewer parameters and trains faster, and K=12 windows
don't obviously need LSTM's extra gating — a reasonable, explainable
choice, not an exhaustive architecture search.

## 12. Train/validation/test strategy

Chronological 80/20 (train/test) throughout; hyperparameter search and
early stopping use a further chronological inner-validation split of
the training portion only (last 20%) — the real test set is **never**
touched until final evaluation, for every model type including the
GRU search (initially implemented incorrectly — scoring candidates
against the real test set — caught and fixed before any tuned result
was recorded; see Section 21).

## 13. Metrics

RMSE, MAE, nRMSE (range-normalized — `RMSE / (max(y) - min(y))`,
**the same convention established in Phase 2's `src/evaluation.py`**,
kept consistent throughout this project rather than switching
definitions), R² (added to `src/evaluation.py` this phase — additive,
non-breaking, existing tests unaffected).

## 14. Same-city results

| City | Best model | RMSE (kW) | MAE (kW) | nRMSE | R² |
|---|---|---|---|---|---|
| Davis | gradient_boosting | 15.17 | 8.31 | 0.0600 | 0.953 |
| Amherst | gradient_boosting_tuned | 20.29 | 13.61 | 0.1595 | 0.723 |

Full model-by-model table (mean ± std across 3 seeds):
`results/problem2/problem2_results.csv`. **Notable: untuned
`gradient_boosting` beat every tuned variant for Davis** (15.17 vs.
15.39 tuned) — the same "small search doesn't always help" pattern
already documented in `PROBLEM1_REPORT.md`. `mean_baseline` (71.84
Davis, 38.71 Amherst) confirms every real model is doing genuine work.
Figure: `figures/problem2/problem2_model_comparison.png`.

## 15. Cross-city results

| Target | Zero-shot RMSE | Zero-shot R² | Oracle-mean RMSE | Scale-corrected (diagnostic) RMSE | Scale-corrected R² |
|---|---|---|---|---|---|
| Huron | 125.81 | **-58.07** | 16.60 | 6.65 | 0.835 |
| Santa Barbara | 125.48 | **-48.95** | 17.78 | 11.90 | 0.551 |
| La Jolla | 124.39 | **-71.97** | 14.63 | 7.73 | 0.718 |

**Not hidden, as instructed:** raw zero-shot transfer is a severe
failure — R² far below zero means the Davis model is drastically worse
than simply guessing the target city's own mean. But the diagnostic
scale-correction (Section 7) proves this is **almost entirely a scale
problem, not a pattern-recognition failure**: once rescaled, R² jumps
to 0.55–0.84, and RMSE (6.65–11.90) actually **beats** the oracle mean
baseline. Figure: `figures/problem2/problem2_cross_city_comparison.png`.

## 16. Sequence results

| Model | RMSE (kW) | MAE (kW) | nRMSE | R² |
|---|---|---|---|---|
| persistence_baseline | 21.97 | 15.04 | 0.087 | 0.901 |
| gru | 17.58 | 10.04 | 0.070 | 0.937 |
| gru_tuned | 17.59 | 10.20 | 0.070 | 0.937 |

**GRU clearly beats the persistence baseline** — it's learning real
structure, not just repeating the last observation. **GRU does not
beat Davis's best non-sequence model** (15.17) — see Section 20 for
why this isn't an apples-to-apples comparison. Tuning made no
measurable difference here either (17.58 → 17.59). Figure:
`figures/problem2/problem2_sequence_comparison.png`.

## 17. Learning curve

| Train fraction | n rows | RMSE | R² |
|---|---|---|---|
| 0.2 | 3,857 | 16.21 | 0.946 |
| 0.4 | 7,715 | 15.69 | 0.950 |
| 0.6 | 11,573 | 15.51 | 0.951 |
| 0.8 | 15,431 | 15.27 | 0.952 |
| 1.0 | 19,289 | 15.17 | 0.953 |

Clear diminishing returns — most of the gain happens by 40% of
current data; doubling from 60%→100% only improves RMSE by 0.34.
Suggests Davis's 6 years of history is already past the point of
strong marginal value from more of the same. Figure: `figures/
problem2/problem2_learning_curve.png`.

## 18. Prediction-vs-truth analysis

`figures/problem2/problem2_predicted_vs_actual_davis.png`: tight
clustering around the perfect-prediction line, with the largest
scatter at low-to-mid Output Power values (matches Section 19's
finding). `figures/problem2/problem2_prediction_timeseries_davis.png`:
a representative 5-day window shows the model tracking the daily
rise/fall generation cycle closely, including a low-output cloudy day
(2023-10-24 in the plotted window) where prediction and actual both
dip together.

## 19. Error analysis

Mean absolute error by condition (Davis, `gradient_boosting`):

| Condition | Mean abs. error (kW) |
|---|---|
| Overall | 8.31 |
| Low GHI (bottom tercile) | 12.40 |
| High GHI (top tercile) | 3.99 |
| Rapidly changing GHI (top decile of \|ΔGHI\|) | **17.42** |
| Cloudy (Clear-Sky Index < 0.4) | 13.02 |
| Clear (Clear-Sky Index ≥ 0.85) | 5.36 |

Consistent, physically sensible pattern: **errors concentrate in
low-irradiance, cloudy, and especially rapidly-changing conditions** —
more than 2x the overall average when irradiance is swinging quickly.
Stable, high-irradiance (clear-sky) periods are the easiest to predict
by a wide margin. This directly explains why the sequence model
(Section 16), which only sees *past* weather, struggles relatively more
during volatile periods than the non-sequence model, which gets the
*concurrent* GHI reading for free.

## 20. Best model

**Same-city:** `gradient_boosting` (Davis), `gradient_boosting_tuned`
(Amherst) — both untuned/lightly-tuned gradient boosting, consistent
with Problem 1's finding that ensemble methods handle this dataset's
structure well. **Not the GRU** — see the important caveat below.

**Why the sequence model "losing" to the non-sequence model doesn't
mean sequence modeling failed:** these are different tasks. The
non-sequence same-city model gets the **concurrent** timestep's GHI
and other irradiance readings — i.e., it "cheats" relative to a true
forecast by seeing the exact weather at the moment being predicted.
The GRU only sees the **previous 12 steps** and must forecast the
*next* one using no information about that future moment's actual
weather. A fairer comparison is GRU vs. persistence (Section 16),
where the GRU's advantage (17.58 vs. 21.97) is clear and meaningful.

## 21. Limitations

- Hyperparameter grids were small and fixed (3-4 candidates per model
  type); tuning improved almost nothing across this whole phase,
  matching Problem 1's finding — worth a larger/different search space
  in future work, though the pattern suggests defaults are already
  reasonably good here.
- **A real bug was caught and fixed during development:** the first
  GRU hyperparameter search implementation scored candidates directly
  against the real test set — a violation of the project's own "never
  tune against the test set" rule. Caught before any result was
  recorded and rewritten to use a proper chronological inner-validation
  split. Documented here per the instruction to be honest about
  mistakes, not just successes.
- The 3-year vs. 6-year comparison (Section 5) has a confound: the two
  Davis sheets don't share a test period (3yr covers roughly mid-2016
  onward, 6yr's test period starts earlier and spans a longer, more
  varied stretch) — so "3yr beat 6yr" (RMSE 12.04 vs. 15.17) may
  reflect an easier test window as much as (or instead of) less data
  being better. Not disentangled this phase.
- Error analysis and prediction plots use a single seed/model instance,
  not averaged across seeds.
- The sequence model was only tested for Davis, not Amherst or the
  cross-city targets — time/scope tradeoff given the phase's size.

## 22. Reproducibility information

Seeds: 42, 123, 2026. Python 3.12.3; pandas 3.0.2; numpy 2.4.4;
scikit-learn 1.8.0; torch 2.13.0 (CPU in this sandbox — no GPU present
here; will use CUDA automatically via `src/utils.py`'s `get_device()`
on the project owner's RTX 2070). Dataset: `course/Further Consolidated
Data, HnL.xlsx`. Full run history: `results/problem2/
problem2_results.csv` (108 rows). Hyperparameter search results:
`results/problem2/hyperparameter_search_results.json`,
`gru_hyperparameter_search.json`. Best models: `results/problem2/
models/` (`davis_best_model.joblib`, `amherst_best_model.joblib`,
`cross_city_model.joblib`, `sequence_model.pt`, all preprocessors, and
`model_config.json` documenting exact configurations) — all verified
loadable. Note on execution: like Problem 1, this phase's stages were
run as several separate script invocations to stay within this
sandbox's per-command execution-time limit; every stage remains a
real, deterministic, independently-reproducible run.
