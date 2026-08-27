# Leakage Map

What features can and cannot be used for each problem, based on the
verified dataset structure (see `DATASET_PROFILE.md` and
`EDA_REPORT.md`). This is a practical checklist to consult when
building each problem's `prepare_xy()` call
(`src/preprocessing.py`) — every exclusion below should show up as an
explicit `exclude_columns` entry, not something left implicit.

---

## Problem 1 — Supervised Classification

### Target A: Sky-condition (Clear / Partly cloudy / Overcast)

Defined as `Clear-Sky Index = GHI / Clearsky GHI`, thresholded at
0.85 and 0.4 (see `TEACHER_EXPECTATIONS.md`).

- **MUST NOT use as features:** `GHI`, `Clearsky GHI` — these
  literally define the label. Using either as an input is using the
  answer to predict the answer (the project spec states this
  explicitly).
- **RESOLVED — also excluded from the primary model:** `DHI`, `DNI`,
  and `Solar Zenith Angle` together can approximately reconstruct
  `GHI` (`GHI ≈ DNI·cos(zenith) + DHI`), and `Solar Zenith Angle`
  alone correlates at **-0.74** with `GHI` (`EDA_REPORT.md`,
  correlation section) — strong enough to risk defeating the spec's
  exclusion rule in spirit even while obeying it in letter. **Primary
  model excludes all three.** A secondary, explicitly-labeled ablation
  *including* them is planned to quantify and demonstrate the leakage
  effect for the report's "Analysis & insight" section — not to
  inform the headline result.
- **Safe to use:** `Cloud Type` (verified in EDA to correlate with,
  but not be redundant with, Clear-Sky Index — Section 21), all
  weather columns (`Temperature`, `Relative Humidity`, `Wind Speed`,
  `Wind Direction`, `Dew Point`, `Pressure`, `Surface Albedo`,
  `Precipitable Water`), and time features (`Hour`/`Month`/
  `DayOfYear` cyclical encodings from `src/feature_engineering.py`).
- **Note — reviewed, not a leakage risk (EDA Section 16):**
  `Relative Humidity` and `Wind Direction` have unusual-looking values
  for all of Davis 2013 and all of Huron 2012 (~14% of each city's
  data). **Resolved:** not an error — Relative Humidity can
  legitimately exceed 100%, and `Wind Direction`'s small values for
  those two city-years are valid, most likely a different unit
  (e.g. radians). Both columns are kept; normalize `Wind Direction`'s
  units consistently before use. See `DATASET_PROFILE.md` for detail.

### Target B: Generation-regime (Low/Medium/High, per-city terciles of Output Power)

- **MUST NOT use as a feature:** `Output Power` itself (it's literally
  the value the terciles are computed from).
- **Terciles MUST be computed per city, not globally** — verified in
  EDA Section 19 why: cities' Output Power scales differ up to ~3.6x
  (Section 18), so global terciles would mostly just re-encode "which
  city is this," not "was this a relatively high/low generation
  moment for its own city."
- **Safe to use:** everything else, including `GHI`/`Clearsky GHI`
  (no leakage risk here — the label isn't derived from them).

---

## Problem 2 — Output Power Regression

- **MUST NOT use as a feature:** the current-timestep `Output Power`
  (that's the target).
- **Lagged Output Power (`Output Power_lag1`, `_lag2`, ...) may ONLY
  be used when the forecasting setup makes that lag legitimately
  available** — i.e. for the K=12 sequence-forecasting sub-task,
  where predicting step *t* from steps *t-12...t-1* is the actual
  task definition. For the same-city and cross-city (non-sequence)
  experiments, don't add Output Power lag features — that's not part
  of those experiments' setup and would silently turn them into a
  different (easier) task.
- **Lag features must be built BEFORE any train/test split, on a
  single city's data, sorted chronologically** — see the warning
  already in `src/feature_engineering.py`'s `add_lag_features()`
  docstring. Splitting first and lagging after (or lagging across
  concatenated cities) can leak a test-adjacent value into a training
  lag column.
- **Cross-city experiments:** verified in EDA Section 18 that Output
  Power scales differ up to ~3.6x across cities — use
  `src/preprocessing.py`'s `fit_target_scaler()` (fit on the SOURCE
  city's training data only) when combining cities, and always
  `inverse_transform_target()` before computing final RMSE/MAE/nRMSE.
- **Safe to use:** all weather and irradiance columns, Clear-Sky
  Index, Cloud Type (encoded), time features. No target-derivation
  leakage risk here — `GHI` etc. are legitimate predictive features
  for actual Output Power, unlike for the sky-condition classifier.

---

## Problem 3 — Dimension Reduction

- **Dimension reduction (PCA/Kernel PCA/autoencoder) MUST be fit
  without seeing any target label** — not `Output Power`, not
  sky-condition, not generation-regime. This is inherent to what
  "unsupervised" means, but worth stating explicitly: if the
  DataFrame passed into `fit_scaler()`/PCA-fitting still has a target
  column attached, drop it first via `prepare_xy()`.
- **Standardize features before PCA/Kernel PCA** — both are
  scale-sensitive, and the raw columns here have very different
  natural scales (e.g. `GHI` in the hundreds vs. `Surface Albedo`
  between 0-1) — verified ranges in `EDA_REPORT.md`'s weather-feature
  summary.
- **Downstream evaluation (feeding reduced features into the Problem
  1/2 models) inherits all the leakage rules above** — e.g. if the
  downstream task is sky-condition classification, `GHI`/
  `Clearsky GHI` still shouldn't have been in the feature set that
  went into the reduction step in the first place.

---

## Problem 4 — Semi-Supervised Learning

- **Withheld ("unlabeled") samples' true labels must never be used**
  by the SSL method itself — only their feature values. This sounds
  obvious but is an easy mistake with e.g. graph-based label
  propagation methods, where it's tempting to peek at ground truth
  while tuning the propagation.
- **The labeled/unlabeled split must happen AFTER the train/test
  split, and only within the training portion** — see
  `src/splitting.py`'s `random_labeled_subset()` docstring warning.
  Splitting labels before the chronological split risks leaking
  test-period information into what's treated as "labeled training
  data."
- **The supervised-only baseline and the SSL method must be compared
  using the exact same p% label draw** (same random seed, same rows
  labeled) — otherwise apparent SSL gains could just be sampling
  noise, not a real effect.

---

## Problem 5 — Transfer Learning

- **Zero-shot baseline must see ZERO target-city (Amherst) rows
  during training** — not even for validation/early-stopping. Any
  Amherst data touching the "zero-shot" model at all invalidates that
  baseline's name.
- **Few-shot baseline and the transfer method's fine-tuning step
  should use the SAME small Amherst sample** (same `k`, same seed via
  `src/splitting.py`'s `few_shot_sample()`) — using different samples
  for the two would confound the comparison between them.
- **Source (Davis) and target (Amherst) share zero calendar years**
  (verified in `DATASET_PROFILE.md` and re-confirmed in
  `EDA_REPORT.md`'s city coverage table: Davis 2011-2016, Amherst
  2018-2020) — this transfer problem is simultaneously a cross-city
  AND cross-time-period transfer, which is worth stating plainly in
  the analysis rather than treating it as a clean geography-only
  comparison.
- **Scale difference:** Davis's Output Power is roughly 2.7x
  Amherst's (via the max-ratio table in `EDA_REPORT.md`, Section 18)
  — predictions need rescaling (`fit_target_scaler()` on Davis,
  applied consistently) before comparing zero-shot/few-shot/transfer
  results on Amherst's scale.
