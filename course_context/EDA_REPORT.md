# EDA Report — Phase 3

Internal technical report. Every number below comes from actually
running `scripts/run_eda.py` against the real dataset
(`course/Further Consolidated Data, HnL.xlsx`, 18.0 MB) — nothing here
is estimated or assumed. Full tables are in `results/eda/*.csv`,
figures in `figures/eda/*.png`. Reproducibility: seed 42, Python
3.12.3, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, run 2026-08-27.

Where a fact came from the written project spec vs. was independently
verified against the actual workbook, `DATASET_PROFILE.md` now marks
this explicitly — this report focuses on the verified analysis itself.

---

## Dataset Overview

9 sheets, 22 columns each, all sheets share an identical column set
matching the spec's expected column list exactly (no missing, extra,
or renamed columns found). See `results/eda/dataset_summary.csv` for
the full per-sheet table (rows, first/last timestamp, etc.).

## Sheet / City / Year Structure

| City | Available years | # sheets | Total rows (canonical sheet) |
|---|---|---|---|
| Amherst | 2018-2020 | 1 | 12,056 |
| Davis | 2011-2016 | 2 | 24,112 |
| Huron | 2011-2016 | 2 | 24,112 |
| Santa Barbara | 2011-2016 | 2 | 24,112 |
| La Jolla | 2011-2016 | 2 | 24,112 |

Confirms `DATASET_PROFILE.md`: only 5 distinct city-datasets despite 9
sheets (the four `'14-'16` sheets are exact subsets of their
`'11-'16` counterparts). Amherst shares zero calendar years with any
other city — directly relevant to Problem 5.

## Missing Values

Exactly **4** missing values in the entire workbook, all in
`Output Power`, all in the Amherst sheet, all on **2020-07-06,
10:00-11:30** (4 consecutive rows). This exactly matches the spec's
claim of "~4 missing rows" — verified, not just repeated. No other
column in any of the 9 sheets has a single missing value.

## Duplicates

**Zero** fully-duplicated rows and **zero** duplicate timestamps in
any of the 9 sheets (`results/eda/duplicate_summary.csv`).

## Timestamp Validation

- Every sheet samples exactly 11 timestamps/day: 10:00, 10:30, ...,
  15:00 (30-minute steps, 15:30 excluded).
- **Confirmed discrepancy:** the spec's stated "10:00 to ~14:30"
  window is wrong — verified 2,192 rows (the entire 15:00 slot, every
  day) fall after 14:30. The real window is 10:00-15:00 (a genuine 5
  hours, matching the "5hr-daily" sheet name).
- **Zero missing calendar days** in any city's date range (checked by
  comparing each city's unique dates against a complete calendar range
  for its span).
- See `figures/eda/temporal_sampling.png` for the sampling pattern
  visualization.

## Output Power

| City | N | Mean (kW) | Median | Std | Min | Q25 | Q75 | Max |
|---|---|---|---|---|---|---|---|---|
| Amherst | 12,052 | 60.93 | 60.60 | 41.07 | 0.0 | 20.90 | 100.30 | 132.00 |
| Davis | 24,112 | 164.02 | 178.87 | 67.41 | 0.0 | 117.91 | 222.01 | 262.83 |
| Huron | 24,112 | 50.11 | 53.77 | 16.39 | 0.0 | 42.30 | 62.95 | 77.05 |
| Santa Barbara | 24,112 | 49.08 | 53.83 | 18.31 | 0.0 | 38.60 | 64.19 | 76.65 |
| La Jolla | 24,112 | 47.26 | 50.74 | 15.59 | 0.0 | 38.55 | 59.28 | 72.32 |

(Amherst's N=12,052 instead of 12,056 reflects the 4 missing rows.)
Davis's scale is clearly the outlier — roughly 3x every other city's
mean. See `figures/eda/output_power_by_city_boxplot.png` and
`output_power_distribution.png`.

## Output Power by Time

`figures/eda/output_power_by_time.png` shows the expected midday peak
(sampling only covers 10:00-15:00) and a summer peak in every city
(more daylight / higher sun angle). No surprises — included as a
sanity check that the data behaves physically reasonably, which it
does.

## Irradiance

Mean GHI ranges 451.6 W/m² (Amherst, cloudier/more northern climate)
to 673.1 W/m² (Santa Barbara). Full table:
`results/eda/irradiance_summary_by_city.csv`.
`figures/eda/ghi_vs_power.png`: GHI shows the visually strongest,
most linear relationship with Output Power; DHI (diffuse-only light)
shows a much weaker, noisier one — consistent with PV panels
responding much more strongly to direct sunlight than to scattered/
diffuse light.

## Clear-Sky Index

`k = GHI / Clearsky GHI`, computed as a derived column (raw dataset
never modified). 0 missing (no `Clearsky GHI == 0` rows in this
daytime-only dataset), 0 infinite (verified, not just assumed — the
safe-division code prevents this by construction), 92 rows above 1.5
(physically extreme — brief instrument spikes, not investigated
further this phase).

**Threshold check (spec's Clear ≥0.85 / Partly cloudy 0.4-0.85 /
Overcast <0.4):** produces 72.0% / 18.7% / 9.3% of rows respectively.
None of the three classes is vanishingly small — **the thresholds
look usable as given.** This is a finding, not a change: the
thresholds have not been altered.

## Cloud Type

11 of the 13 possible NSRDB codes actually appear. Clear (code 0)
dominates at 59.65%. Three codes — Overshooting (0.08%), Unknown
(0.00%, 1 row), Smoke (0.00%, 1 row) — are each under 1% of the data.
Full table: `results/eda/cloud_type_distribution.csv`,
`figures/eda/cloud_type_distribution.png`.

## Weather Features

Summary table: `results/eda/weather_feature_summary.csv`. All columns
have 0% missing. One genuine data-quality issue found (not previously
documented) — see below.

## Correlations

`figures/eda/correlation_heatmap.png`. Pooled-across-cities, GHI vs.
Output Power correlation is only **0.432** — but this number is
**misleading on its own**: computed **within each city separately**,
correlations are 0.75-0.97 (Davis 0.97, Amherst 0.80, Huron 0.93,
Santa Barbara 0.75, La Jolla 0.84 —
`results/eda/ghi_power_correlation_by_city.csv`). Pooling cities with
very different Output Power *scales* (see Target Scale Analysis below)
dilutes the linear correlation even though the physical relationship
is strong in every single city. Solar Zenith Angle vs. GHI:
**-0.740** (higher sun angle number = lower elevation = less direct
sunlight, as expected). Correlation is not causation — the GHI/Output
Power relationship is trusted here because it matches known PV
physics, not because correlation itself proves causality.

## Temporal Dependence

Output Power autocorrelation (Davis): **lag1 = 0.945, lag2 = 0.879,
lag6 = 0.710**. GHI lag1 autocorrelation: **0.938**. (Note: with only
11 samples/day, "lag 1" means the next 30-minute sampled slot, not a
continuous 24/7 series — see `scripts/run_eda.py` for this caveat in
code.) Strong lag-1 autocorrelation supports the project's
sequence-forecasting sub-task (K=12) and lag-feature/RNN-LSTM-GRU
approaches — adjacent readings carry real predictive signal, which is
the basic assumption those methods rely on. No sequence model was
built this phase.

## Class Imbalance Preview

**Sky-condition** (all cities pooled): Clear 72.04%, Partly cloudy
18.67%, Overcast 9.29%
(`results/eda/sky_condition_distribution.csv`, and by-city in
`sky_condition_distribution_by_city.csv`).

**Generation-regime:** by construction, per-city terciles give each
city ~33/33/33 Low/Medium/High
(`results/eda/generation_regime_distribution.csv`). Terciles
**must** be computed per-city (not globally) — verified reason: Output
Power scales differ up to 3.6x across cities (below), so global
terciles would mostly encode "which city is this," not "was this a
relatively high output moment for its own city."

## Cross-City Scale Differences (Target Scale Analysis)

| City | Mean (kW) | Max (kW) | Max/Mean | Max ratio vs. smallest city |
|---|---|---|---|---|
| Amherst | 60.93 | 132.00 | 2.17 | 1.83 |
| Davis | 164.02 | 262.83 | 1.60 | **3.63** |
| Huron | 50.11 | 77.05 | 1.54 | 1.07 |
| Santa Barbara | 49.08 | 76.65 | 1.56 | 1.06 |
| La Jolla | 47.26 | 72.32 | 1.53 | 1.00 |

Davis's max is **3.63x** La Jolla's. Directly relevant to: cross-city
regression (Problem 2 — combining cities without per-city target
scaling would let Davis dominate), transfer learning (Problem 5 —
Davis/Amherst ratio is 1.99x), and why the spec requires nRMSE
alongside raw RMSE. `src/preprocessing.py`'s target-scaling functions
(built Phase 2) exist specifically for this.

## Data Quality Issues (new findings this phase)

**1. Relative Humidity / Wind Direction anomaly — Davis 2013 and
Huron 2012 (previously undocumented).** `Relative Humidity` values
above 100% (up to 360) were investigated properly rather than just
flagged. Finding: this is **not random noise** — it's isolated to
**exactly one full calendar year each** in two cities:

| City | Affected rows | % of city | Affected years | Wind Dir. range (affected) | Wind Dir. range (normal) |
|---|---|---|---|---|---|
| Davis | 3,480 | 14.43% | 2013 only | 0.36-4.47 | 0.0-360.0 |
| Huron | 3,287 | 13.64% | 2012 only | 0.37-3.79 | 0.0-360.0 |

In the affected rows, `Relative Humidity` reaches values up to 360
(exactly Wind Direction's normal 0-360° range), while `Wind Direction`
sits under ~4.5 (consistent with radians, or some other unit entirely
— nowhere near the normal 0-360° range seen in every other year).
This strongly suggests the two columns were swapped and/or recorded
in a different unit for these two specific city-years. Not corrected
here — see Decisions below.

**2. Confirmed from Phase 0:** 2012-03-22 zero-Output-Power anomaly
across all four CA/inland cities despite normal irradiance —
unchanged from `DATASET_PROFILE.md`, re-confirmed present in this
phase's canonical dataset.

**3. Confirmed from Phase 0:** spec's "10:00-14:30" window
description is wrong; real window is 10:00-15:00 (Timestamp
Validation, above).

## Leakage Risks

Full detail in `course_context/LEAKAGE_MAP.md`. Summary: (1) `GHI`/
`Clearsky GHI` must be excluded from the sky-condition classifier's
features — they define the label; (2) `DHI`/`DNI`/`Solar Zenith Angle`
together can approximately reconstruct `GHI`, worth an ablation; (3)
generation-regime terciles must be per-city; (4) Output Power lag
features are only legitimate for the sequence-forecasting sub-task,
and must be built before any split; (5) cross-city/transfer
experiments need target scaling, fit on training/source data only.

## Decisions That Need To Be Made Later

1. **The Davis-2013 / Huron-2012 Relative Humidity/Wind Direction
   anomaly** (new finding, above): exclude those two city-years'
   affected rows, exclude `Relative Humidity`/`Wind Direction` as
   features project-wide for consistency, attempt to reconstruct/
   correct the swap, or accept the noise and document it as a
   limitation. Not decided in this phase.
2. Sky-condition classifier: whether to run with or without
   `Solar Zenith Angle`/`DHI`/`DNI` as features (leakage-adjacent, not
   leakage-certain) — recommend an ablation rather than a single
   choice.
3. Problem 4's SSL algorithm choice (course-taught vs. pseudo-labeling)
   — still open from Phase 0, unaffected by this phase's findings.
4. Whether to use the 3-year vs. 6-year sheet pairs for the optional
   Problem 2 ablation — the data supports it (confirmed again this
   phase); still a "when we get there" decision, not urgent now.
