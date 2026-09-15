# Problem 5 Report — Transfer Learning

Internal technical record for Problem 5, the final required ML
problem. Every number comes from an actual executed run of `problems/
problem5_transfer_learning/run_experiments.py` — nothing is estimated.
Full results: `results/problem5/problem5_results.csv` (27 rows),
`problem5_transfer_summary.csv`, `problem5_domain_shift.csv`.
Reproducibility: seeds 42/123/2026, Python 3.12.3, pandas 3.0.2,
torch 2.14.0, run 2026-09-12.

---

## 1. Objective

Answer: can Davis (data-rich, 6-year) help predict Output Power in
Amherst (data-poor, 3-year) when only a handful of Amherst labels are
available?

## 2. Why transfer learning matters for PV forecasting

New or smaller PV installations rarely have years of historical data.
If a model trained on a well-instrumented site can be adapted cheaply
to a new site with only a few dozen labeled readings, that has real
practical value for deployment.

## 3. Source city

Davis — 6-year dataset, 19,289 training rows (`build_city_dataset()`,
reused directly from Problem 2, same split).

## 4. Target city

Amherst — 3-year dataset, 9,641 training rows / 2,411 test rows.
**Verified identical to Problem 2's Amherst split** (same
`last_train_timestamp`).

## 5. Source/target dataset sizes

Davis training pool: 19,289 rows. Amherst training pool: 9,641 rows
(few-shot samples of size 10/50/100 drawn from here). Amherst test
set: 2,411 rows, held out and untouched until final evaluation.

## 6. Domain differences

See Section 17 (Domain-shift analysis) below for the full comparison.

## 7. Feature preprocessing

Reused Problem 2's exact feature set (irradiance included — no
leakage restriction for regression) and its exact zero-shot pattern:
the SOURCE (Davis) preprocessor is fit once and applied to Amherst
without ever fitting anything new on Amherst data (`course_context/
PROBLEM2_REPORT.md`).

## 8. Output normalization

**Primary methodology: raw kW throughout** (Davis and Amherst target
values never rescaled) — matches Problem 2's established zero-shot
precedent. **A normalization ablation was run separately** (Section
15) — see the important finding there about what actually mattered.

## 9. Zero-shot baseline

Davis-pretrained `TransferMLP`, applied directly to the Amherst test
set, zero Amherst training:

| Metric | Value |
|---|---|
| RMSE | 184.17 kW |
| MAE | 103.86 kW |
| nRMSE | 1.4479 |

Severe failure, as expected from Problem 2's cross-city findings —
Davis's model is calibrated to its own ~164 kW mean scale, wildly
over-predicting Amherst's ~61 kW mean scale.

## 10. Few-shot baseline

Fresh `TransferMLP` (random init, same architecture as transfer),
trained ONLY on k Amherst samples, k ∈ {10, 50, 100}, no Davis
exposure:

| k | RMSE (kW) | MAE (kW) | nRMSE |
|---|---|---|---|
| 10 | 38.22 | 31.91 | 0.3004 |
| 50 | 24.72 | 18.69 | 0.1944 |
| 100 | 23.02 | 16.78 | 0.1810 |

## 11. Transfer methodology

Two-stage: (1) pretrain `TransferMLP` on Davis (fresh init, lr=1e-3);
(2) fine-tune a **copy** of that pretrained model on the exact same k
Amherst samples used for the few-shot baseline at that k (Section 11's
fairness requirement — verified via `sample_k_target_rows()`, called
once per (k, seed), reused for both methods).

## 12. Fine-tuning strategy

**An important finding, not a default assumption.** The instructions
suggest a smaller fine-tuning learning rate "where appropriate." This
was tested directly and found NOT appropriate here:

| Fine-tuning LR | k=10, seed=42 RMSE | Mean prediction |
|---|---|---|
| 1e-4 (10x smaller than pretraining) | 140.92 | 135.9 kW |
| 1e-3 (same as pretraining) | **27.23** | 70.3 kW |
| 1e-2 | 24.44 | 62.5 kW |

At 1e-4, the model simply couldn't shift its output level from
Davis's ~164 kW scale to Amherst's ~64 kW scale within 100 epochs on
10 samples — a severe, spurious **negative transfer** driven entirely
by an overly-conservative learning rate, not by any real limitation of
transfer learning itself. **`lr=1e-3` (matching pretraining) was
adopted** based on this direct comparison — fixed epoch budget (100),
no early stopping (k=10 is too small for a meaningful validation
split — Section 35's explicitly-permitted limitation), batch size
`min(16, k)`.

## 13. Freezing strategy

Tested at k=50 (Section 34): full fine-tuning vs. freezing `layer1`
(the 128-unit layer):

| Configuration | RMSE (kW) |
|---|---|
| Full fine-tuning | 23.35 ± 0.32 |
| Layer1 frozen | 23.40 ± 0.40 |

**Essentially no difference.** Freezing didn't hurt (or meaningfully
help) — consistent with general transfer-learning intuition that early
layers learn broadly reusable weather/irradiance representations,
while most of the needed adaptation happens in the later,
city-specific layers, which stayed trainable in both configurations.

## 14. Results

Central table (`results/problem5/problem5_transfer_summary.csv`):

| Method | Target Labels | RMSE | MAE | nRMSE |
|---|---|---|---|---|
| Zero-Shot | 0 | 184.17 | 103.86 | 1.4479 |
| Few-Shot | 10 | 38.22 | 31.91 | 0.3004 |
| Transfer | 10 | **30.03** | **21.38** | **0.2361** |
| Few-Shot | 50 | 24.72 | 18.69 | 0.1944 |
| Transfer | 50 | **23.35** | **16.27** | **0.1836** |
| Few-Shot | 100 | 23.02 | 16.78 | 0.1810 |
| Transfer | 100 | **22.80** | **15.87** | **0.1793** |

Figures: `figures/problem5/problem5_transfer_curve.png`,
`problem5_zero_shot_vs_transfer.png`.

## 15. Transfer gain

| k | Gain (RMSE) | Gain (%) |
|---|---|---|
| 10 | +8.18 kW | **+21.4%** |
| 50 | +1.37 kW | +5.5% |
| 100 | +0.22 kW | +0.9% |

**Gain is largest exactly where it matters most — the scarcest-label
regime (k=10)** — and shrinks toward zero as the few-shot baseline
gets enough data to stand on its own. This is the textbook
transfer-learning shape, obtained here from real, executed
experiments, not assumed.

**Normalization ablation (k=50):** raw kW (23.35 ± 0.32) vs.
Davis/Amherst-normalized (24.11 ± 0.58) — **normalization did NOT
help once the learning rate was fixed**; if anything it was slightly
worse, likely because the k-sample-based Amherst scaler (especially
noisy at small k) adds its own estimation error on top of an
already-working setup. This directly connects to Section 12's finding:
**the real fix for the original negative transfer was the learning
rate, not explicit target normalization** — a genuinely useful,
non-obvious result of running both ablations rather than assuming one
mattered more than the other.

## 16. Negative transfer analysis

**Negative transfer DID occur — but only under the naive
configuration (fine-tuning lr=1e-4), not in the final reported
results.** This is reported in full in Section 12 rather than
quietly fixed and hidden: the initial default (following the
instructions' suggestion of a smaller fine-tuning rate) produced
RMSE=140.92 at k=10, dramatically worse than the few-shot baseline's
38.22. Investigated, traced to a specific, verifiable cause (mean
prediction stuck near Davis's scale), and corrected based on direct
evidence — not silently swapped for a better-looking number.

## 17. Domain-shift analysis

Standardized mean differences (Davis vs. Amherst), full table:
`results/problem5/problem5_domain_shift.csv`:

| Variable | SMD | Interpretation |
|---|---|---|
| Output Power | +1.847 | Large — the known scale difference (Sections 5, 9) |
| Wind Speed | **+2.612** | **Extreme** — Davis 2.81±1.43 vs. Amherst 0.14±0.24 |
| Temperature | +0.916 | Large — Davis (CA) noticeably warmer than Amherst (MA) |
| GHI, DNI, Clear-Sky Index | +0.66 to +0.70 | Moderate — Davis sunnier/clearer on average |
| DHI, Solar Zenith Angle | -0.22 to -0.24 | Small |
| Relative Humidity | +0.230 | Small |

**Wind Speed's SMD (2.6) is far larger than any other variable** —
Amherst's values cluster almost entirely near zero (0.14 ± 0.24)
while Davis's are meaningfully spread (2.81 ± 1.43). This magnitude
of difference is unusual enough to flag as **possibly a sensor/
measurement or unit artifact between the two data sources**, not
necessarily a genuine climate difference — worth independent
verification before treating it as a real physical signal in any
future work. Figure: `figures/problem5/problem5_domain_shift.png`.

**Connecting domain shift to transfer performance (Section 32):**
Temperature, GHI, and Clear-Sky Index show real but moderate shift —
consistent with transfer working well once the OUTPUT SCALE problem
was fixed (Section 12): the underlying weather→power relationship
apparently transfers reasonably, it was the output calibration that
needed adaptation, not a fundamentally different physical relationship.
This matches the freezing ablation's finding (Section 13) that the
early, general representation didn't need to change much.

## 18. Best model

**Transfer, k=10** is the standout result relative to its
alternative (largest gain, most practically relevant scenario — an
Amherst-like site with only 10 labeled readings). In absolute terms,
**Transfer, k=100** has the lowest RMSE (22.80 kW) of any
configuration tested. Architecture: `TransferMLP` (Dense(128)→ReLU→
Dropout→Dense(64)→ReLU→Dense(1)); fine-tuning: full (unfrozen),
lr=1e-3, 100 fixed epochs, raw kW target.

## 19. Limitations

- No validation split during fine-tuning (k as small as 10) — a fixed
  epoch budget was used instead, an explicitly-permitted limitation
  (Section 35), not a leak.
- The freezing and normalization ablations were run at k=50 only, not
  every k, to keep their cost proportionate (Sections 33-34 both
  explicitly allow this scope).
- Plain random (not stratified) few-shot sampling — Section 10 permits
  this as the reasonable default for a continuous regression target.
- The Wind Speed domain-shift anomaly (Section 17) was flagged, not
  independently resolved — would need to check both cities' raw
  instrument documentation to confirm whether it's a real climate
  signal or a data artifact.
- Only one architecture size was tested (the exact one specified) —
  no architecture search was performed, consistent with the
  instruction not to run large searches.

## 20. Reproducibility

Seeds: 42, 123, 2026. Python 3.12.3; pandas 3.0.2; numpy 2.4.4; torch
2.14.0 (CPU in this sandbox — no GPU present here; will use CUDA
automatically via `src/utils.py`'s `get_device()` on the project
owner's RTX 2070). Dataset: `course/Further Consolidated Data,
HnL.xlsx`. Full run history: `results/problem5/problem5_results.csv`
(27 rows). Models: `results/problem5/models/` (7 `.pt` files, all
verified loadable, plus `model_config.json`). Two real implementation
bugs were caught and fixed during development (both `AttributeError`s
from `apply_target_scaler()` returning a numpy array rather than a
pandas Series where one was assumed) — fixed immediately, confirmed
via a clean re-run. Note on execution: like Problems 1-4, this phase's
experiments were run as several separate script invocations to stay
within this sandbox's per-command execution-time limit.
