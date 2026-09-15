# My Defense Study Guide

Built from the project's actual saved results and reports — every
number here traces to `results/FINAL_EXPERIMENT_TABLE.csv` or a
`course_context/PROBLEMN_REPORT.md`. This is a study aid, not a script
to memorize word-for-word — the goal is being able to explain each
idea in your own words.

## 1. Project Overview

The project predicts photovoltaic (solar) power output across five
U.S. cities using five different machine learning paradigms:
classification, regression, dimension reduction, semi-supervised
learning, and transfer learning. Rather than one model doing one job,
each paradigm gets its own fair test against its own appropriate
baseline, so you can honestly say which techniques helped and which
didn't — not just report five sets of numbers.

**The one-sentence version:** solar power can't be turned up on
demand like a gas plant — it depends entirely on the weather — so
predicting it accurately matters for grid planning, and this project
tests five different ML approaches to that prediction problem.

## 2. Dataset Facts

- 5 cities: **Davis, CA** (2011-2016, 24,112 rows), **Huron, SD**
  (2011-2016, 24,112 rows), **Santa Barbara, CA** (2011-2016, 24,112
  rows), **La Jolla, CA** (2011-2016, 24,112 rows), **Amherst, MA**
  (2018-2020, 12,056 rows).
- 30-minute readings, daytime window only (10:00-15:00).
- 22 columns per row: irradiance (GHI, DNI, DHI + clear-sky
  equivalents), weather (Temperature, Relative Humidity, Wind Speed,
  Wind Direction, Dew Point, Pressure, Surface Albedo, Precipitable
  Water), Cloud Type (categorical), Solar Zenith Angle, and the
  target, **Output Power** (kW).
- Only 4 missing rows in the entire dataset (Amherst, one date) -
  dropped, not interpolated.
- **The central challenge:** Output Power has a different scale in
  every city - Davis averages 164.0 kW, Amherst 60.9 kW, Huron 50.1
  kW, Santa Barbara 49.1 kW, La Jolla 47.3 kW. Davis is roughly
  2.7-3.5x the others. This is a plant-size difference, not a
  weather-quality difference.

## 3. Important Preprocessing Decisions

- **Clear-Sky Index** (`k = GHI / Clearsky GHI`) - measures how close
  actual irradiance is to the theoretical maximum; defines Problem 1's
  sky-condition label.
- **Cyclical time encoding** - hour/month/day-of-year converted to
  sine/cosine pairs, so e.g. 23:00 and 00:00 are close together
  numerically, not maximally far apart like raw integers would make
  them.
- **Cloud Type is categorical, not numeric** - one-hot encoded
  everywhere. Code 7 isn't "more" than code 3.
- **Chronological 80/20 split, always** - earliest 80% of a city's
  timestamps train, latest 20% test. Never a random shuffle, because
  that would let a model trained on future data predict the past.
- **Scalers/encoders fit on training data only**, then applied
  unchanged to test data - verified directly in the code, never fit on
  combined or test data.

## 4. Problem 1 - Supervised Classification

Two tasks: **sky-condition** (Clear/Partly Cloudy/Overcast, from
thresholding Clear-Sky Index) and **generation-regime** (Low/Medium/
High Output Power, per-city terciles). GHI, Clearsky GHI, DHI, DNI,
and Solar Zenith Angle are all excluded from sky-condition's features
- they define or can reconstruct the label, so including them would
leak the answer (tested directly: including them pushed accuracy from
~0.74-0.81 to ~0.98, proving the leak was real).

**Best result:** logistic regression (balanced class weight), Davis
sky-condition - 0.772 balanced accuracy, 0.720 macro F1. Notably not
an ensemble or neural net. Clear conditions were easiest (F1=0.946);
Partly Cloudy was hardest (F1=0.566) because it sits between the other
two classes and gets confused with both.

## 5. Problem 2 - Supervised Regression (the core task)

Predicts Output Power using the full weather+irradiance feature set
(no leakage restriction here - Output Power isn't derived from a rule
applied to these columns). Three settings: same-city, cross-city
zero-shot, and K=12 sequence forecasting.

**Best result:** gradient boosting, Davis same-city - RMSE=15.17 kW,
MAE=8.31 kW, R²=0.953. The strongest single result in the project.

**Cross-city zero-shot failed severely** (RMSE ~125 kW, Davis->Huron/
Santa Barbara/La Jolla) - almost entirely the scale-mismatch problem
again. A diagnostic rescaling (not a real zero-shot method) recovered
R²=0.55-0.84, proving the underlying weather-power relationship
transfers fine once scale is handled.

**Sequence GRU (K=12) beat a persistence baseline** (17.58 kW vs.
21.97 kW) - real temporal learning - but doesn't beat the non-sequence
model (15.17 kW), which is a fundamentally easier task (it sees the
concurrent weather reading; the GRU only sees the past).

## 6. Problem 3 - Dimension Reduction

Compressed the 23-feature space with PCA (linear) and a small
autoencoder (nonlinear) at d=2, 5, 10 - fit with zero label access, on
training features only.

**The finding: raw, uncompressed features beat every reduced
representation, at every dimension, on both tasks.** Raw: 0.743
balanced accuracy / 23.86 kW RMSE. Best reduced (Autoencoder, d=10):
0.661 / 29.69 kW. The autoencoder consistently beat PCA (nonlinear
compression helps) but neither closed the gap with raw features.
Traced to a specific cause: Cloud Type is highly predictive but
compresses poorly into a handful of continuous dimensions (verified:
removing it raised explained variance but lowered both reconstruction
quality and downstream accuracy).

## 7. Problem 4 - Semi-Supervised Learning

Hid most of Davis's sky-condition labels (down to 10%), tested whether
pseudo-labeling (train on the small labeled set, predict on unlabeled
data, keep confident predictions as new "labels," retrain) beats
training on the labeled subset alone.

**The finding: SSL did not help.** Gain was slightly negative at every
fraction: 10% (-0.0048 macro F1), 30% (-0.0006), 50% (-0.0010).
Diagnosed why: pseudo-labels were 100% accurate (checked via hidden
labels, after training only - never used to influence training) but
98% of the first round were the easy "Clear" class - redundant with
what the model already knew, not wrong information.

## 8. Problem 5 - Transfer Learning

Can Davis (6 years, data-rich) help predict Amherst (3 years,
data-poor) Output Power? Compared zero-shot (no Amherst training),
few-shot (fresh model, k Amherst samples only), and transfer
(Davis-pretrained model, fine-tuned on the same k samples).

**The finding: transfer clearly beat few-shot at every k, most at
k=10** (RMSE 30.03 vs. 38.22 kW, +21.4%), shrinking to +0.9% at k=100
as few-shot catches up with more real data.

**The most interesting story:** an early version used a smaller
fine-tuning learning rate (following the assignment's own suggestion)
and produced severe negative transfer - RMSE 140.9 kW at k=10, worse
than few-shot. Diagnosed: the model's output was stuck near Davis's
~164 kW scale, because the learning rate was too small to shift it to
Amherst's ~64 kW scale in time. Matching the pretraining learning rate
instead fixed it completely (RMSE dropped to 27.2 kW in that
diagnostic run). Two follow-up ablations: target normalization showed
no extra benefit once the LR was fixed (23.35 kW raw vs. 24.11 kW
normalized at k=50); freezing the first layer during fine-tuning also
showed no meaningful difference (23.35 vs. 23.40 kW).

## 9. Important Results (headline numbers)

| Problem | Best result |
|---|---|
| P1 | 0.772 balanced accuracy (Davis sky-condition, logistic regression) |
| P2 | RMSE=15.17 kW, R²=0.953 (Davis same-city, gradient boosting) |
| P3 | Raw features win: 0.743 bal. acc / 23.86 kW RMSE beats every compressed version |
| P4 | SSL gain ~ -0.001 to -0.005 at every label fraction - did not help |
| P5 | Transfer +21.4% RMSE improvement at k=10 |

## 10. Important Metrics

- **RMSE** (root mean squared error) - penalizes large errors more
  than small ones (squares them before averaging); same units as the
  target (kW).
- **MAE** (mean absolute error) - average error size, doesn't
  over-penalize big misses; easier to interpret directly.
- **nRMSE** - RMSE divided by the target's range (max-min); lets you
  compare error across cities/targets with very different scales.
- **Balanced accuracy** - averages per-class recall, so a majority
  class (like "Clear" at 72% of the data) can't inflate the score.
- **Macro F1** - averages F1 across classes equally, same
  imbalance-resistance idea as balanced accuracy.
- **R²** - fraction of variance explained; 0.953 means the model
  explains 95.3% of Output Power's variation.

## 11. Important Limitations

- 5 cities, mostly 6 years - enough to demonstrate methods, not to
  claim broad generalization.
- Different plant scales across cities is a recurring complication,
  not a fully solved problem.
- Hyperparameter searches were deliberately kept small (3-5
  candidates) throughout.
- No GPU during development - all training was CPU-only (will use
  CUDA automatically on GPU hardware, no code changes needed).
- No real-time weather forecast input - every experiment uses
  historical, already-measured weather, not a forecast of future
  weather.
- One flagged-but-unresolved anomaly: Wind Speed's domain-shift
  difference between Davis and Amherst is unusually large (SMD=2.61)
  - possibly a sensor/unit artifact, not confirmed either way.

## 12. Common Professor Questions

1. Why chronological, not random, splitting? - Random shuffling on
   time-series data lets the model see the future; every metric would
   be meaningless for real forecasting.
2. Why exclude GHI from sky-condition classification? - It defines the
   label; including it leaks the answer (proven: accuracy jumped to
   ~0.98 when included).
3. Why didn't dimension reduction help? - Traced to Cloud Type
   compressing poorly into continuous dimensions, verified via
   ablation.
4. Why didn't SSL help? - Pseudo-labels were correct but redundant
   (concentrated on the easy class), not wrong.
5. What is negative transfer, and did you see it? - Yes, directly: an
   early Problem 5 version with too-small a fine-tuning learning rate
   performed worse than training from scratch. Found, diagnosed, and
   fixed.
6. How do you know your results aren't leaking? - Every scaler is fit
   on training data only; test sets appear only in prediction/metric
   calls, never in training calls - verified directly in the code
   during two separate audits.

## 13. Strong Short Answers

- **"What's your best result?"** -> Davis same-city regression, RMSE =
  15.17 kW, R² = 0.953.
- **"What's your most interesting finding?"** -> In Problem 5, the same
  transfer setup was either a severe failure or a clear win depending
  only on the fine-tuning learning rate - found, diagnosed, fixed.
- **"What didn't work?"** -> Dimension reduction hurt performance at
  every tested size, and semi-supervised learning's gain was
  essentially zero - both honestly reported, both explained.
- **"Why five different paradigms?"** -> To test, fairly and
  separately, which ML techniques genuinely help this kind of
  forecasting problem, instead of assuming any of them automatically
  would.

## 14. Concepts I Personally Struggled With

(from this study session - worth extra review before the real defense)

- **Classification vs. regression** - got these backwards initially.
  Fix: "could the answer be any number?" -> regression. "Is it one of a
  short fixed list?" -> classification. Problem 1 = classification
  (Clear/Partly Cloudy/Overcast, Low/Medium/High). Problem 2 =
  regression (exact kW value).
- **Why Davis and Amherst's power numbers differ** - first explanation
  reached for "different weather," but the actual, provable cause is
  plant *scale* (installation size), not weather quality. Davis
  averages 164.0 kW, Amherst 60.9 kW - roughly 2.7x, and this scale
  gap is exactly what caused the Problem 5 zero-shot failure and the
  early negative-transfer episode.
