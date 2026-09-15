# Project Story

Not the final report — a concise outline the eventual report will
draw from. Every claim here traces to a specific problem report.

## 1. Problem motivation

Predicting photovoltaic (PV) solar power output matters for grid
planning and operations, but real deployments face two hard
constraints: (a) new or smaller installations rarely have years of
labeled history, and (b) even well-instrumented sites need models that
generalize across weather regimes, not just fit historical averages.
This project uses a 5-city, multi-year irradiance/weather/PV-output
dataset to explore five distinct ML paradigms against those exact
constraints.

## 2. Dataset

9 sheets, 5 cities (Amherst MA, Davis CA, Huron SD, Santa Barbara CA,
La Jolla CA), 2011-2020 depending on city, ~22 columns per row
(irradiance, weather, Cloud Type, Output Power). Verified clean (0
duplicate rows, 4 missing values total) with one genuinely
undocumented data-quality finding (a Relative Humidity/Wind Direction
anomaly, reviewed and accepted per project-owner guidance — see
`EDA_REPORT.md`).

## 3. Problem 1 finding

A 3-class sky-condition classifier reached 0.77-0.82 balanced accuracy
(Davis/Amherst), and a 3-class generation-regime classifier reached
0.76-0.95. The strongest sky-condition model was, unexpectedly, plain
logistic regression — not an ensemble or neural net. A leakage
ablation directly proved the exclusion of DHI/DNI/Solar Zenith Angle
mattered: including them inflated accuracy to ~0.98, confirming they
could reconstruct the excluded label-defining features.

## 4. Problem 2 finding

Same-city regression (Davis, gradient boosting) reached RMSE=15.17 kW,
R²=0.953 — the project's strongest single result. Cross-city zero-shot
transfer failed severely in raw kW (R² as low as -72) purely from
scale mismatch — a diagnostic rescaling showed the underlying
weather-to-power relationship was actually well captured (R² 0.55-0.84
once rescaled). A K=12 GRU sequence model beat a naive persistence
baseline clearly, though it's solving a fundamentally harder task
(true forecasting) than the same-city model, so the two aren't
directly comparable.

## 5. Problem 3 finding

PCA and a small autoencoder were compared at d=2/5/10 against raw
features, on both downstream tasks. **Raw features won at every
dimension, on both tasks.** The autoencoder consistently beat PCA
(nonlinear compression helps), but neither closed the gap with using
all 23 original features. A feature ablation traced part of this to
Cloud Type compressing poorly despite being highly predictive.

## 6. Problem 4 finding

Pseudo-labeling/self-training was compared against a supervised-only
baseline at 10%/30%/50% labels. **SSL gain was slightly negative at
every fraction.** An offline diagnostic (hidden labels, post-hoc only)
showed pseudo-labels were 100% accurate but concentrated on the
already-easy "Clear" class — redundant confirmation, not new
information about the harder classes. A necessary safeguard (capping
pseudo-labels per iteration) was found empirically to prevent worse
class-collapse behavior.

## 7. Problem 5 finding

Davis→Amherst transfer learning (zero-shot, few-shot, and fine-tuned
transfer) was compared at k=10/50/100 target samples. Zero-shot failed
severely (RMSE=184 kW, same raw-scale problem as Problem 2). Transfer
beat few-shot at every k, most dramatically at k=10 (+21.4% RMSE
improvement) — the textbook "transfer helps most when data is
scarcest" shape. A genuine negative-transfer episode was found and
fixed during development: an overly-conservative fine-tuning learning
rate (following the assignment's own suggestion) caused RMSE=140.9 at
k=10; matching pretraining's learning rate fixed it to RMSE=27.2.

## 8. Most interesting result

The Problem 5 learning-rate finding: a single hyperparameter choice
(fine-tuning LR) was the difference between severe negative transfer
and transfer clearly beating few-shot — and a separate ablation
(target normalization) that might plausibly have been "the fix" turned
out not to matter once the LR was corrected. Two ablations, run
independently, pointed to different — and correctly disambiguated —
root causes.

## 9. Most surprising failure

Problem 3's headline finding: dimension reduction, the entire point of
which is usually to help or at least not hurt, hurt performance on
BOTH downstream tasks at every tested dimension. This wasn't an
implementation problem — it was traced to a specific, verifiable cause
(Cloud Type's categorical information compressing poorly).

## 10. Main limitation

No single limitation dominates, but the recurring theme is scope
discipline: hyperparameter searches were deliberately small
(3-5 candidates) throughout, per the project's own "do not overengineer"
instructions, and repeatedly found that tuning barely moved results
(Problems 1, 2, and 4 all found untuned or lightly-tuned models beat
tuned ones at least once). A larger search might find more, but the
consistent pattern across three separate problems suggests the
dataset/model combinations here are already close to what small
model-family changes can achieve.

## 11. Main takeaway

Every advanced technique tested (dimension reduction, semi-supervised
learning, transfer learning) was evaluated against a strong, honest
raw/supervised/few-shot baseline — and the results were genuinely
mixed: transfer learning clearly helped, dimension reduction clearly
hurt, and semi-supervised learning was a wash. This mixed picture,
obtained by actually running the comparisons rather than assuming an
answer, is the project's real contribution — a fair, evidence-based
read on when each paradigm is worth using for this kind of data.
