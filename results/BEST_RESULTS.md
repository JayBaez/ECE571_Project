# Best Results — All Five Problems

Every number below is pulled directly from `results/FINAL_EXPERIMENT_TABLE.csv`
and each problem's own report — nothing here is re-estimated.

---

## PROBLEM 1

**Best task:** sky-condition (Davis) and generation-regime (Davis) are
the two standout results; Amherst is included for completeness.

**Best method:** `logistic_regression_balanced_weight` (Davis
sky-condition), `mlp_tuned` (Davis generation-regime).

**Balanced Accuracy:** 0.7720 (Davis sky-condition), 0.9460 (Davis
generation-regime).

**Macro F1:** 0.7200 (Davis sky-condition), 0.9453 (Davis
generation-regime).

**Why this is the selected result:** chosen purely by mean balanced
accuracy across 3 seeds among every model tried (majority baseline
through tuned MLP) — not hand-picked. Notably, the strongest
sky-condition model is a plain logistic regression, not an ensemble
or neural net — a genuinely useful finding for the report (simpler
isn't always weaker).

---

## PROBLEM 2

**Best setup:** same-city Davis regression.

**Best method:** `gradient_boosting` (untuned — beat every tuned
variant, a repeated pattern across this project).

**RMSE:** 15.17 kW · **MAE:** 8.31 kW · **nRMSE:** 0.0600.

**Why this is the selected result:** highest R² (0.953) and lowest
RMSE of every model/city/task combination tested in Problem 2,
including the sequence model. The sequence GRU (17.58 kW) is a
genuinely different, harder task (true forecasting, no concurrent
weather) and is reported separately, not as a worse version of this
result — see `PROBLEM2_REPORT.md`, Section 20.

---

## PROBLEM 3

**Best classification representation:** Autoencoder, d=10.
**Balanced Accuracy:** 0.6606 (still below raw features' 0.7431).

**Best regression representation:** Autoencoder, d=10.
**RMSE:** 29.69 kW **MAE:** 21.88 kW (still above raw features' 23.86
kW RMSE / 15.84 kW MAE).

**Best dimensionality:** d=10 — the largest tested, for both methods
and both tasks. Cumulative explained variance at d=10 is 0.9455.

**Why:** **raw features beat every reduced representation at every
tested dimension**, on both downstream tasks — the honest finding, not
adjusted. Among the reduced representations, the autoencoder
consistently beat PCA (nonlinear compression genuinely helped, just
not enough to close the gap with using all the original features).
See `PROBLEM3_REPORT.md`, Sections 16-17, for why (Cloud Type
compresses poorly; the 23-feature space is already small enough that
aggressive compression has little to gain).

---

## PROBLEM 4

**Best SSL method:** pseudo-labeling/self-training (beat the optional
second method, Label Spreading, at every fraction — Label Spreading
scored only 0.585-0.631 balanced accuracy vs. pseudo-labeling's
0.744-0.767).

**10%:** supervised 0.6996 / SSL 0.6948 macro F1 (gain: -0.0048).
**30%:** supervised 0.7045 / SSL 0.7039 macro F1 (gain: -0.0006).
**50%:** supervised 0.7147 / SSL 0.7138 macro F1 (gain: -0.0010).

**Best SSL gain:** none are positive — the least-negative gain is at
30% (-0.0006), essentially a tie.

**Did SSL beat supervised at 10%? NO.**

**Why/why not:** an offline diagnostic (hidden labels used only for
this post-hoc check, never during training) showed pseudo-labels were
**100% accurate** but concentrated almost entirely on the already-easy
"Clear" class — they reinforced what the model already knew rather
than helping with the genuinely hard Partly Cloudy/Overcast boundary.
Not a methodology failure; a genuine, evidence-backed finding about
when pseudo-labeling does and doesn't help. See `PROBLEM4_REPORT.md`,
Section 16.

---

## PROBLEM 5

**Best transfer method:** MLP fine-tuning (Davis-pretrained, full
fine-tuning, raw kW, lr=1e-3 matching pretraining).

**Target sample count:** k=10 (largest relative gain) and k=100
(lowest absolute error) are both worth reporting.

**RMSE:** 30.03 kW (k=10) down to 22.80 kW (k=100).
**MAE:** 21.38 kW (k=10) down to 15.87 kW (k=100).
**nRMSE:** 0.2361 (k=10) down to 0.1793 (k=100).

**Transfer gain:** +8.18 kW (+21.4%) at k=10, shrinking to +0.22 kW
(+0.9%) at k=100 — gain is largest exactly where labels are scarcest,
the expected and desired transfer-learning shape.

**Why this is the selected result:** transfer beat few-shot at every
k tested, with the clearest, most practically meaningful margin at
k=10 (the realistic "brand new site" scenario). A real negative-
transfer episode occurred during development (an overly-conservative
fine-tuning learning rate) — found, diagnosed, and fixed before this
final result was reported; see `PROBLEM5_REPORT.md`, Section 12.
