# Problem 4 Report — Semi-Supervised Learning

Internal technical record for Problem 4. Every number comes from an
actual executed run of `problems/problem4_semi_supervised/
run_experiments.py` — nothing is estimated. Full results: `results/
problem4/problem4_results.csv` (27 rows), `problem4_label_efficiency.csv`,
`problem4_pseudolabel_summary.csv`. Reproducibility: seeds 42/123/2026,
Python 3.12.3, pandas 3.0.2, scikit-learn 1.8.0, run 2026-09-08.

---

## 1. Objective

Answer, with evidence: how much can unlabeled data improve sky-condition
classification when only 10%, 30%, or 50% of training labels are
available?

## 2. Why semi-supervised learning is useful for PV data

Labeled weather/sky-condition data requires either manual annotation or
a co-located sensor that isn't always available; irradiance and weather
*readings* are comparatively cheap and plentiful. SSL asks whether the
abundant unlabeled readings can substitute, in part, for scarce labels.

## 3. Classification task

Reused directly from Problem 1: 3-class sky-condition (Clear/Partly
Cloudy/Overcast) from `Clear-Sky Index = GHI / Clearsky GHI`, same
thresholds. `GHI`/`Clearsky GHI` excluded from features (define the
label); `DHI`/`DNI`/`Solar Zenith Angle` also excluded, carrying
forward Problem 1's resolved leakage decision even though Phase 7's
instructions only explicitly name GHI/Clearsky GHI — kept for full
methodological consistency (`course_context/LEAKAGE_MAP.md`).

## 4. Train/test split

**Reused Problem 1's exact Davis chronological split** via direct
function reuse (`problem1_classification.run_experiments.
build_task_dataset()`), not a re-implementation — verified identical:
19,289 train / 4,823 test. The test set was never touched by any
labeled/unlabeled sampling, training, or threshold selection.

## 5. Labeled/unlabeled setup

Within the 19,289-row training pool: stratified sampling selects the
labeled subset (guarantees every class survives even at 10%); the
remainder becomes the unlabeled pool (X visible, y hidden from every
training/selection step). Verified at 10% (seed 42): Overcast=151,
Partly Cloudy=308, Clear=1,469 — no class missing.

## 6. Label fractions

10%, 30%, 50% — exact split indices are reproducible via
`train_test_split(..., random_state=seed, stratify=y_train)`, not
separately saved to disk as raw index lists, but fully reconstructible
from the seed.

## 7. Supervised baseline

Logistic Regression, `class_weight="balanced"` — **Problem 1's actual
best Davis sky-condition model** (verified: `results/problem1/
problem1_results.csv`, balanced_accuracy=0.772, beating every Random
Forest/Gradient Boosting/MLP variant tried there). Chosen by evidence,
not assumption, per Section 13's explicit instruction.

## 8. SSL method

**Primary: pseudo-labeling/self-training** (`pseudo_labeling.
self_train()`) — iteratively trains on the labeled set, predicts on
the unlabeled pool, accepts predictions above a confidence threshold as
pseudo-labels, retrains, repeats (max 5 iterations, or until no
predictions clear the threshold).

**Optional second method: Label Spreading** (`sklearn.semi_supervised.
LabelSpreading`, KNN kernel, k=7) — the course-taught, graph-based SSL
family (`COURSE_CONTEXT.md`, Week12), already selected as this
project's "breadth" SSL choice back in Phase 3
(`TEACHER_EXPECTATIONS.md`).

## 9. Pseudo-labeling procedure

Exactly the 6-step loop in Section 9 of the Phase 7 instructions,
implemented directly: train → predict unlabeled → threshold → add →
retrain → repeat. True hidden labels are never passed into
`self_train()` — verified by inspection of every call site, and
confirmed empirically via the offline diagnostic (Section 16).

## 10. Confidence threshold

**0.90**, chosen via `select_confidence_threshold()` — an inner
train/validation split of the *labeled* subset only (never the test
set), scored on validation balanced accuracy:

| Threshold | Validation balanced accuracy |
|---|---|
| 0.80 | 0.765 |
| **0.90** | **0.772** |
| 0.95 | 0.755 |

## 11. Class imbalance handling

Stratified sampling (Section 5) prevents class disappearance at
sampling time. **A second, more important safeguard was needed and
found empirically**: uncapped self-training's first iteration alone
added 12,170 pseudo-labels, **98% of them "Clear"** — a real
class-collapse pattern, not a hypothetical one. Tested directly (10%
labels, seed 42):

| Per-iteration cap | Pseudo-labels added | Balanced accuracy |
|---|---|---|
| None | 15,431 | 0.7438 |
| 500 | 2,500 | 0.7506 |
| **1,000** | 5,000 | **0.7511** |

**`max_pseudo_labels_per_iteration=1000` was adopted** based on this
comparison — a real, evidence-based safeguard, not a default assumed
in advance.

## 12. Evaluation metrics

Balanced accuracy, accuracy, macro-precision, macro-recall, macro-F1 —
all via `src/evaluation.py` (Phase 2, reused unchanged). Balanced
accuracy and macro-F1 treated as primary, per instructions.

## 13. Label-efficiency results

Mean ± std across 3 seeds:

| Fraction | Supervised Bal. Acc. | SSL Bal. Acc. | Supervised Macro F1 | SSL Macro F1 |
|---|---|---|---|---|
| 10% | 0.7476 ± 0.0134 | 0.7442 ± 0.0177 | 0.6996 ± 0.0066 | 0.6948 ± 0.0034 |
| 30% | 0.7597 ± 0.0102 | 0.7590 ± 0.0097 | 0.7045 ± 0.0151 | 0.7039 ± 0.0148 |
| 50% | 0.7675 ± 0.0052 | 0.7671 ± 0.0048 | 0.7147 ± 0.0072 | 0.7138 ± 0.0063 |

Figure: `figures/problem4/problem4_label_efficiency_curve.png`.

## 14. SSL gain

| Fraction | Gain (Balanced Acc.) | Gain (Macro F1) |
|---|---|---|
| 10% | -0.0034 | -0.0048 |
| 30% | -0.0007 | -0.0006 |
| 50% | -0.0005 | -0.0010 |

**Negative at every fraction — reported honestly, not adjusted.**
Figure: `figures/problem4/problem4_ssl_gain.png`.

## 15. AUC

Trapezoidal, x ∈ [0.10, 0.50] (not extrapolated to the full 0-1 range —
only 3 points were measured):

| Metric | Supervised AUC | SSL AUC | Better |
|---|---|---|---|
| Macro F1 | 0.2823 | 0.2816 | Supervised |
| Balanced Accuracy | 0.3035 | 0.3029 | Supervised |

## 16. Pseudo-label analysis

5,000 pseudo-labels added at every fraction (the 1,000-per-iteration
cap × 5 iterations, consistently maxed out). **OFFLINE DIAGNOSTIC
ONLY** (hidden true labels used solely for this post-hoc check, never
during training or threshold selection): pseudo-label accuracy was
**exactly 1.0 (100%) at every fraction and seed tested.** This is not
a bug — verified directly (predicted vs. true values inspected
row-by-row, zero mismatches out of 5,000). **This is the key to
understanding the near-zero gain**: the confidence threshold + cap
selects only the most obviously-correct unlabeled rows (overwhelmingly
clear-sky days), which the model already classifies correctly using
just the labeled data — the pseudo-labels are accurate but
**redundant**, not wrong. Figures: `figures/problem4/
problem4_pseudolabel_confidence.png`, `problem4_pseudolabel_class_distribution.png`.

## 17. Confusion matrices

`figures/problem4/problem4_confusion_matrix_{10,30}pct_{supervised,ssl}.png`
— supervised and SSL confusion matrices look nearly identical at both
fractions, consistent with the near-zero gain (SSL isn't making
qualitatively different errors, just marginally different ones).

## 18. 10% label result

**Did SSL beat supervised-only at 10% labels? NO.**
Supervised: balanced accuracy 0.7476, macro F1 0.6996.
SSL: balanced accuracy 0.7442, macro F1 0.6948.
Gain: -0.0034 (balanced accuracy), -0.0048 (macro F1).

**Why:** the pseudo-labels added were essentially all correct (Section
16) but concentrated on the class the model already handles well
("Clear"), so they added redundant confirmation rather than new
discriminating information about the harder Partly Cloudy/Overcast
boundary — the actual bottleneck this task has (confirmed in
`PROBLEM1_REPORT.md`: Partly Cloudy and Overcast are consistently the
hardest classes). The methodology was not altered to force a different
outcome.

## 19. Failure modes

Pseudo-labeling did not "fail" in the sense of adding wrong labels
(100% offline accuracy) — its failure mode here is **redundancy, not
error**: it reinforces the easy majority class without helping the
genuinely hard classes. **Label Spreading fared worse in absolute
terms** — balanced accuracy 0.585-0.631 across all fractions, well
below both the supervised baseline and pseudo-labeling — plausibly due
to its KNN graph construction struggling with a large unlabeled pool
(17K+ rows) in a 23-dimensional mixed continuous/categorical feature
space.

## 20. Limitations

- Only Davis was tested (consistent with the primary-task scope
  instruction to prioritize the classification comparison itself).
- The 5-iteration cap and 1,000-per-iteration cap were chosen from a
  small, documented comparison (Section 11), not an exhaustive search.
- Label Spreading used one fixed configuration (KNN, k=7) — not tuned,
  consistent with "choose only ONE additional method" and the
  instruction not to build an elaborate SSL search.
- Regression SSL (Section 27, optional) was not attempted — classification
  was sufficient and the primary required task.

## 21. Reproducibility

Seeds: 42, 123, 2026. Python 3.12.3; pandas 3.0.2; numpy 2.4.4;
scikit-learn 1.8.0. Dataset: `course/Further Consolidated Data,
HnL.xlsx`. Full run history: `results/problem4/problem4_results.csv`
(27 rows). Models: `results/problem4/models/` (6 `.joblib` files, all
verified loadable, plus `model_config.json`). One reproducibility
detail addressed proactively: `LogisticRegression` was given an
explicit `random_state` for consistency with every other model factory
in this project, even though empirically verified deterministic
without it (confirmed via 3 repeated fits producing identical results
to 10 decimal places). Note on execution: like Problems 1-3, this
phase's experiments were run as several separate script invocations to
stay within this sandbox's per-command execution-time limit; a
duplicate-row artifact from re-running confusion-matrix generation was
caught and cleaned via exact-duplicate removal before final analysis.
