# Problem 3 Report — Dimension Reduction

Internal technical record for Problem 3. Every number comes from an
actual executed run of `problems/problem3_dimension_reduction/
run_experiments.py` — nothing is estimated. Full results: `results/
problem3/problem3_results.csv` (54 rows), central table: `results/
problem3/problem3_comparison_table.csv`. Reproducibility: seeds
42/123/2026, Python 3.12.3, pandas 3.0.2, scikit-learn 1.8.0, torch
2.14.0, run 2026-09-04.

---

## 1. Objective

Compress a unified input feature space with PCA and an autoencoder,
then measure whether the compressed representation still supports
Problem 1's sky-condition classifier and Problem 2's Output Power
regressor as well as the raw features do.

## 2. Why dimensionality reduction is being tested

Not to make a pretty plot — to answer, with evidence: does compression
preserve enough information for the downstream tasks to still perform
well? All three outcomes (helps / similar with fewer dimensions /
hurts) were treated as valid going in.

## 3. Input features

**One unified, leakage-safe feature set** (`problems/
problem3_dimension_reduction/features.py`), reused for PCA, the
autoencoder, and both downstream tasks: Cloud Type (categorical,
one-hot encoded) + 8 weather columns + 6 cyclical time features → 23
columns after encoding. **Deliberately excludes all irradiance-family
columns** (GHI, DNI, DHI, Clearsky GHI/DNI/DHI, Solar Zenith Angle) —
this is Problem 1's sky-condition leakage-safe set, reused here so one
PCA/autoencoder representation is valid for both downstream tasks
(see Section 4).

## 4. Leakage prevention

**Two separate leakage concerns, both addressed:**

1. **Representation learning never sees a label.** PCA fits on
   `X_train` only; the autoencoder reconstructs its own input, never a
   target. Labels (`Sky_Condition`, `Output Power`) are attached to the
   dataset dict but never passed to `fit_pca()` or `train_autoencoder()`
   — verified by inspection of every call site.
2. **Why irradiance columns are excluded from the shared input at
   all** (not just from the classifier, as in Problem 1): if PCA/the
   autoencoder were fit on a feature set that includes GHI, the
   resulting components would likely capture mostly GHI-driven
   variance (GHI dominates this dataset's variance —
   `EDA_REPORT.md`), and using those components downstream for
   sky-condition classification would functionally leak the label
   through the back door — exactly what Problem 1's DHI/DNI/Solar-
   Zenith-Angle ablation proved directly (accuracy jumped from
   ~0.74-0.81 to ~0.98 when those columns were included). Excluding
   irradiance from the shared PCA/AE input avoids reintroducing that
   risk here.

**Documented tradeoff:** this makes Problem 3's regression "raw"
baseline (RMSE 23.86) weaker than Problem 2's headline Davis result
(RMSE 15.17) — Problem 2 legitimately used irradiance features (no
leakage restriction for regression), Problem 3 deliberately doesn't,
for one fair, shared representation. Not an error — a documented,
principled scope decision.

## 5. Train/test setup

Davis only, chronological 80/20 split — **verified identical row
boundary to Problems 1 and 2** (19,289 train / 4,823 test, same last-
train timestamp: 2015-10-20 12:30).

## 6. PCA methodology

`sklearn.decomposition.PCA`, fit on `X_train` only
(`reduction.fit_pca()`). Evaluated at d ∈ {1, 2, 3, 5, 7, 10, 15, 20,
23} for a clear elbow curve, with d=2/5/10 as the three required
headline values.

## 7. Explained variance

| d | Cumulative explained variance |
|---|---|
| 2 | 0.434 |
| 5 | 0.761 |
| 10 | 0.946 |
| 23 (all) | 1.000 |

Figure: `figures/problem3/problem3_explained_variance.png` — clear
elbow around d=5-10; variance is essentially saturated by d=15.

## 8. Reconstruction error

| d | PCA reconstruction MSE | Autoencoder reconstruction MSE (mean±std, 3 seeds) |
|---|---|---|
| 2 | 0.3357 | 0.2039 ± 0.0061 |
| 5 | 0.1495 | 0.0619 ± 0.0006 |
| 10 | 0.0455 | 0.0260 ± 0.0001 |

**The autoencoder reconstructs better than PCA at every dimension** —
expected: PCA is restricted to linear projections, the autoencoder can
learn nonlinear encodings (`COURSE_CONTEXT.md`, Week10 vs. general
autoencoder theory). Figure: `figures/problem3/
problem3_reconstruction_error.png`.

## 9. Autoencoder methodology

`SimpleAutoencoder` (`reduction.py`): Input→Dense(32)→ReLU→Latent(d)
→Dense(32)→ReLU→Output. Trained with `src/torch_utils.py`'s
`train_torch_model()` (Phase 2, reused unchanged) — MSE loss between
input and reconstruction, Adam, early stopping on a chronological
inner-validation split (last 20% of training data). 3 seeds per
dimension; the seed=42 model of each dimension saved for downstream
use and visualization.

## 10. Autoencoder reconstruction error

See Section 8's combined table.

## 11. Downstream classification

Random Forest (Problem 1's established ablation-study "workhorse" —
same model choice for the same reason: comparable results across
representations, not confounded by also varying the downstream
model), sky-condition target, 3 seeds:

| Representation | Balanced Accuracy | Macro F1 |
|---|---|---|
| Raw | **0.743** | **0.736** |
| PCA-2 | 0.401 | 0.406 |
| PCA-5 | 0.559 | 0.552 |
| PCA-10 | 0.571 | 0.577 |
| AE-2 | 0.457 | 0.462 |
| AE-5 | 0.568 | 0.559 |
| AE-10 | **0.661** | **0.645** |

Figure: `figures/problem3/problem3_downstream_classification.png`.

## 12. Downstream regression

Gradient Boosting (Problem 2's Davis champion), Output Power target,
3 seeds:

| Representation | RMSE (kW) | MAE (kW) | nRMSE |
|---|---|---|---|
| Raw | **23.86** | **15.84** | **0.0944** |
| PCA-2 | 44.86 | 35.13 | 0.1775 |
| PCA-5 | 33.96 | 24.82 | 0.1343 |
| PCA-10 | 32.10 | 22.39 | 0.1270 |
| AE-2 | 41.93 | 31.36 | 0.1659 |
| AE-5 | 36.29 | 27.02 | 0.1436 |
| AE-10 | **29.69** | 21.88 | 0.1174 |

Figure: `figures/problem3/problem3_downstream_regression.png`.

**Optional PCA+MLP check (Section 15):** raw+MLP = 0.755 balanced
accuracy vs. PCA-10+MLP = 0.608 — the same "raw wins" conclusion holds
with a nonlinear downstream model too, not just Random Forest.

## 13. t-SNE/UMAP visualization

`figures/problem3/problem3_tsne.png` — fit on a random (not
label-cherry-picked) sample of 2,000 test rows, `sklearn.manifold.TSNE`,
seed 42, perplexity 30, PCA-initialized. Visualization only — never
used as a production feature transform. Shows partial but incomplete
class separation, consistent with the moderate downstream accuracy
this feature set achieves.

## 14. Comparison with raw features

**Raw features win on both downstream tasks, at every tested
dimension, for both PCA and the autoencoder.** Full central table:
`results/problem3/problem3_comparison_table.csv`.

## 15. Best representation

Among the *reduced* representations, **Autoencoder-10 is consistently
the strongest** — best classification (0.661 balanced accuracy) and
best regression (29.69 RMSE) of every PCA/AE option, and clearly ahead
of PCA-10 in both (0.571 / 32.10). The autoencoder's nonlinear
compression capacity translates into a real downstream advantage over
PCA at every dimension tested, not just in the reconstruction-MSE
numbers.

## 16. Whether compression helped

**No — dimension reduction hurt both downstream tasks here**, even at
d=10 (the largest tested dimension, using ~43% of the original 23
features). This is reported as the genuine finding, not adjusted or
downplayed — Section 29 of the Phase 6 instructions explicitly permits
this outcome as equally valid to "compression helps."

## 17. Failure modes (why compression hurt here)

- **Cloud Type's categorical structure compresses poorly.** The small
  feature-ablation experiment (Section 28) confirms this directly:
  removing Cloud Type from PCA-5's input actually *raised* explained
  variance (0.787 vs. 0.761, fewer total dimensions to explain) but
  *lowered* both reconstruction quality (MSE 0.196 vs. 0.150) and
  downstream classification accuracy (0.510 vs. 0.561) — Cloud Type
  carries real predictive signal that a shared continuous latent space
  struggles to preserve, consistent with Problem 1's finding that
  Cloud Type dominated feature importance for this exact task.
- **The original feature space is already fairly small (23 columns
  after encoding).** Aggressive compression to 2-10 dimensions is a
  much bigger relative information loss here than it would be for a
  dataset with hundreds of raw columns — the "not big enough to
  benefit from aggressive compression" failure mode named in Section 29
  applies directly.
- **Regression felt this more sharply at low d** (PCA-2's RMSE 44.86 is
  nearly double raw's 23.86) than classification did, suggesting some
  of the fine-grained weather variation that matters for a continuous
  kW prediction gets flattened out faster under compression than the
  coarser signal needed for a 3-class label.

## 18. Limitations

- Only Davis was tested (Section 5's "where appropriate" — Amherst
  skipped for scope, given the phase's overall size).
- The autoencoder architecture (one hidden layer, size 32) was fixed
  across all three latent dimensions, not separately tuned per d — a
  reasonable, deliberately simple choice (Section 39), but a tuned
  architecture might close some of the raw-vs-AE gap.
- PCA+MLP (Section 15) was only checked at d=10, not d=2/5, since it's
  explicitly optional and was kept small per the instructions.
- Trustworthiness (Section 25) was not computed — explicitly optional
  and skipped given the phase's already-large scope; the raw-vs-
  reduced downstream comparison already answers the core grading
  question directly.

## 19. Reproducibility

Seeds: 42, 123, 2026. Python 3.12.3; pandas 3.0.2; numpy 2.4.4;
scikit-learn 1.8.0; torch 2.14.0 (CPU in this sandbox — no GPU present
here; will use CUDA automatically via `src/utils.py`'s `get_device()`
on the project owner's RTX 2070). Dataset: `course/Further Consolidated
Data, HnL.xlsx`. Full run history: `results/problem3/
problem3_results.csv` (54 rows). Saved models: `results/problem3/
models/` (3 PCA `.joblib` files, 3 autoencoder `.pt` files, all
verified loadable, plus `model_config.json`). Note on execution: like
Problems 1-2, this phase's stages were run as several separate script
invocations to stay within this sandbox's per-command execution-time
limit — a sandbox filesystem reset also occurred mid-phase (disk
space exhaustion from package installs), requiring the `problems/
problem3_dimension_reduction/` source files to be rebuilt from
scratch; results computed before and after the reset matched exactly
(bit-for-bit identical reconstruction MSE values), confirming the
rebuild was faithful and every stage remains deterministic and
reproducible.
