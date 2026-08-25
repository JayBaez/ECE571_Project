# Experiment Plan — Problems 1–5

**Status: proposed plan only. No experiments have been executed. No code
has been written yet.** This is a high-level plan for a future stage,
built from the real project spec (§3, §4) and the dataset facts in
`DATASET_PROFILE.md`. Order/scope may change once you approve moving to
Phase 1/2.

Shared setup for all problems: fixed random seed(s), reported per
`TEACHER_EXPECTATIONS.md`; scalers fit on TRAIN only; chronological
splits wherever time series is involved; `Cloud Type` always categorical.

---

## Problem 1 — Supervised Classification

- **Objective:** predict either the 3-class sky-condition label or the
  3-class generation-regime label from weather/irradiance features.
- **Dataset:** any city sheet(s); likely start with one city (e.g.
  Davis, richest history) before generalizing.
- **Targets:**
  - Sky-condition: `k = GHI / Clearsky GHI` → Clear (≥0.85) / Partly
    cloudy (0.4–0.85) / Overcast (<0.4).
  - Generation-regime: `Output Power` binned into per-city terciles
    (Low/Medium/High).
- **Preprocessing:** exclude `GHI`, `Clearsky GHI` from features when
  predicting sky-condition (label leakage); consider also excluding or
  separately ablating `DHI`/`DNI`/`Solar Zenith Angle` together, since
  they can approximately reconstruct `GHI`. One-hot/embed `Cloud Type`.
  Standardize continuous features (fit on train only).
- **Baseline:** majority-class / simple Logistic Regression.
- **Classical models:** Decision Tree, Random Forest, Naive Bayes, kNN,
  Linear SVM (all taught, Weeks 03–07).
- **Deep model:** MLP (Week13).
- **Evaluation metrics:** Balanced accuracy (headline metric per spec),
  confusion matrix, per-class precision/recall/F1.
- **Required plots:** confusion matrix (at least for the best model per
  target).
- **Expected outputs:** a results table (model × target × metric) and a
  short discussion of which classes are most confused and a hypothesis
  why (e.g., partly-cloudy vs. overcast boundary cases).
- **Major risks:** GHI/Clearsky-GHI leakage into the sky-condition
  target (explicit spec warning); class imbalance (clear-sky dominates)
  making raw accuracy misleading — use balanced accuracy as instructed;
  generation-regime terciles must be computed **per city**, not globally,
  or cross-city imbalance will distort the labels.
- **Recommended order:** sky-condition first (labels are physically
  well-defined and the leakage rule is explicit and easy to test), then
  generation-regime.

## Problem 2 — Supervised Regression (spec's "core task")

- **Objective:** predict continuous `Output Power` (kW) from the other
  21 features.
- **Dataset / experiments (three required):**
  1. **Same-city:** chronological 80/20 split; try Davis and Amherst.
  2. **Cross-city (zero-shot):** train on Davis (6-year sheet), test on
     Huron / Santa Barbara / La Jolla, each normalized per its own city
     scale.
  3. **Sequence variant:** sliding window of last K=12 steps → predict
     next step's `Output Power`.
- **Preprocessing:** Clear-Sky Index `k`, sin/cos encodings of Hour and
  Day-of-Year, lag features of `Output Power` (t-1, t-2) for the
  sequence variant only (lag features must never be constructed across
  a train/test boundary — see leakage note below). Standardize
  continuous features (train-only fit).
- **Baseline:** persistence (predict `Output Power[t] ≈ Output Power[t-1]`)
  for the sequence variant; simple Linear Regression for same-city/
  cross-city.
- **Classical models:** Random Forest Regression, kNN Regression (Weeks
  04, 06b).
- **Deep models:** MLP (same-city/cross-city); LSTM/GRU (sequence
  variant, taught, Week15); optionally TCN or a small Transformer for
  the sequence variant (spec-suggested, not taught — good "breadth"
  addition if time allows).
- **Evaluation metrics:** RMSE, MAE, and **nRMSE** (required alongside
  RMSE for cross-city comparability); mean ± std over ≥3 seeds.
- **Required plots:** prediction-vs-truth scatter or learning curve per
  experiment.
- **Optional extension (spec §3.1):** 3-year vs. 6-year ablation —
  compare a model trained on a city's `'14-'16` sheet vs. its `'11-'16`
  sheet.
- **Major risks:** random-shuffling a time series before splitting
  (must be chronological); fitting the scaler on the full dataset
  instead of train-only; building lag/window features that reach across
  the train/test boundary (a window ending just before the split point
  but including test-side history, or vice versa, is a subtle leak);
  comparing raw RMSE across cities with very different Output Power
  scales without nRMSE (Davis's scale is ~3–3.5x Huron/Santa
  Barbara/La Jolla's — see `DATASET_PROFILE.md`); the 2012-03-22
  zero-output anomaly (all four CA/inland cities) silently corrupting a
  same-city or cross-city training set that includes 2012.
- **Recommended order:** same-city first (simplest, validates the
  pipeline) → cross-city → sequence variant last (most complex,
  reuses lessons from the first two).

## Problem 3 — Dimension Reduction

- **Objective:** reduce the feature space to d = 2/5/10 dimensions,
  unsupervised, then measure whether the reduced representation helps
  or hurts the Problem-1 classifier and Problem-2 regressor.
- **Dataset:** same feature set as Problems 1–2 (excluding target
  columns and label-leaking columns as appropriate per target).
- **Preprocessing:** dimensionality reduction must not see labels
  (`Output Power`, sky-condition, or generation-regime) during fitting —
  it's unsupervised by definition; standardize inputs before PCA/KPCA
  (both are scale-sensitive).
- **Baseline:** raw (unreduced) features fed to the same P1 classifier /
  P2 regressor, for direct comparison.
- **Classical/taught models:** PCA, Kernel PCA (Week10), explained
  variance ratio, elbow-based `d` selection.
- **Deep/extension models:** Autoencoder (built from the taught MLP
  architecture, Week13, with a reconstruction loss) as the "useful but
  not taught" option; VAE only if time allows (adds real complexity for
  the return, per your stated preference for understandable code).
- **Visualization (optional but encouraged):** 2-D t-SNE or UMAP,
  colored by sky-condition or Output-Power bin (not itself a graded
  reduction method, purely illustrative).
- **Evaluation metrics:** reconstruction MSE (AE/VAE), explained
  variance ratio (PCA/KPCA) — intrinsic; balanced accuracy (P1) and
  RMSE/MAE (P2) on reduced vs. raw features — downstream, the real grade
  driver per spec.
- **Required plots:** explained-variance / reconstruction-error-vs-d
  curve to justify the chosen `d`.
- **Major risks:** accidentally leaking labels into the reduction step
  (e.g., PCA fit on a dataframe that still includes the target column);
  fitting PCA/AE on the combined train+test set instead of train-only,
  then transforming test data with it (a classic scaler-style leak);
  forgetting to standardize before PCA, which lets high-magnitude raw
  columns (e.g. `GHI`, `Pressure`) dominate the components regardless of
  actual importance.
- **Recommended order:** PCA → Kernel PCA (both taught, cheap to run) →
  downstream evaluation on both P1 and P2 → autoencoder if time allows →
  t-SNE/UMAP visualization last (it's optional and purely illustrative).

## Problem 4 — Semi-Supervised Learning

- **Objective:** show that using unlabeled data alongside a small
  labeled fraction beats a supervised-only model trained on that same
  small fraction.
- **Dataset:** one base task, either P1 classification or P2 regression
  (spec allows either — classification is likely simpler to start with
  given more taught SSL methods target classification).
- **Preprocessing:** randomly withhold labels so only p ∈ {10%, 30%,
  50%} of the *training* split is labeled; the test set stays fully
  labeled and untouched. The unlabeled/labeled split itself must be
  done only within the training portion, after the chronological
  train/test split, not before (otherwise "unlabeled" test-time
  information could leak back into the labeled training summary
  statistics).
- **Baseline:** supervised-only model trained on just the p% labeled
  subset (no use of the unlabeled majority) — the exact comparison point
  the spec requires at each p.
- **SSL method — pending your decision (see `TEACHER_EXPECTATIONS.md`):**
  Transductive SVM, Co-training, or graph-based label propagation (all
  taught, Week12) vs. pseudo-labeling/self-training (simpler, not
  taught). Co-training needs two reasonably independent feature "views"
  — e.g., irradiance-based features (GHI-family, Clearsky-family) vs.
  meteorological features (temperature, humidity, wind, pressure) is a
  natural split for this dataset if Co-training is chosen.
- **Evaluation metrics:** the spec suggests Macro-F1 (classification) or
  nRMSE (regression) as the label-efficiency curve's y-axis; AUC of
  that curve (metric vs. %labels) as the single-number summary — **not**
  classifier ROC-AUC.
- **Required plot:** label-efficiency curve (metric vs. p ∈
  {10,30,50}%), SSL vs. supervised-only.
- **Required check:** does the SSL method beat the 10%-only supervised
  baseline specifically at p=10%? (Spec calls this out explicitly —
  it's the strongest test of whether unlabeled data actually helps.)
- **Major risks:** constructing the labeled/unlabeled split before the
  chronological train/test split (order matters — split first, then
  subsample labels only within train); accidentally using true labels
  of "unlabeled" points anywhere in the SSL method itself; comparing
  SSL and supervised-only baselines on different random subsets instead
  of the same p% draw (adds noise that looks like a false SSL gain or
  loss).
- **Recommended order:** build and validate the supervised-only baseline
  at all three p values first (this alone answers "how much do labels
  matter"), then add one SSL method and compare.

## Problem 5 — Transfer Learning

- **Objective:** show a transfer method that beats both a zero-shot and
  a few-shot baseline on a data-scarce target city.
- **Dataset:** SOURCE = Davis (6-year sheet). TARGET = Amherst (only
  3-year city available, no other choice given the dataset — see
  `DATASET_PROFILE.md`'s note that Amherst shares no calendar years with
  any other city).
- **Preprocessing:** normalize/scale `Output Power` per city before any
  cross-city comparison (Davis's plant capacity is roughly 3–3.5x
  Amherst's peak, per `DATASET_PROFILE.md`), consistent with Problem 2's
  cross-city handling.
- **Baseline 1 — Zero-shot:** train on Davis only, evaluate directly on
  Amherst, no fine-tuning.
- **Baseline 2 — Few-shot:** train only on a small Amherst-labeled
  subset, k ∈ {10, 50, 100} samples.
- **Transfer method:** likely candidate given taught material — train
  an MLP (or the best P2 regressor) on Davis, then fine-tune its later
  layers on the same small Amherst subset used for the few-shot
  baseline (general practice; not covered in the course, see
  `ML_METHOD_MAP.md`).
- **Evaluation metrics:** target-domain RMSE (and/or accuracy if the
  chosen task is classification-flavored); transfer gain = metric on
  target using the transfer method minus metric of a target-only
  baseline.
- **Required output:** one table with zero-shot, few-shot, and transfer
  method side by side, for each k.
- **Required discussion:** what transferred and what didn't; explicitly
  report negative transfer if the source-trained model actually did
  worse than a target-only model — don't hide a negative result.
- **Major risks:** any Amherst data leaking into "Davis-only" source
  training (must be zero-shot means truly zero Amherst rows seen);
  reusing the *same* few-shot Amherst sample for both the few-shot
  baseline and the transfer method's fine-tuning step is fine and
  arguably required for a fair comparison — but using two *different*
  samples would confound the comparison; remembering the year gap
  between Davis (2011–2016) and Amherst (2018–2020) means this transfer
  problem also crosses time, not just geography — worth naming in the
  discussion.
- **Recommended order:** zero-shot and few-shot baselines first (cheap,
  establish the numbers to beat) → transfer method last.

## Cross-problem recommended sequencing

Given the grading weights (Best metric 30 pts, Breadth 20 pts), and that
Problem 2 is explicitly the "core task": **P1 → P2 → P3 → P4 → P5** is a
reasonable default order — P1 is the simplest full pipeline to get
working end-to-end (validates data loading, splitting, scaling, a
classical model, and a deep model all at once), P2 is the core task and
reuses everything from P1, P3 depends on P1+P2 already existing (it
evaluates reduced features on both), and P4/P5 are the most novel /
highest-risk paradigms, best tackled once the shared framework is solid.
