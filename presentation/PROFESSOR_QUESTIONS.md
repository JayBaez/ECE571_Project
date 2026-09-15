# Professor Question Preparation

25 likely questions, organized by category. Every answer traces to
`report/FINAL_REPORT.md` or a `course_context/PROBLEMN_REPORT.md`.

---

## General

**Q1. Why did you structure this as five separate problems instead of one model?**
- **Short answer:** Each ML paradigm needed its own fair baseline and evaluation protocol — folding everything into one pipeline would make it impossible to say which technique contributed what.
- **Detailed answer:** Classification, regression, dimension reduction, semi-supervised learning, and transfer learning each ask a genuinely different question of the data. Testing them separately, with the same shared preprocessing framework but paradigm-appropriate baselines (majority-class, raw-feature, supervised-only, few-shot), lets me isolate what each technique actually adds rather than getting one tangled result.

**Q2. What was the single most important result in the whole project?**
- **Short answer:** Same-city Davis regression: RMSE = 15.17 kW, R² = 0.953 — the strongest, most direct evidence that weather predicts power well.
- **Detailed answer:** It's the project's core task (Problem 2) and the result every other comparison implicitly measures against. Every other paradigm's "does this help?" question is ultimately asking whether it can approach, preserve, or improve on this number under harder constraints (less data, compressed features, a different city).

**Q3. Did you use AI assistance on this project? How much?**
- **Short answer:** Yes, throughout — code generation, debugging, and report writing, all reviewed and executed by me at each step.
- **Detailed answer:** See `report/FINAL_REPORT.md`, Section 15. AI tools helped scaffold code and generate this report and presentation, but every result was actually run and verified, every methodological decision (target definitions, leakage rules, which experiments to run) was mine, and several real bugs (documented, not hidden) were caught through review and execution.

**Q4. Why five paradigms specifically, and not others (e.g., reinforcement learning, ensembling)?**
- **Short answer:** These five map directly to the course's actual syllabus and grading rubric.
- **Detailed answer:** `course_context/TEACHER_EXPECTATIONS.md` and the assignment specification define exactly these five problems. Staying within course-taught methods was a deliberate choice, not a limitation — the goal was demonstrating mastery of what was taught, not novelty for its own sake.

---

## Dataset

**Q5. Why didn't you randomly shuffle the data before splitting into train/test?**
- **Short answer:** This is time-series data — shuffling first would let a model trained on future readings predict past ones.
- **Detailed answer:** A random shuffle would place, say, a Tuesday-afternoon reading in training and the preceding Tuesday-morning reading in test. The model could then implicitly "see the future" relative to what it's being tested on — every reported metric would be optimistic and meaningless for real forecasting. Every same-city split in this project is chronological: earliest 80% of timestamps train, latest 20% test.

**Q6. Why does Output Power need normalization across cities?**
- **Short answer:** Davis's plant produces ~2.7–3.5× the average output of the smaller cities — a model calibrated to one city's scale fails badly applied directly to another's.
- **Detailed answer:** This is a physical plant-capacity difference, not a data-quality issue. Raw zero-shot transfer (Problem 2's cross-city experiment, Problem 5's zero-shot baseline) failed with RMSE around 125–184 kW purely from this scale mismatch — a diagnostic rescaling (analysis only, not a real zero-shot method) recovered R² of 0.55–0.84, proving the underlying weather-power relationship transfers fine once scale is handled.

**Q7. How much missing data did you have, and how did you handle it?**
- **Short answer:** Only 4 missing rows in the entire dataset (Amherst, one date) — dropped, not interpolated, wherever Output Power was the target.
- **Detailed answer:** See `course_context/DATASET_PROFILE.md`. Interpolating a missing target value and then training on it as if it were real ground truth would fabricate data; dropping those 4 rows was the more defensible choice given how few there were.

**Q8. What is the Clear-Sky Index and why does it matter?**
- **Short answer:** `k = GHI / Clearsky GHI` — measures how close actual irradiance is to the theoretical clear-sky maximum; it defines Problem 1's sky-condition label.
- **Detailed answer:** Clear ≥0.85, Partly Cloudy 0.4–0.85, Overcast <0.4 (project spec's exact thresholds, used as-is). Because it's literally computed from GHI, GHI and Clearsky GHI (and anything that can reconstruct them) must be excluded from the sky-condition classifier's features.

---

## Problem 1 — Classification

**Q9. Why can't you use GHI as a feature for sky-condition classification?**
- **Short answer:** GHI defines the label — using it as a feature would leak the answer directly.
- **Detailed answer:** Sky-condition is a fixed threshold function of GHI/Clearsky GHI. I also excluded DHI, DNI, and Solar Zenith Angle, because together they can approximately reconstruct GHI — tested directly, including them pushed accuracy from ~0.74–0.81 to ~0.98, confirming the leak was real, not theoretical.

**Q10. Why did a simple model (logistic regression) beat your ensembles and neural network?**
- **Short answer:** Once leakage-risk features are excluded, the remaining predictors relate to sky-condition fairly directly, so a well-regularized linear model doesn't lose much by skipping complex interactions, while it gains from lower variance.
- **Detailed answer:** This pattern repeated across the project (Problem 2's untuned gradient boosting beating tuned versions) — simpler models were consistently competitive whenever the underlying relationship didn't need heavy nonlinearity to capture.

**Q11. Why use balanced accuracy instead of raw accuracy?**
- **Short answer:** The classes are imbalanced (Clear is 72% of the data) — a model that always predicts "Clear" would score 70%+ raw accuracy while being useless.
- **Detailed answer:** Balanced accuracy averages per-class recall, so it can't be inflated by a majority class. It was the primary metric throughout Problem 1 for exactly this reason.

---

## Problem 2 — Regression

**Q12. Why did cross-city zero-shot transfer fail so badly (RMSE ~125 kW)?**
- **Short answer:** Scale mismatch — the Davis model outputs values near Davis's ~164 kW average, applied to cities averaging 47–50 kW.
- **Detailed answer:** Confirmed via a diagnostic: rescaling Davis's raw predictions by the target city's own mean (analysis only) recovered R² of 0.55–0.84, showing the weather-to-power relationship itself transfers reasonably — it's the output calibration that fails without adaptation.

**Q13. Why does the sequence (GRU) model not beat the same-city regression model?**
- **Short answer:** They're solving different tasks — the non-sequence model sees the concurrent GHI reading at the exact moment predicted; the GRU only sees past readings.
- **Detailed answer:** The GRU (17.58 kW) does clearly beat a persistence baseline (21.97 kW), proving it learns real temporal structure. Comparing it to the non-sequence model (15.17 kW) isn't apples-to-apples, since the non-sequence model has access to information a true forecast wouldn't.

**Q14. What does nRMSE tell you that RMSE doesn't?**
- **Short answer:** RMSE normalized by the target's range — makes error comparable across targets/cities with very different scales.
- **Detailed answer:** Range-normalized (RMSE / (max − min)) consistently throughout the project, a convention fixed early and applied everywhere it's computed, documented in every report.

---

## Problem 3 — Dimension Reduction

**Q15. What does PCA actually do?**
- **Short answer:** Finds the directions the data varies most along, and re-expresses each row as coordinates along those directions instead of the original columns.
- **Detailed answer:** It's a linear, unsupervised method — it never looks at any label, only at how the input features vary among themselves. Explained variance measures how much of that total variation the kept components capture.

**Q16. Why is dimension reduction unsupervised?**
- **Short answer:** By design — it should learn the structure of the DATA, not the structure of the LABEL, so the downstream comparison (does compression preserve useful information?) is meaningful.
- **Detailed answer:** If labels were used to shape the representation, that would be a different (supervised) technique, and "did compression help" would really be answering "did cheating help." Verified in this project: neither `fit_pca()` nor the autoencoder's training function accepts a label argument at all.

**Q17. Why did dimension reduction hurt performance here?**
- **Short answer:** Traced to Cloud Type — a highly predictive categorical feature that compresses poorly into a small number of continuous dimensions.
- **Detailed answer:** A feature ablation confirmed this directly: removing Cloud Type from the PCA input raised explained variance (fewer total dimensions to explain) but lowered both reconstruction quality and downstream accuracy.

**Q18. Why did the autoencoder beat PCA?**
- **Short answer:** It can learn nonlinear (curved) compressions; PCA is restricted to linear projections.
- **Detailed answer:** Confirmed at every tested dimension (d=2, 5, 10), both in reconstruction MSE and downstream accuracy/RMSE — though neither method closed the gap with using raw, uncompressed features.

---

## Problem 4 — Semi-Supervised Learning

**Q19. How does pseudo-labeling work?**
- **Short answer:** Train on the small labeled set, predict on the unlabeled pool, keep only high-confidence predictions as "pseudo-labels," add them to training, repeat.
- **Detailed answer:** Confidence threshold 0.90, chosen by testing 0.80/0.90/0.95 on a validation split (not assumed). A cap of 1,000 pseudo-labels per round was added after testing showed uncapped self-training added 12,170 labels in round one, 98% the same class — a real class-collapse risk.

**Q20. How did you decide whether a pseudo-label was trustworthy?**
- **Short answer:** Confidence threshold at training time; an offline accuracy check afterward, never used to influence training.
- **Detailed answer:** The confidence threshold (0.90) is the actual trust mechanism used during training. Separately, after training was already finished, I checked pseudo-label accuracy against the true hidden labels — purely diagnostic, confirming they were 100% accurate at every fraction and seed, never fed back into the model.

**Q21. If the pseudo-labels were 100% accurate, why didn't SSL help?**
- **Short answer:** They were overwhelmingly for the easy "Clear" class — redundant with what the model already knew, not informative about the hard classes.
- **Detailed answer:** This is the key distinction in Problem 4's findings: a failure of redundancy, not of correctness. It's a different, subtler failure mode than "SSL added wrong information," and worth stating precisely rather than collapsing into a generic "SSL didn't work."

---

## Problem 5 — Transfer Learning

**Q22. What is negative transfer?**
- **Short answer:** When using a pretrained model actively makes performance worse than not using it at all.
- **Detailed answer:** Observed directly in this project: an early version of the transfer pipeline (smaller fine-tuning learning rate, as the assignment itself suggested) produced RMSE 140.9 kW at k=10 — worse than the 38.2 kW few-shot baseline trained from scratch. Diagnosed and fixed by matching the pretraining learning rate instead.

**Q23. Why might Davis transfer poorly to Amherst?**
- **Short answer:** Different output scale (2.7× on average) and some climate differences (Davis sunnier and warmer) — though the underlying weather-power relationship transfers reasonably once scale is handled.
- **Detailed answer:** A domain-shift analysis found moderate differences in Temperature, GHI, DNI, and Clear-Sky Index (standardized mean differences ~0.66–0.92) and one striking outlier, Wind Speed (2.61) — flagged as possibly a sensor/unit artifact rather than a confirmed climate fact.

**Q24. Why did you test both a normalization ablation and a freezing ablation?**
- **Short answer:** To find the actual cause of the original negative transfer, not assume it.
- **Detailed answer:** Both were tested independently. Normalization showed no benefit once the learning rate was fixed (23.35 kW raw vs. 24.11 kW normalized at k=50) — proving the learning rate, not the absence of explicit scale handling, was the real problem. Freezing the first layer also showed no meaningful difference (23.35 vs. 23.40 kW), suggesting the general representation transferred largely intact.

---

## Reproducibility

**Q25. How reproducible are your results?**
- **Short answer:** Very — fixed seeds throughout, every experiment saved to CSV, and 5 representative experiments (one per problem) were independently rerun during a dedicated audit phase and matched saved results exactly.
- **Detailed answer:** See `course_context/FINAL_AUDIT.md`, Section 10, and `REPRODUCIBILITY_CHECKLIST.md`. Seeds [42, 123, 2026] used identically everywhere; every scaler/encoder fit on training data only (verified via code inspection, not just assumed); no GPU was available during development so all models trained on CPU, but the code will use CUDA automatically with no changes on GPU hardware.

**Q26. What would you do with more time?**
- **Short answer:** A larger hyperparameter search, investigate the Wind Speed anomaly, and extend transfer learning to the other data-scarce cities.
- **Detailed answer:** See `report/FINAL_REPORT.md`, Section 16 (Conclusion) and Section 13 (Limitations) for the full list — hyperparameter searches were deliberately kept small throughout per the project's own scope guidance, and repeatedly found tuning barely moved results, suggesting limited (but unconfirmed) headroom left with this approach.
