# Likely Professor Questions

Every answer traces to `report/FINAL_REPORT.md` or a saved results
file — nothing here is invented. This file replaces an earlier
version built for a different presentation draft; the underlying
project facts are unchanged.

## General Project

**Q: Why did you structure this as five separate problems instead of one model?**
A: Each ML paradigm needed its own fair baseline and evaluation setup — folding everything into one pipeline would make it impossible to say which technique actually contributed what. Testing them separately, but with the same shared preprocessing and leakage rules, keeps the comparisons honest.

**Q: What is the single most important result in the whole project?**
A: Same-city Davis regression: RMSE = 15.17 kW, R² = 0.953. It's the project's core task, and every other comparison in the project is implicitly measured against it.

**Q: Did you use AI assistance? How much?**
A: Yes, throughout — for code generation, debugging, and drafting the report and presentation. Every result was actually run and verified by me, and every methodological decision (target definitions, leakage rules, which experiments to run) was mine. Full disclosure is in the report.

## Problem 1 — Classification

**Q: Why didn't you use GHI as a feature for sky-condition classification?**
A: The sky-condition label is a fixed threshold function of GHI divided by Clearsky GHI, so GHI literally defines the label. Using it as a feature would leak the answer directly. I also excluded DHI, DNI, and Solar Zenith Angle since they can approximately reconstruct GHI — I tested this directly, and including them pushed accuracy from ~0.74-0.81 to ~0.98, confirming the leak was real.

**Q: Why did a simple model (logistic regression) beat your ensembles and neural network?**
A: Once the leakage-risk features are excluded, the remaining predictors (mainly Cloud Type and general weather) relate to sky-condition fairly directly, so a well-regularized linear model doesn't lose much by skipping complex interactions, while it gains from lower variance.

**Q: Why balanced accuracy instead of raw accuracy?**
A: The classes are imbalanced — Clear is about 72% of the data. A model that always predicted "Clear" would score 70%+ raw accuracy while being useless. Balanced accuracy averages per-class recall, so it can't be inflated by the majority class.

## Problem 2 — Regression

**Q: Why is cross-city prediction harder than same-city prediction?**
A: Mostly scale, not a failure to learn weather patterns. Davis's plant averages ~164 kW; the other cities average 47-50 kW. Applying a Davis-calibrated model directly produces systematically oversized predictions. A diagnostic rescaling by the target city's own mean recovered R² of 0.55-0.84, showing the underlying weather-to-power relationship transfers reasonably well once scale is handled.

**Q: Why report RMSE, MAE, AND nRMSE — isn't that redundant?**
A: They answer different questions. RMSE penalizes large errors more heavily (useful for spotting bad outlier predictions). MAE is the plain average error size, easier to interpret directly. nRMSE normalizes by the target's range, which lets you compare error across cities with very different power scales — something raw RMSE alone can't do fairly.

**Q: Why doesn't the sequence (GRU) model beat the same-city regression model?**
A: They're not solving the same task. The non-sequence model sees the concurrent weather reading at the exact moment being predicted. The GRU only sees past readings and has to genuinely forecast forward. It does clearly beat a naive persistence baseline (17.58 kW vs. 21.97 kW), which is the fairer comparison for a true forecasting task.

## Problem 3 — Dimension Reduction

**Q: Why did you choose these numbers of components (d=2, 5, 10)?**
A: To span a range from aggressive compression (d=2) to a fairly generous one (d=10, which keeps 94.6% of PCA's explained variance) — enough to see a clear trend rather than testing just one arbitrary size.

**Q: If PCA preserves most of the variance, why didn't it preserve prediction performance?**
A: Explained variance measures how much of the *input features'* spread is captured — it says nothing about whether that spread includes the specific information needed to predict a particular label. In this case, Cloud Type (categorical but highly predictive) compresses poorly into a few continuous dimensions — I verified this directly with a feature-ablation test.

**Q: Why is PCA "unsupervised" — doesn't it use the data?**
A: It uses only the input features, never the label. I verified this directly in the code — the PCA-fitting function's signature doesn't even accept a label argument. That's what makes the downstream comparison meaningful: it tests whether the data's own structure, without looking at the answer, still supports predicting the answer well.

## Problem 4 — Semi-Supervised Learning

**Q: How do you know the unlabeled data actually helped (or didn't)?**
A: By comparing the SSL model against a supervised-only baseline trained on the exact same labeled subset — the only difference is whether the unlabeled pool was used. The SSL gain (macro F1) was slightly negative at all three fractions tested (10%/30%/50%), so in this case it measurably didn't help.

**Q: Why can pseudo-labeling be dangerous?**
A: If the model is confidently wrong about something, adding that wrong guess as a "real" label just teaches the model to be more confidently wrong — it can reinforce mistakes rather than correct them. I guarded against this with a confidence threshold (0.90, chosen via validation) and a cap on pseudo-labels added per round after testing showed uncapped self-training could flood the training set with one class.

**Q: If the pseudo-labels were 100% accurate, why didn't performance improve?**
A: They were correct but redundant — 98% of the first round were the already-easy "Clear" class, which the model already handled well from the labeled data alone. They reinforced existing knowledge instead of helping with the genuinely hard Partly Cloudy/Overcast boundary.

## Problem 5 — Transfer Learning

**Q: What exactly is being transferred?**
A: The pretrained weights of a small neural network trained on Davis — specifically, whatever general weather-to-power relationship those weights encode. Fine-tuning then adapts those weights slightly using a handful of real Amherst samples, rather than learning everything from scratch.

**Q: How do you know transfer learning actually helped, rather than just having a good architecture?**
A: By comparing transfer (Davis-pretrained, fine-tuned) against few-shot (an identical fresh model, trained only on the same k Amherst samples, no Davis involvement). Same architecture, same data, only the starting weights differ — and transfer won at every k tested, most strongly at k=10 (+21.4% RMSE improvement).

**Q: What is negative transfer, and did it happen here?**
A: When using a pretrained model makes things worse than not using it. Yes — directly, during development. An earlier version with a smaller fine-tuning learning rate scored RMSE=140.9 kW at k=10, worse than the few-shot baseline's 38.2 kW. I diagnosed why (the model's output was stuck near Davis's scale, unable to adjust fast enough) and fixed it by matching the pretraining learning rate instead.

## Results

**Q: What was the best model overall?**
A: There isn't a single "best model overall" — balanced accuracy (classification) and RMSE (regression) aren't comparable quantities, so I report the best result per problem rather than a fake combined ranking. If forced to pick one standout: Davis same-city regression, RMSE=15.17 kW, R²=0.953.

**Q: Which Problem 1 class was hardest, and why?**
A: Partly Cloudy (F1=0.566, vs. 0.946 for Clear). It sits between the other two classes on the Clear-Sky Index scale and gets confused with both neighbors, rather than having one dominant failure mode — the expected pattern for a class defined by two adjacent thresholds instead of one.

**Q: Did dimension reduction help?**
A: No — raw, uncompressed features beat every tested PCA and autoencoder representation, at every dimension, on both downstream tasks.

**Q: Did semi-supervised learning help at 10% labels?**
A: No — SSL gain was slightly negative even at 10%, the smallest and most label-scarce condition tested.

**Q: Did transfer learning produce positive or negative transfer?**
A: Both, at different points. The final reported results show clearly positive transfer (+21.4% at k=10). But a negative-transfer episode occurred during development due to a learning-rate issue, before being diagnosed and fixed.

## Limitations

**Q: Why is the mid-day-only dataset a limitation?**
A: Every reading falls in a 10:00-15:00 window — nothing in these results describes model behavior outside that window (early morning, evening, or night, where output is near zero and behaves very differently). A real deployment would need that gap addressed separately.

**Q: Would this model work on data from another year?**
A: Untested directly — every same-city split trains on earlier years and tests on the most recent ~20% within the same multi-year window, but no experiment specifically holds out an entire unseen year to check for longer-term drift (equipment degradation, changing local development, etc.).

**Q: Why only 3-5 hyperparameter candidates per search, instead of an exhaustive grid?**
A: A deliberate scope decision, not an oversight — kept the project focused on testing five distinct ML paradigms rather than exhaustively tuning any one of them. Worth noting: in Problems 1, 2, and 4, untuned or lightly-tuned models repeatedly matched or beat their tuned counterparts, suggesting there may not have been much headroom left to find anyway — though a larger search was never run to confirm that.

**Q: What would you do with another month on this project?**
A: A larger hyperparameter search now that working baselines exist for each paradigm; investigate the Wind Speed domain-shift anomaly flagged in Problem 5; extend transfer learning to the other data-scarce cities (Huron, Santa Barbara, La Jolla) that were only tested zero-shot in Problem 2.
