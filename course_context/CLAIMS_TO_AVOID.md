# Claims to Avoid

Conclusions the final report and presentation must NOT make, because
the experiments don't support them. Each is paired with what the
evidence actually shows.

## General (apply to any ML project, restated for this one)

- ❌ "Model X is universally better than Model Y."
  ✅ Model X performed best **on this dataset, this city, this split,
  these seeds** — e.g. logistic regression beat every ensemble for
  Davis sky-condition, but Random Forest won for Amherst
  generation-regime. Different tasks favored different models.

- ❌ "Transfer learning always works."
  ✅ Transfer beat few-shot at every k tested here — but only AFTER a
  genuine negative-transfer episode was found and fixed
  (`PROBLEM5_REPORT.md`, Section 12). The same architecture with a
  smaller, "more conservative" learning rate produced severe negative
  transfer. Transfer learning's benefit here was conditional on
  getting the fine-tuning setup right, not automatic.

- ❌ "Semi-supervised learning always improves performance."
  ✅ It didn't, here — SSL gain was slightly negative at every label
  fraction (10%/30%/50%). This project's own evidence directly
  contradicts a blanket "SSL helps" claim.

- ❌ "PCA proves the features are independent."
  ✅ PCA's explained-variance curve shows how much VARIANCE is
  captured by fewer dimensions — it says nothing about statistical
  independence, and this project never tested or claimed the latter.

- ❌ "High accuracy means perfect forecasting."
  ✅ Even the strongest same-city model (Davis, gradient_boosting,
  R²=0.953) has real, characterized error patterns — errors
  concentrate 2x above average during rapidly-changing irradiance
  (`PROBLEM2_REPORT.md`, Section 19). Strong aggregate metrics don't
  mean uniformly reliable predictions.

- ❌ "The model generalizes to every city."
  ✅ The opposite was directly demonstrated: a Davis-trained model
  applied raw to Huron/Santa Barbara/La Jolla/Amherst produced R² as
  low as -72 (`PROBLEM2_REPORT.md`, `PROBLEM5_REPORT.md`) — severe
  negative transfer without adaptation. Generalization required
  either explicit scale handling or fine-tuning; it was never free.

## Specific to this project's actual results

- ❌ "Dimension reduction is a good idea for this dataset."
  ✅ The opposite was found: raw features beat PCA and the autoencoder
  at every tested dimension (d=2/5/10), on both downstream tasks
  (`PROBLEM3_REPORT.md`). Compression hurt here — a real, reported
  finding, not a caveat to downplay.

- ❌ "The sky-condition classifier works equally well for all three
  classes."
  ✅ Partly Cloudy and Overcast are consistently harder than Clear —
  confirmed across confusion matrices in both Problem 1 and Problem 4.

- ❌ "Pseudo-labeling failed because it added wrong labels."
  ✅ The opposite — pseudo-labels were 100% accurate (verified via
  offline diagnostic). The failure mode was redundancy (reinforcing
  the easy class), not incorrectness. This distinction matters and
  should not be flattened into a generic "SSL didn't work" statement.

- ❌ "Target normalization is necessary for cross-city transfer."
  ✅ Tested directly and found NOT to help once the fine-tuning
  learning rate was corrected (`PROBLEM5_REPORT.md`, Section 15) — raw
  kW performed as well as or slightly better than explicit
  normalization at k=50. The real fix was the learning rate.

- ❌ "Wind Speed is dramatically different between Davis and Amherst
  because of climate."
  ✅ The standardized mean difference (2.6) is unusually large — large
  enough that it was explicitly flagged as **possibly a sensor or unit
  artifact**, not confirmed as a genuine climate fact
  (`PROBLEM5_REPORT.md`, Section 17). Do not present this as settled.

- ❌ "3 years of data is better than 6 years for regression."
  ✅ The 3-year Davis sheet did score better (RMSE 12.04 vs. 15.17),
  but the two sheets don't share a test period, so this comparison is
  confounded — it may reflect an easier test window, not less data
  being genuinely better (`PROBLEM2_REPORT.md`, Section 21). Present
  this as an open, unresolved question, not a conclusion.

- ❌ "Freezing early layers helps transfer learning."
  ✅ Tested directly at k=50 and found to make essentially no
  difference (23.35 full fine-tuning vs. 23.40 frozen).
