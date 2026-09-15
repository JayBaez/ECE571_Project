# Final Figure Inventory

Every figure that could appear in the final report, organized by
problem. **MUST INCLUDE** = tells a core part of the project's story
and has no redundant substitute. **OPTIONAL** = useful supporting
detail, include only if space allows.

## EDA (Phase 3)

| Figure | Shows | Why it matters | Path | Priority |
|---|---|---|---|---|
| output_power_by_city_boxplot | Output Power scale differs sharply by city | Motivates the normalization discussion in P2/P5 | `figures/eda/output_power_by_city_boxplot.png` | MUST INCLUDE |
| correlation_heatmap | GHI/DNI/Output Power correlation structure | Motivates leakage decisions in P1 | `figures/eda/correlation_heatmap.png` | MUST INCLUDE |
| temporal_sampling | 11 readings/day, 30-min spacing | Explains the K=12 sequence window design (P2) | `figures/eda/temporal_sampling.png` | OPTIONAL |
| clear_sky_index_distribution | Sanity-checks P1's sky-condition thresholds | Supports Problem 1's target construction | `figures/eda/clear_sky_index_distribution.png` | OPTIONAL |

## Problem 1 — Classification

| Figure | Shows | Why it matters | Path | Priority |
|---|---|---|---|---|
| problem1_sky_condition_confusion_matrix_Davis | Where the best sky-condition model errs | Core result visualization | `figures/problem1/problem1_sky_condition_confusion_matrix_Davis.png` | MUST INCLUDE |
| problem1_model_comparison | All models, both tasks, both cities | Shows the full model progression at a glance | `figures/problem1/problem1_model_comparison.png` | MUST INCLUDE |
| problem1_sky_condition_feature_importance_Davis | Cloud Type dominates | Supports the P3 "Cloud Type compresses poorly" finding | `figures/problem1/problem1_sky_condition_feature_importance_Davis.png` | OPTIONAL |
| problem1_generation_regime_confusion_matrix_Davis | Best generation-regime model's errors | Secondary task result | `figures/problem1/problem1_generation_regime_confusion_matrix_Davis.png` | OPTIONAL |
| problem1_class_distribution | Class imbalance context | Explains why balanced accuracy was used | `figures/problem1/problem1_class_distribution.png` | OPTIONAL |

## Problem 2 — Regression

| Figure | Shows | Why it matters | Path | Priority |
|---|---|---|---|---|
| problem2_predicted_vs_actual_davis | Best regression model's fit quality | Core result visualization | `figures/problem2/problem2_predicted_vs_actual_davis.png` | MUST INCLUDE |
| problem2_prediction_timeseries_davis | Model tracks daily generation cycle | Concrete, intuitive demonstration of model quality | `figures/problem2/problem2_prediction_timeseries_davis.png` | MUST INCLUDE |
| problem2_cross_city_comparison | Raw zero-shot vs. oracle vs. scale-corrected | Central cross-city finding (scale mismatch, not pattern failure) | `figures/problem2/problem2_cross_city_comparison.png` | MUST INCLUDE |
| problem2_sequence_comparison | GRU vs. persistence vs. non-sequence model | Supports the sequence-modeling finding | `figures/problem2/problem2_sequence_comparison.png` | OPTIONAL |
| problem2_learning_curve | Diminishing returns from more training data | Supporting analysis | `figures/problem2/problem2_learning_curve.png` | OPTIONAL |
| problem2_model_comparison | All same-city models, both cities | Supporting detail | `figures/problem2/problem2_model_comparison.png` | OPTIONAL |

## Problem 3 — Dimension Reduction

| Figure | Shows | Why it matters | Path | Priority |
|---|---|---|---|---|
| problem3_downstream_classification | Raw vs. PCA vs. AE, classification | The central "did compression help?" answer | `figures/problem3/problem3_downstream_classification.png` | MUST INCLUDE |
| problem3_downstream_regression | Raw vs. PCA vs. AE, regression | Same question, regression task | `figures/problem3/problem3_downstream_regression.png` | MUST INCLUDE |
| problem3_reconstruction_error | PCA vs. AE reconstruction quality | Explains WHY AE beat PCA downstream | `figures/problem3/problem3_reconstruction_error.png` | MUST INCLUDE |
| problem3_explained_variance | PCA elbow curve | Standard, expected dimension-reduction figure | `figures/problem3/problem3_explained_variance.png` | OPTIONAL |
| problem3_pca_2d_sky / problem3_autoencoder_2d_sky | 2-D visualizations colored by label | Visual intuition for why d=2 performs poorly | `figures/problem3/problem3_pca_2d_sky.png`, `problem3_autoencoder_2d_sky.png` | OPTIONAL (pick one, not both) |
| problem3_tsne | t-SNE of raw features | Visualization only, not a result | `figures/problem3/problem3_tsne.png` | OPTIONAL |

## Problem 4 — Semi-Supervised Learning

| Figure | Shows | Why it matters | Path | Priority |
|---|---|---|---|---|
| problem4_label_efficiency_curve | Supervised vs. SSL across label fractions | The central SSL-gain result | `figures/problem4/problem4_label_efficiency_curve.png` | MUST INCLUDE |
| problem4_pseudolabel_class_distribution | Pseudo-labels skew toward "Clear" | Explains WHY SSL gain was near-zero | `figures/problem4/problem4_pseudolabel_class_distribution.png` | MUST INCLUDE |
| problem4_ssl_gain | Gain bar chart by fraction | Compact summary of the main finding | `figures/problem4/problem4_ssl_gain.png` | OPTIONAL |
| problem4_confusion_matrix_10pct_supervised / _ssl | Direct error comparison at 10% | Supporting detail | `figures/problem4/problem4_confusion_matrix_10pct_*.png` | OPTIONAL |
| problem4_pseudolabel_confidence | Confidence stayed high across iterations | Supporting detail | `figures/problem4/problem4_pseudolabel_confidence.png` | OPTIONAL |

## Problem 5 — Transfer Learning

| Figure | Shows | Why it matters | Path | Priority |
|---|---|---|---|---|
| problem5_transfer_curve | RMSE vs. Amherst sample count, all 3 methods | THE central Problem 5 figure | `figures/problem5/problem5_transfer_curve.png` | MUST INCLUDE |
| problem5_domain_shift | Davis vs. Amherst feature distributions | Explains why transfer works/is hard | `figures/problem5/problem5_domain_shift.png` | MUST INCLUDE |
| problem5_prediction_vs_truth | Few-shot vs. transfer scatter at k=10 | Direct visual comparison of the two methods | `figures/problem5/problem5_prediction_vs_truth.png` | MUST INCLUDE |
| problem5_time_series | Actual vs. predicted over a test window | Concrete demonstration of transfer model quality | `figures/problem5/problem5_time_series.png` | OPTIONAL |
| problem5_zero_shot_vs_transfer | Bar chart, zero-shot vs. transfer at each k | Redundant with the transfer curve — pick one | `figures/problem5/problem5_zero_shot_vs_transfer.png` | OPTIONAL |

## Summary

**MUST INCLUDE total: 13 figures** (roughly 2-3 per problem) — enough
to tell the complete story without overwhelming an 8-12 page report.
**OPTIONAL: 15 figures** — draw from these only if a specific point
needs more support or space allows an appendix.
