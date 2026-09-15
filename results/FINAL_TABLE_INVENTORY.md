# Final Table Inventory

Tables that could appear in the final report.

| Table | Problem | Purpose | File path / data source |
|---|---|---|---|
| Model comparison (Davis sky-condition) | 1 | Show the full baseline→tuned model progression and headline balanced accuracy/macro F1 | `results/problem1/problem1_results.csv` (filter city=Davis, task=sky_condition, ablation_group=full) |
| Per-class metrics (best sky-condition model) | 1 | Show which classes are hardest (Partly Cloudy/Overcast) | `results/problem1/per_class_metrics_sky_condition_Davis.csv` |
| Feature ablation (sky-condition) | 1 | Demonstrate the DHI/DNI/Solar Zenith Angle leakage risk empirically | `results/problem1/problem1_results.csv` (ablation_group column) |
| Same-city model comparison | 2 | Show RMSE/MAE/nRMSE across all models, Davis and Amherst | `results/problem2/problem2_results.csv` (experiment=same_city) |
| Cross-city zero-shot results | 2 | The central scale-mismatch finding, with oracle and diagnostic comparisons | `results/problem2/problem2_results.csv` (experiment=cross_city_zero_shot) |
| Sequence vs. non-sequence comparison | 2 | GRU vs. persistence vs. same-city model | `results/problem2/problem2_results.csv` (experiment=sequence) |
| 3yr vs. 6yr ablation | 2 | The confounded-but-honest data-volume comparison | `results/problem2/problem2_results.csv` (experiment=3yr_vs_6yr) |
| Central Problem 3 comparison table | 3 | Raw vs. PCA-2/5/10 vs. AE-2/5/10, all metrics side by side | `results/problem3/problem3_comparison_table.csv` (already report-ready) |
| Feature ablation (Cloud Type) | 3 | Shows Cloud Type's value even under compression | `results/problem3/problem3_results.csv` (notes column, "feature_ablation") |
| Label-efficiency table | 4 | Supervised vs. SSL mean±std at 10/30/50%, with gain | `results/problem4/problem4_label_efficiency.csv` (already report-ready) |
| Pseudo-label summary | 4 | Count, accuracy (offline diagnostic), class distribution per fraction | `results/problem4/problem4_pseudolabel_summary.csv` |
| Central Problem 5 transfer table | 5 | Zero-shot/Few-shot/Transfer at every k, all metrics | `results/problem5/problem5_transfer_summary.csv` (already report-ready) |
| Domain-shift table | 5 | Davis vs. Amherst distributional comparison (SMD) | `results/problem5/problem5_domain_shift.csv` (already report-ready) |
| Freezing & normalization ablations | 5 | Both Problem 5 ablations, side by side | `results/problem5/problem5_results.csv` (frozen_layers, target_normalization columns) |
| **Master experiment table** | **All** | **Single table spanning all 5 problems, ready to drop into the report or appendix** | `results/FINAL_EXPERIMENT_TABLE.csv` |
| **Best-result summary** | **All** | **One paragraph per problem justifying the selected headline result** | `results/BEST_RESULTS.md` |

## Notes

- Tables marked "already report-ready" were built during each
  problem's own phase specifically for this purpose — no further
  reformatting needed.
- The master table (`FINAL_EXPERIMENT_TABLE.csv`) and `BEST_RESULTS.md`
  are new this phase (Phase 9) and are the recommended starting point
  for the report's results section — every other table above supports
  or elaborates on one row/claim from these two.
