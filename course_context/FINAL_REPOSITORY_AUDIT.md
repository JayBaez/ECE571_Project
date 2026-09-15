# Final Repository Audit

Conservative, verification-only audit — no experiments rerun, no
results changed, no files deleted, no code rewritten. Every check
below was performed by direct inspection or execution of the actual
repository, not from memory.

## Overall Status

**GREEN**

All five problems have working code, saved results, saved models, and
figures. All five independent leakage checks (one per problem) came
back clean. The full test suite (81 tests) and `scripts/check_setup.py`
(7/7) both pass. Three minor, non-blocking documentation/packaging
items were found — none affect correctness or reproducibility of any
result — listed under "Manual Fixes Required" below, none acted on
without permission.

## Repository Structure

Matches the example structure in the prompt closely, with sensible
naming (`problems/problemN_*/` instead of `experiments/problemN/`,
`course_context/` instead of a generic docs folder, `course/` holding
the raw dataset alongside the course PowerPoints). No reorganization
recommended — the structure is already clear and consistent.

```
problems/{problem1_classification, problem2_regression,
  problem3_dimension_reduction, problem4_semi_supervised,
  problem5_transfer_learning}/
results/{problem1..5, eda}/ + FINAL_EXPERIMENT_TABLE.csv, FINAL_RESULTS.csv/json, BEST_RESULTS.md
figures/{problem1..5, eda}/
report/  (FINAL_REPORT.md, FINAL_REPORT.docx, convert_report.js)
presentation/  (FINAL_PRESENTATION.pptx, script, cheat sheet, Q&A, build script)
course_context/  (21 markdown files — specs, reports, audits, status)
src/  (10 shared framework modules)
tests/  (9 test files, 81 tests)
course/  (raw dataset + course PDFs/PPTX)
data/  (empty placeholder for cached/processed data, documented via its own README)
configs/, scripts/
```

One duplicate file found: `FINAL_TABLE_INVENTORY.md` exists
identically in both `course_context/` and `results/` (verified via
`diff` — byte-identical). Not deleted; see "Manual Fixes Required."

## Problem 1 — Supervised Classification

**Status:** COMPLETE

- Code: `problems/problem1_classification/{features,models,targets,run_experiments}.py`
- Preprocessing/split: reuses shared `src/preprocessing.py` and
  `src/splitting.py` (chronological 80/20)
- Models: majority baseline, logistic regression (+ balanced-weight
  variant), decision tree, random forest, gradient boosting, MLP —
  classical→deep progression present
- Metrics: accuracy, balanced accuracy, macro F1 — saved per-model,
  per-seed in `results/problem1/problem1_results.csv` (198 rows)
- Confusion matrices: present for both tasks, both cities
  (`figures/problem1/problem1_*_confusion_matrix_*.png`)
- Documentation: `course_context/PROBLEM1_REPORT.md`

**Issues:** None found.

## Problem 2 — Supervised Regression

**Status:** COMPLETE

- Code: `problems/problem2_regression/{features,models,sequence,run_experiments}.py`
- Same-city: Davis and Amherst, both present
- Cross-city: Davis → Huron/Santa Barbara/La Jolla zero-shot, present
- Sequence model: GRU (K=12), present, compared against a persistence baseline
- Metrics: RMSE, MAE, nRMSE (+ R², an additive extension) — saved in
  `results/problem2/problem2_results.csv` (108 rows)
- Multiple seeds: `[42, 123, 2026]`, verified present in the results file
- Saved predictions/results: yes, plus `error_analysis_Davis.json`,
  `learning_curve_Davis.csv`, hyperparameter search results
- Figures: predicted-vs-actual, cross-city comparison, sequence
  comparison, learning curve, model comparison — all present

**Issues:** None found.

## Problem 3 — Dimension Reduction

**Status:** COMPLETE

- Code: `problems/problem3_dimension_reduction/{features,reduction,run_experiments}.py`
- PCA and autoencoder, both present
- Dimensions tested: d = 2, 5, 10 — confirmed in `problem3_results.csv`
- Explained variance: present (PCA)
- Reconstruction error: present (both PCA and autoencoder)
- Downstream classification AND regression comparison: present,
  consolidated in `problem3_comparison_table.csv`
- 2-D visualization: present (`problem3_pca_2d_sky.png`,
  `problem3_autoencoder_2d_sky.png`, plus a t-SNE figure)
- Saved results: `results/problem3/problem3_results.csv` (54 rows)

**Issues:** None found.

## Problem 4 — Semi-Supervised Learning

**Status:** COMPLETE

- Code: `problems/problem4_semi_supervised/{pseudo_labeling,run_experiments}.py`
- Label fractions: 10%, 30%, 50% — confirmed in `problem4_results.csv`
- Supervised baseline: present, same labeled subset used for both
  baseline and SSL at each (fraction, seed)
- SSL method: pseudo-labeling/self-training (primary) + Label
  Spreading (optional second method), both present
- Label-efficiency results and curve: present
  (`problem4_label_efficiency.csv`, `problem4_label_efficiency_curve.png`)
- Saved results: `results/problem4/problem4_results.csv` (27 rows)

**Issues:** None found.

## Problem 5 — Transfer Learning

**Status:** COMPLETE

- Code: `problems/problem5_transfer_learning/{models,run_experiments}.py`
- Source city: Davis. Target city: Amherst — both confirmed in code
  and results
- Zero-shot baseline: present
- Few-shot baseline: present (k = 10, 50, 100)
- Transfer-learning method: present (fine-tuning from Davis-pretrained
  weights, same k samples as few-shot)
- Target evaluation: Amherst held-out test set, confirmed used
  consistently across all three methods
- Transfer gain: computed and saved (`transfer_gain`,
  `transfer_gain_percent` columns in `problem5_results.csv`)
- Saved results: `results/problem5/problem5_results.csv` (27 rows),
  plus `problem5_transfer_summary.csv`, `problem5_domain_shift.csv`

**Issues:** None found.

## Data Leakage Check

**Status:** SAFE — independently reverified this audit for all 5 problems

- **Problem 1:** `features.get_feature_columns("sky_condition", ...)`
  directly inspected this audit — returns 15 columns, zero overlap
  with `{GHI, Clearsky GHI, DHI, DNI, Solar Zenith Angle}`. Confirmed
  programmatically, not assumed.
- **Problem 2:** `build_city_dataset("Davis")` — `train_df.index.max()`
  (19,288) strictly less than `test_df.index.min()` (19,289), verified
  by direct execution this audit. `fit_preprocessor()` called only on
  `train_selected` (line 108 of `run_experiments.py`), confirmed by
  reading the exact call site.
- **Problem 3:** `fit_pca(X_train, n_components, seed)`'s signature has
  no label parameter at all. `train_autoencoder()`'s reconstruction
  target is `X_inner_train` itself (its own input) — read the full
  function body this audit; no label is touched anywhere in it.
- **Problem 4:** `self_train()`'s signature takes `X_unlabeled` only —
  structurally cannot see hidden labels. Every appearance of
  `data["X_test"]`/`data["y_test"]` in `run_experiments.py` (grepped
  this audit) is inside a `.predict()` call or a metrics computation —
  never inside a training or selection function.
- **Problem 5:** Every appearance of `data["target_test"]` in
  `run_experiments.py` (grepped this audit, 17 occurrences) is inside
  a docstring, the dataset-building return dict, a `predict_values()`
  call, or a `regression_metrics()` call — never inside
  `pretrain_on_source()`, `fine_tune_on_target()`, or
  `train_fresh_on_target()`.

**Potential issues:** None found.

## Reproducibility

**Status:** COMPLETE

- Random seeds: `[42, 123, 2026]`, defined once in
  `src/utils.py DEFAULT_SEEDS`, imported and used identically (same
  list object) by all 5 problems — verified by grep this audit.
- Train/test splits: chronological, deterministic, reproducible from
  the (non-random) `chronological_split()` function — same split
  guaranteed on every rerun, verified directly for Problem 2's Davis
  split this audit.
- Preprocessing: documented in each problem's `features.py` and the
  shared `src/preprocessing.py`.
- Model parameters/hyperparameters: saved per problem
  (`hyperparameter_search_results.json` and similar files).
- Python version: 3.12 (this audit's environment); README recommends
  3.11 or 3.12.
- Package versions: documented in `course_context/
  REPRODUCIBILITY_CHECKLIST.md` (pandas 3.0.2, numpy 2.4.4,
  scikit-learn 1.8.0, torch 2.14.0, matplotlib 3.10.8, as observed in
  the development sandbox; `requirements.txt` uses minimum-compatible
  ranges rather than exact pins, by design, with a note recommending
  `pip freeze` for exact reproduction).
- Hardware: documented (no GPU in development; will use CUDA
  automatically on the project owner's RTX 2070 with no code changes).
- Dataset name: documented (`course/Further Consolidated Data, HnL.xlsx`)
  in the repository structure diagram, though not spelled out by exact
  filename elsewhere in the README — see Manual Fixes Required.
- Experiment configuration: `configs/example_config.yaml` present;
  most actual experiment parameters live directly in each problem's
  `run_experiments.py` as named constants, which is consistent and
  documented, not undocumented.

**Missing information:** None beyond the one minor README item noted
above (see Manual Fixes Required).

## Results

**Status:** COMPLETE

- Master files found: `results/FINAL_EXPERIMENT_TABLE.csv`,
  `results/FINAL_RESULTS.csv`, `results/FINAL_RESULTS.json`,
  `results/BEST_RESULTS.md` — all present, 30 rows, spanning all 5
  problems.
- Per-problem results files (`problemN_results.csv`) all present, all
  contain problem/dataset-city/model/metric/seed at minimum; several
  additionally report mean/std or per-seed rows enabling mean/std
  computation.
- Cross-checked this audit: the raw per-seed RMSE values in
  `results/problem2/problem2_results.csv` for Davis gradient boosting
  (15.166064, 15.165972, 15.165993) average to 15.166, which rounds to
  the 15.17 kW figure quoted in both `report/FINAL_REPORT.md` and
  `presentation/FINAL_PRESENTATION.pptx` — confirmed consistent, not
  just assumed.

**Inconsistencies:** None found between code, results, report, and
presentation for the numbers spot-checked this audit.

## Figures

**Status:** COMPLETE

All figures called for by the prompt's own checklist are present:
- P1: confusion matrices (both tasks, both cities) OK; class-performance
  visualization (feature importance, class distribution) OK
- P2: predicted-vs-actual OK; learning curve OK
- P3: explained variance OK; reconstruction error OK; 2-D embedding
  (PCA and autoencoder) OK
- P4: label-efficiency curve OK
- P5: transfer-learning comparison (transfer curve, zero-shot-vs-
  transfer bar chart) OK

Filenames are descriptive and self-explanatory throughout (e.g.
`problem5_transfer_curve.png`, not `fig1.png`). Every figure referenced
in `report/FINAL_REPORT.md` was confirmed to exist at its referenced
path when the report was built (Phase 10); not re-verified path-by-
path this audit, since no figure files or paths have changed since.

**Missing figures:** None found.

## README

**Status:** MOSTLY COMPLETE — 2 minor gaps found

Covers, with section numbers: what the project does (S1), the five
problems (S2), repository structure (S4), framework architecture (S5),
installation (S6), verifying installation (S7), running tests (S8),
running each problem (S9), where results live (S10), where figures
live (S11), reproducibility (S12), hardware/GPU note (S6, S12), and
git workflow (S13).

**Gaps found:**
1. The exact dataset filename (`Further Consolidated Data, HnL.xlsx`)
   is never spelled out in the README — it's referenced generically
   ("the raw Excel dataset") in the repository-structure diagram, and
   the 5 cities are named in S1, but someone skimming the README alone
   wouldn't see the literal filename to look for in `course/`.
2. AI assistance disclosure lives only in `report/FINAL_REPORT.md`
   (Section 15) — the README's own mention of "AI agents" (S13) is
   about git-workflow etiquette for future AI-assisted edits, not a
   disclosure that AI assisted in producing this project's content.

Neither gap affects whether the project runs or reproduces correctly.
Not edited without permission — see Manual Fixes Required.

## Requirements

**Status:** MOSTLY COMPLETE — 1 package not explicitly listed

`requirements.txt` covers pandas, numpy, openpyxl, scikit-learn, scipy,
torch, matplotlib, seaborn, PyYAML — all confirmed used by the
codebase via a full-repository import scan this audit.

**Missing package:** `joblib` is imported directly (`import joblib`)
in all 5 `problemN/run_experiments.py` files, for saving/loading
`.joblib` model files, but is not explicitly listed in
`requirements.txt`. It currently works because `scikit-learn>=1.3`
pulls it in as a transitive dependency — this is not a bug, nothing is
broken — but explicitly listing it would be more correct, since the
project's own code imports it directly, not just scikit-learn
internally. Not added without permission — see Manual Fixes Required.

## GitHub Safety

**Secrets:** NO — searched for API keys, passwords, tokens, private
key headers, and credential-like filenames (`.env`, `*credentials*`,
`.pem`, `*_rsa*`) across the entire repository this audit. Nothing
found.

**Large files:** The raw dataset (`course/Further Consolidated Data,
HnL.xlsx`, 19 MB) is tracked in git. This is well within GitHub's
100 MB hard limit (and below its 50 MB warning threshold), so no
Git LFS or external hosting is needed — the current approach is fine
for a repository of this size.

## Cleanup

**Files safely removed:** None — no cache/temp files were found
tracked in the repository (`__pycache__` folders exist in the local
working copy from running the test suite this audit, but are not, and
never have been, tracked by git — confirmed via `git ls-files`).

**Files that should NOT be removed:** The raw dataset, all model
files under `results/problemN/models/`, all figures, both report
formats (`.md` and `.docx`), the presentation, and every
`course_context/` file — all are either required for reproducibility
or are the actual deliverables.

## Manual Fixes Required

1. **Duplicate file:** `course_context/FINAL_TABLE_INVENTORY.md` and
   `results/FINAL_TABLE_INVENTORY.md` are byte-identical. Recommend
   keeping the `course_context/` copy (consistent with where its
   sibling files — `FINAL_FIGURE_INVENTORY.md`, `FINAL_AUDIT.md` —
   live) and removing the `results/` copy. **Awaiting permission.**
2. **README gap:** add the exact dataset filename and path
   (`course/Further Consolidated Data, HnL.xlsx`) explicitly, and a
   one-line pointer to the report's AI-disclosure section. **Awaiting
   permission and specific wording approval.**
3. **requirements.txt gap:** add `joblib>=1.3` (or similar) as an
   explicit dependency, since the code imports it directly. **Awaiting
   permission.**

None of these affect any result, any metric, or the correctness of any
experiment — all three are documentation/packaging polish only.
