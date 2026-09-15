# Solar PV Power Prediction — ECE571 Machine Learning Course Project

## 1. What this project is

This project uses a real-world solar irradiance and photovoltaic (PV)
output dataset covering 5 cities (Amherst MA, Davis CA, Huron SD,
Santa Barbara CA, La Jolla CA) to explore five machine learning
paradigms. The dataset and full project spec are summarized in
`course_context/` — see especially `TEACHER_EXPECTATIONS.md` for the
exact requirements/grading rubric and `PROJECT_STORY.md` for a concise
summary of what was found.

**All five required ML problems are complete.** See
`course_context/PROJECT_STATUS.md` for the up-to-date tracker and
`results/BEST_RESULTS.md` for the headline result of each problem.

## 2. The five problems (all complete)

1. **Supervised Classification** (`problems/problem1_classification/`)
   — sky-condition and generation-regime classifiers. Best result:
   0.77-0.95 balanced accuracy depending on task/city. See
   `course_context/PROBLEM1_REPORT.md`.
2. **Supervised Regression** (`problems/problem2_regression/`, core
   task) — same-city, cross-city, and K=12 sequence forecasting of
   Output Power. Best result: RMSE=15.17 kW (Davis, same-city). See
   `course_context/PROBLEM2_REPORT.md`.
3. **Dimension Reduction** (`problems/problem3_dimension_reduction/`)
   — PCA and an autoencoder at d=2/5/10, compared against raw features
   on both downstream tasks. Finding: raw features won at every
   dimension. See `course_context/PROBLEM3_REPORT.md`.
4. **Semi-Supervised Learning** (`problems/problem4_semi_supervised/`)
   — pseudo-labeling and Label Spreading at 10%/30%/50% labels.
   Finding: SSL gain was slightly negative at every fraction (with a
   clear, evidence-based explanation). See
   `course_context/PROBLEM4_REPORT.md`.
5. **Transfer Learning** (`problems/problem5_transfer_learning/`) —
   Davis→Amherst zero-shot/few-shot/transfer for Output Power. Finding:
   transfer beat few-shot at every k, most strongly at k=10 (+21.4%
   RMSE). See `course_context/PROBLEM5_REPORT.md`.

Full detail on each problem, including exact label definitions,
required metrics, and known risks, is in
`course_context/TEACHER_EXPECTATIONS.md` and
`course_context/EXPERIMENT_PLAN.md`. `course_context/FINAL_AUDIT.md`
(Phase 9) documents a full leakage/reproducibility audit across all
five; `course_context/CLAIMS_TO_AVOID.md` lists conclusions the
experiments do NOT support.

## 3. Current project status

**All 5 ML problems are COMPLETE, audited (Phase 9), and
reproducible** — 5 representative experiments (one per problem) were
independently rerun during the Phase 9 audit and matched saved results
exactly. The final written report and presentation have not yet been
created. See `course_context/PROJECT_STATUS.md` for the detailed
tracker.

## 4. Repository structure

```
.
├── course/                  Course PowerPoints/PDFs + the raw Excel dataset
├── course_context/          Knowledge base — READ THIS FIRST, especially
│                             PROJECT_STORY.md and the 5 PROBLEMN_REPORT.md files
├── src/                     Reusable framework code (see Architecture below)
├── problems/                One folder per problem, each with its own
│   ├── problem1_classification/     features.py, models.py, targets.py, run_experiments.py
│   ├── problem2_regression/         features.py, models.py, sequence.py, run_experiments.py
│   ├── problem3_dimension_reduction/ features.py, reduction.py, run_experiments.py
│   ├── problem4_semi_supervised/    pseudo_labeling.py, run_experiments.py
│   └── problem5_transfer_learning/  models.py, run_experiments.py
├── results/                 Per-problem results CSVs, saved models, plus
│   ├── problem1/ .. problem5/       problemN_results.csv, models/, etc.
│   ├── FINAL_EXPERIMENT_TABLE.csv   master table spanning all 5 problems
│   ├── FINAL_RESULTS.csv/.json      same data, alternate format
│   └── BEST_RESULTS.md              headline result + justification, per problem
├── figures/                 Saved plots, one subfolder per problem (+ eda/)
├── configs/                 YAML experiment configs
├── scripts/
│   ├── check_setup.py       Environment/framework validation script
│   ├── run_eda.py           Phase 3 exploratory data analysis script
│   └── framework_demo.py    Tiny synthetic-data pipeline demo (NOT a project result)
├── tests/                   Unit tests for every src/ module (pytest) — 81 tests
├── requirements.txt
├── requirements-dev.txt     Extra packages needed only to run the test suite
├── pytest.ini
└── .gitignore
```

## 5. Framework architecture

The pipeline flows one direction, and every "fit" step only ever sees
training data — this is the shared foundation every one of the 5
problems builds on:

```
Excel file (course/*.xlsx)
    ↓  src/data_loader.py          — load_city("Davis"), load_sheet(...)
Raw DataFrame
    ↓  src/cleaning.py             — detect + report missing values, duplicates, anomalies
    ↓  src/feature_engineering.py  — Clear-Sky Index, time-cyclical, lag features (toggleable)
Cleaned + featured DataFrame
    ↓  src/splitting.py            — chronological / cross-city / random-subset / few-shot
Train DataFrame, Test DataFrame
    ↓  src/preprocessing.py        — fit_preprocessor() on TRAIN ONLY, apply_preprocessor() on both
    ↓  src/preprocessing.py        — prepare_xy() to explicitly split into X (features) and y (target)
Scaled, encoded X_train/y_train, X_test/y_test
    ↓  a model — any scikit-learn estimator (.fit/.predict), or any
    ↓  torch.nn.Module trained via src/torch_utils.train_torch_model()
Predictions
    ↓  src/evaluation.py           — classification/regression metrics, multi-seed aggregation
    ↓  src/visualization.py        — plots, always saved to file
    ↓  src/experiment_runner.py    — full artifact folder + one summary row in
                                      results/experiment_history.csv (never overwritten)
```

**Why no separate "Model" class:** scikit-learn estimators already
share `.fit(X, y)` / `.predict(X)` for free — wrapping them would only
add indirection. PyTorch doesn't give you that for free, so
`torch_utils.py` provides just the missing piece (a training loop with
early stopping and checkpointing) that works with *any* `nn.Module`
each problem defines — proven across 5 different network architectures
(P1's classifier MLP, P2's regressor MLP and GRU, P3's autoencoder,
P5's transfer MLP) with zero changes to the shared training loop.

**Leakage protection, concretely (audited in full in
`course_context/FINAL_AUDIT.md`, Section 8):**
- Preprocessing: `fit_*()` functions take a training DataFrame only;
  `apply_*()` functions take an already-fitted object.
- Splitting: `chronological_split()` never shuffles and returns
  exactly which rows/timestamps ended up on each side.
- Target/feature separation: each problem's `features.py` makes "what's
  a feature" an explicit, documented list — leakage-risk columns
  (e.g. `GHI` when predicting the sky-condition label derived from it)
  are deliberately excluded, not accidentally included.

## 6. How to create/install the Python environment

Recommended Python version: 3.11 or 3.12.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**GPU note (RTX 2070):** the plain `pip install torch` in
`requirements.txt` may give you a CPU-only build depending on your
system. For CUDA acceleration, instead run the install command
generated for your system at
https://pytorch.org/get-started/locally/. Every neural network in this
project was developed and validated on CPU (no GPU in the development
sandbox) — `src/utils.py`'s `get_device()` will automatically switch
to CUDA with no code changes once you run it on a CUDA-capable machine.

## 7. How to verify the installation

```bash
python scripts/check_setup.py
```

Checks that all packages import, all `src/` modules import, the Excel
dataset can be found and opened, a sheet can be loaded, random seeds
are reproducible, and GPU detection runs — without training any model.
Every check should print `[PASS]` (7/7).

## 8. How to run the test suite

```bash
pip install -r requirements-dev.txt
pytest tests/
```

81 tests covering every `src/` module: data loading, cleaning,
feature engineering, preprocessing (explicitly checking a scaler is
fit on train data only), splitting (order preservation, zero overlap,
reproducible sampling), evaluation (hand-verified metric values,
including the R² addition made during Problem 2), the results/
leaderboard system, and the PyTorch training loop (loss decreasing,
early stopping, and the classification `y_dtype` support added during
Problem 1).

## 9. How to run each problem

Each problem's experiments are driven from its own
`run_experiments.py`, which imports shared functions from `src/` and
(where applicable) from earlier problems' modules — e.g. Problem 5
reuses Problem 2's exact dataset-building functions, and Problem 4
reuses Problem 1's exact dataset build, to guarantee identical splits.

```bash
# From the project root, with the venv active:
python problems/problem1_classification/run_experiments.py
python problems/problem2_regression/run_experiments.py
python problems/problem3_dimension_reduction/run_experiments.py
python problems/problem4_semi_supervised/run_experiments.py
python problems/problem5_transfer_learning/run_experiments.py
```

**Note:** during development, each script's stages were run as several
smaller invocations (rather than one continuous run) to stay within
the development sandbox's per-command execution-time limit — this is
documented in each problem's own report and does not affect result
validity (Phase 9's audit independently reran one representative
experiment per problem and matched saved results exactly — see
`course_context/FINAL_AUDIT.md`, Section 10). Running the full script
top-to-bottom on your own machine is expected to work but may take
longer in a single sitting than the individual functions did when run
separately.

## 10. Where results are stored

- `results/problemN/problemN_results.csv` — every experiment for that
  problem, one row per (model, seed, ...) — never overwritten, only
  appended.
- `results/problemN/models/` — saved fitted models (`.joblib` for
  scikit-learn, `.pt` for PyTorch state dicts) and preprocessing
  objects, all verified loadable.
- `results/FINAL_EXPERIMENT_TABLE.csv` / `FINAL_RESULTS.csv` /
  `FINAL_RESULTS.json` — one master table spanning all 5 problems
  (Phase 9), ready to use directly when writing the final report.
- `results/BEST_RESULTS.md` — the selected headline result for each
  problem, with justification.
- `results/experiment_history.csv` — every experiment ever run across
  every problem, queryable via `src/experiment_runner.py`'s
  `get_leaderboard()`.

## 11. Where figures are stored

`figures/problemN/` (one subfolder per problem) and `figures/eda/`
(Phase 3 exploratory analysis). `course_context/
FINAL_FIGURE_INVENTORY.md` (Phase 9) catalogs every figure across the
whole project with a MUST INCLUDE / OPTIONAL recommendation for the
final report.

## 12. Reproducibility information

See `course_context/REPRODUCIBILITY_CHECKLIST.md` (Phase 9) for the
full checklist. In brief: fixed seeds `[42, 123, 2026]` used
identically across every problem; every preprocessing/scaling step is
fit on training data only (audited, `FINAL_AUDIT.md` Section 8); every
hyperparameter search result is saved; 5 representative experiments
(one per problem) were independently rerun during the Phase 9 audit
and matched saved results exactly.

## 13. Using Git/GitHub Desktop

This repo is managed with GitHub Desktop. After AI-assisted changes
(Claude Code or otherwise) are made to the repository, review the diff
in GitHub Desktop before committing — that's the right point to catch
anything you don't understand or agree with. Commit messages should
briefly describe what phase/problem the change belongs to. AI agents
working in this repo are instructed not to commit or push on their
own — see `course_context/AI_AGENT_INSTRUCTIONS.md`.
