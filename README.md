# Solar PV Power Prediction — ECE571 Machine Learning Course Project

## 1. What this project is

This project uses a real-world solar irradiance and photovoltaic (PV)
output dataset covering 5 cities (Amherst MA, Davis CA, Huron SD,
Santa Barbara CA, La Jolla CA) to explore five machine learning
paradigms. The dataset is profiled in `course_context/DATASET_PROFILE.md`.
The final write-up is `report/FINAL_REPORT.docx`, and the accompanying
slide deck is `presentation/PV_ML_Final_Presentation.pptx` (a narrated
version is `presentation/PV_ML_Final_Presentation_Voice_Over.mp4`).

**Dataset not included.** The raw Excel workbook is not stored in this
repo. Before running anything, place it at
`course/Further Consolidated Data, HnL.xlsx` (path relative to the repo
root, set by `DEFAULT_DATA_PATH` in `src/data_loader.py`).

## 2. The five problems

1. **Supervised Classification** (`problems/problem1_classification/`)
   — sky-condition and generation-regime classifiers. Best result:
   0.77-0.95 balanced accuracy depending on task/city.
2. **Supervised Regression** (`problems/problem2_regression/`, core
   task) — same-city, cross-city, and K=12 sequence forecasting of
   Output Power. Best result: RMSE=15.17 kW (Davis, same-city).
3. **Dimension Reduction** (`problems/problem3_dimension_reduction/`)
   — PCA and an autoencoder at d=2/5/10, compared against raw features
   on both downstream tasks. Finding: raw features won at every
   dimension.
4. **Semi-Supervised Learning** (`problems/problem4_semi_supervised/`)
   — pseudo-labeling and Label Spreading at 10%/30%/50% labels.
   Finding: SSL gain was slightly negative at every fraction (with a
   clear, evidence-based explanation).
5. **Transfer Learning** (`problems/problem5_transfer_learning/`) —
   Davis→Amherst zero-shot/few-shot/transfer for Output Power. Finding:
   transfer beat few-shot at every k, most strongly at k=10 (+21.4%
   RMSE).

Each problem folder has its own `README.md` (files, how to run, outputs,
headline result). Full detail on methods and results is in
`report/FINAL_REPORT.docx`. `course_context/LEAKAGE_MAP.md` lists which
features are safe vs. off-limits for each target, and
`course_context/ML_METHOD_MAP.md` maps course-taught methods to problems.

## 3. Repository structure

```
.
├── course/                  NOT in the repo: put the raw Excel dataset here
├── course_context/          Working notes: DATASET_PROFILE.md, LEAKAGE_MAP.md,
│                             ML_METHOD_MAP.md
├── report/                  FINAL_REPORT.docx — the final written report
├── presentation/            Final slide deck (.pptx) and narrated version (.mp4)
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
├── scripts/
│   ├── check_setup.py       Environment/framework validation script
│   └── run_eda.py           Exploratory data analysis script
├── tests/                   Unit tests for every src/ module (pytest) — 81 tests
├── requirements.txt
├── requirements-dev.txt     Extra packages needed only to run the test suite
├── pytest.ini
└── .gitignore
```

## 4. Framework architecture

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

**Leakage protection, concretely (see `course_context/LEAKAGE_MAP.md`
for the full per-problem breakdown):**
- Preprocessing: `fit_*()` functions take a training DataFrame only;
  `apply_*()` functions take an already-fitted object.
- Splitting: `chronological_split()` never shuffles and returns
  exactly which rows/timestamps ended up on each side.
- Target/feature separation: each problem's `features.py` makes "what's
  a feature" an explicit, documented list — leakage-risk columns
  (e.g. `GHI` when predicting the sky-condition label derived from it)
  are deliberately excluded, not accidentally included.

## 5. How to create/install the Python environment

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

## 6. How to verify the installation

```bash
python scripts/check_setup.py
```

Checks that all packages import, all `src/` modules import, the Excel
dataset can be found and opened, a sheet can be loaded, random seeds
are reproducible, and GPU detection runs — without training any model.
Every check should print `[PASS]` (7/7); the dataset checks need the Excel
file in `course/` (see Section 1).

## 7. How to run the test suite

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

## 8. How to run each problem

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

## 9. Where results are stored

- `results/problemN/problemN_results.csv` — every experiment for that
  problem, one row per (model, seed, ...) — never overwritten, only
  appended.
- `results/problemN/models/` — saved fitted models (`.joblib` for
  scikit-learn, `.pt` for PyTorch state dicts) and preprocessing
  objects, all verified loadable.
- `results/FINAL_EXPERIMENT_TABLE.csv` / `FINAL_RESULTS.csv` /
  `FINAL_RESULTS.json` — one master table spanning all 5 problems,
  the source data for the final report's tables.
- `results/BEST_RESULTS.md` — the selected headline result for each
  problem, with justification.
- `results/experiment_history.csv` — every experiment ever run across
  every problem, queryable via `src/experiment_runner.py`'s
  `get_leaderboard()`.

## 10. Where figures are stored

`figures/problemN/` (one subfolder per problem) and `figures/eda/`
(exploratory analysis). The subset actually used in the final report
and presentation is shown in `report/FINAL_REPORT.docx` and the slides.

## 11. Reproducibility information

Fixed seeds `[42, 123, 2026]` (`src/utils.py`'s `DEFAULT_SEEDS`) are
used identically across every problem. Every preprocessing/scaling
step is fit on training data only — enforced by the
`fit_*()`/`apply_*()` split in `src/preprocessing.py` and checked
directly in `tests/test_preprocessing.py`. Every hyperparameter search
result is saved into its problem's `results/problemN/` folder (e.g.
`hyperparameter_search_results_<city>.json`).

