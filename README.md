# Solar PV Power Prediction — ECE571 Machine Learning Course Project

## 1. What this project is

This project uses a real-world solar irradiance and photovoltaic (PV)
output dataset covering 5 cities (Amherst MA, Davis CA, Huron SD,
Santa Barbara CA, La Jolla CA) to explore five different machine
learning paradigms. The dataset and full project spec are summarized
in `course_context/` — see especially `TEACHER_EXPECTATIONS.md` for
the exact requirements and grading rubric.

## 2. The five problems

1. **Supervised Classification** — predict a sky-condition or
   generation-regime category from weather features.
2. **Supervised Regression** (core task) — predict continuous PV
   Output Power, same-city, cross-city, and as a short-term forecast.
3. **Dimension Reduction** — compress features to 2/5/10 dimensions
   and measure whether that helps or hurts Problems 1 and 2.
4. **Semi-Supervised Learning** — learn from a small labeled fraction
   (10%/30%/50%) plus a larger unlabeled pool.
5. **Transfer Learning** — use data-rich Davis to help data-scarce
   Amherst.

Full detail on each problem, including exact label definitions,
required metrics, and known risks, is in
`course_context/TEACHER_EXPECTATIONS.md` and
`course_context/EXPERIMENT_PLAN.md`.

## 3. Current project status

**Phase 2 (this stage) is complete: the reusable ML experimentation
framework exists, is fully tested (79/79 tests passing), and has been
proven end-to-end with a framework demo. No ML problems have been
solved with real project data yet.** See
`course_context/PROJECT_STATUS.md` for the up-to-date tracker.

## 4. Repository structure

```
.
├── course/                  Course PowerPoints/PDFs + the raw Excel dataset
├── course_context/          Knowledge base (read this first)
├── data/                    Cached/processed data (empty for now — see data/README.md)
├── src/                     Reusable framework code (see Architecture below)
├── configs/                 YAML experiment configs
├── problems/                One folder per problem — not yet implemented
├── results/                 experiment_history.csv, per-problem/demo results
├── figures/                 Saved plots, one subfolder per problem/demo
├── models/                  Saved trained models (empty for now)
├── logs/                    Run logs (empty for now)
├── scripts/
│   ├── check_setup.py       Basic environment/framework validation script
│   └── framework_demo.py    Tiny end-to-end pipeline demo (NOT a project result)
├── tests/                   Unit tests for every src/ module (pytest)
├── requirements.txt
├── requirements-dev.txt     Extra packages needed only to run the test suite
├── pytest.ini
└── .gitignore
```

## 5. Framework architecture (Phase 2)

The pipeline flows one direction, and every "fit" step only ever sees
training data:

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
early stopping and checkpointing) that works with *any* `nn.Module` a
later phase defines — no custom model hierarchy needed.

**Leakage protection, concretely:**
- Preprocessing: `fit_*()` functions take a training DataFrame only;
  `apply_*()` functions take an already-fitted object. There's no
  function that fits on combined or test-only data.
- Splitting: `chronological_split()` never shuffles and returns exactly
  which rows/timestamp ended up on each side; `verify_no_overlap()`
  can double-check any split.
- Target/feature separation: `prepare_xy()` makes "what's a feature"
  an explicit, loggable list — so a leakage-risk column (e.g. `GHI`
  when predicting the sky-condition label derived from it) has to be
  deliberately excluded, not accidentally included.

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
https://pytorch.org/get-started/locally/.

## 7. How to verify the installation

```bash
python scripts/check_setup.py
```

This checks that all packages import, all `src/` modules import, the
Excel dataset can be found and opened, a sheet can be loaded, random
seeds are reproducible, and GPU detection runs — without training any
model. Every check should print `[PASS]`.

## 8. How to run the test suite

```bash
pip install -r requirements-dev.txt
pytest tests/
```

This runs 79 tests covering every `src/` module: data loading (happy
path and error cases), cleaning (including a regression test against
the real, known 4-missing-row Amherst finding), preprocessing
(explicitly checking a scaler is fit on train data only, not
combined), splitting (order preservation, zero overlap, reproducible
random sampling), evaluation (hand-verified metric values), the
results/leaderboard system (including a test for the exact CSV-schema
bug found and fixed during this phase), and the PyTorch training loop
(loss decreasing, early stopping triggering correctly). All 79 should
pass.

## 9. How to run the framework demonstration

```bash
python scripts/framework_demo.py
```

This runs the entire pipeline above — load, clean, engineer features,
split, preprocess, train, evaluate, save — on a small synthetic
dataset with a plain Linear Regression model. **This is a framework
test, not a project result:** it proves the pipeline works end-to-end,
using made-up data and a deliberately simple model. Its output is
saved under `results/framework_demo/` and `figures/framework_demo/`,
clearly separate from any real Problem 1–5 results.

## 10. How experiments will eventually be run

Each problem will get its own runner script inside its
`problems/problemN_*/` folder. That script will:

1. Load a config with `experiment_runner.load_config()` (see
   `configs/example_config.yaml` for the shape).
2. Use `data_loader`, `cleaning`, `feature_engineering`, and
   `splitting` to prepare data.
3. Use `preprocessing.fit_preprocessor()` / `apply_preprocessor()` /
   `prepare_xy()` to get train/test X and y.
4. Train a model (problem-specific code — a scikit-learn estimator, or
   a PyTorch model trained via `torch_utils.train_torch_model()`).
5. Use `evaluation.py` to compute metrics (aggregated across seeds
   with `aggregate_across_seeds()` where the spec requires it) and
   `visualization.py` to save plots.
6. Use `experiment_runner.create_experiment_dir()` +
   `save_metrics_json()` / `save_experiment_config()` /
   `save_predictions_csv()` to save full artifacts, and
   `save_result()` to append the summary row.

## 11. How results are stored

- `results/experiment_history.csv` — every experiment ever run, one
  row each, never overwritten.
- `results/problemN/EXPERIMENT_ID/` — full artifacts per experiment:
  `metrics.json`, `config.yaml`, `predictions.csv`, and
  `training_log.csv` for neural network runs.
- `experiment_runner.get_leaderboard(metric="rmse")` — queries
  `experiment_history.csv` on demand and returns the best results,
  correctly sorted for whichever metric you ask about (lower-is-better
  metrics like RMSE sort ascending, higher-is-better metrics like
  balanced accuracy sort descending). Pass `save_to=...` to also save
  a snapshot to a file.

**Only one row currently exists in `results/experiment_history.csv`:
the framework demo (clearly labeled, not a real result). No number
anywhere in this repository represents an actual Problem 1–5 finding
yet.**

## 12. Using Git/GitHub Desktop

This repo is managed with GitHub Desktop. After AI-assisted changes
(Claude Code or otherwise) are made to the repository, review the diff
in GitHub Desktop before committing — that's the right point to catch
anything you don't understand or agree with. Commit messages should
briefly describe what phase/problem the change belongs to. AI agents
working in this repo are instructed not to commit or push on their own — see `course_context/AI_AGENT_INSTRUCTIONS.md`.
