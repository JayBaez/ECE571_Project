# src/: shared framework code

Reusable modules used by all five problems. Every "fit" step only ever sees training data.

| Module | Purpose |
|---|---|
| `data_loader.py` | Load a city sheet from the raw Excel workbook (default path `course/Further Consolidated Data, HnL.xlsx`, relative to the repo root). |
| `cleaning.py` | Detect and report missing values, duplicates, and anomalies. |
| `feature_engineering.py` | Clear-Sky Index, cyclical time features, lag features. |
| `splitting.py` | Chronological, cross-city, random-labeled-subset, and few-shot splits. |
| `preprocessing.py` | `fit_preprocessor()` on train only, `apply_preprocessor()` on both, `prepare_xy()`. |
| `torch_utils.py` | Training loop with early stopping and checkpointing for any `nn.Module`. |
| `evaluation.py` | Classification and regression metrics, multi-seed aggregation. |
| `visualization.py` | Plots, always saved to file. |
| `experiment_runner.py` | Experiment artifact folders and the append-only `results/experiment_history.csv`. |
| `utils.py` | Seeds (`DEFAULT_SEEDS = [42, 123, 2026]`), `get_device()`, `ensure_dir()`. |

Tests for every module are in `tests/` (`pytest tests/`). See the top-level `README.md` for setup and how to run each problem.
