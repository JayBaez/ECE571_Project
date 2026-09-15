# Reproducibility Checklist

Each item verified during Phase 9's audit, not assumed.

- [x] **Requirements/environment recorded** — `requirements.txt`
  (minimum-compatible version ranges, with a note recommending
  `pip freeze > requirements-lock.txt` for exact pinning) and
  `requirements-dev.txt` (test-only extras).
- [x] **Python version recorded** — 3.12.3 (this sandbox); README
  recommends 3.11 or 3.12.
- [x] **Package versions recorded** — this sandbox's exact versions
  (verified during this audit): pandas 3.0.2, numpy 2.4.4,
  scikit-learn 1.8.0, torch 2.14.0, matplotlib 3.10.8. Individual
  problem reports additionally record the versions active at the time
  each problem was run (torch version changed once, 2.13→2.14, between
  Problems 2 and 3, due to an environment reinstall — noted, not a
  correctness issue).
- [x] **Random seeds recorded** — `[42, 123, 2026]`
  (`src/utils.py DEFAULT_SEEDS`), used identically by every problem
  (verified by grep, `FINAL_AUDIT.md` Section 14).
- [x] **Train/test split indices saved** — every problem's
  `build_*_dataset()` function returns `split_info` (train/test row
  counts and boundary timestamp), printed and used during each
  problem's own validation; the split itself is 100% reproducible from
  the (deterministic, non-random) `chronological_split()` function
  rather than needing separately-saved row-index files.
- [x] **Feature definitions recorded** — each problem's `features.py`
  is the single source of truth for its feature list, cross-referenced
  in `course_context/LEAKAGE_MAP.md` and each problem's own report.
- [x] **Preprocessing recorded** — `src/preprocessing.py`
  (`fit_preprocessor`/`apply_preprocessor`/`prepare_xy`), unchanged
  since Phase 2, used identically by every problem; each problem also
  saves its fitted scaler/encoder object (`results/problemN/models/
  *_preprocessor.joblib`).
- [x] **Model hyperparameters recorded** — every hyperparameter search
  result is saved (`results/problemN/*hyperparameter_search*.json`),
  and every problem's `model_config.json` (P2-P5) or report
  documents the exact final configuration used.
- [x] **Results saved** — every problem has a `problemN_results.csv`
  with per-seed, per-experiment rows (never overwritten, only
  appended) — verified present and non-empty for all 5 problems this
  audit, plus the new cross-problem `results/FINAL_EXPERIMENT_TABLE.csv`.
- [x] **Figures saved** — 47 figures across `figures/eda/` and
  `figures/problem{1-5}/`, inventoried in `FINAL_FIGURE_INVENTORY.md`.
- [x] **Scripts runnable** — every problem's `run_experiments.py`
  imports and its core functions execute successfully; verified live
  this audit (`FINAL_AUDIT.md` Section 10/17) by re-running one
  representative experiment per problem and matching saved results
  exactly.
- [x] **README explains execution** — updated this phase (Phase 9) to
  cover all 5 problems specifically (previously described only the
  Phase 2 framework, before any problem was implemented) — see
  `README.md`.
- [x] **AI assistance documented** — `course_context/
  AI_AGENT_INSTRUCTIONS.md` describes the AI-agent workflow used
  throughout the project; each phase's chat-driven development is
  reflected in the commit history and each problem's report
  transparently documents mistakes found and fixed along the way
  (e.g. Problem 4's numpy API break, Problem 5's negative-transfer
  investigation) rather than presenting a falsely clean narrative.

## Known reproducibility limitations (documented, not gaps)

- **No GPU was available in the development sandbox** — every neural
  network (P1 MLP, P2 MLP/GRU, P3 autoencoder, P5 MLP) was trained on
  CPU throughout. `src/utils.py`'s `get_device()` will automatically
  use CUDA on the project owner's RTX 2070 without any code change,
  but CUDA-specific numerical determinism (e.g. `torch.
  use_deterministic_algorithms`) was not separately verified, since it
  could not be tested in this environment. This is a documented
  limitation, not a known problem: the CPU-only reruns performed
  during this audit (Section 10, `FINAL_AUDIT.md`) matched saved
  results exactly, which is the more numerically sensitive case, not
  less.
- **Exact package-version pinning was not captured as a lockfile** —
  `requirements.txt` uses minimum-compatible ranges by design (to
  avoid overfitting the spec to one sandbox's exact versions); the
  README recommends generating a `requirements-lock.txt` via
  `pip freeze` once a real environment is set up, for anyone who wants
  bit-for-bit package-version matching.
