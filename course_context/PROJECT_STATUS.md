# Project Status

**Last updated:** Phase 2 completion (ML framework build + validation).
**Future AI agents: update this file as the project progresses. Keep
entries short — status + one-line note, not a log of everything done.**

```
Phase 0 — Course material & spec analysis: COMPLETE
Phase 1 — Repository construction:          COMPLETE
Phase 2 — ML framework:                     COMPLETE
Phase 3 — Dataset validation (code-level):  NOT STARTED
Problem 1 (Classification):                 NOT STARTED
Problem 2 (Regression):                     NOT STARTED
Problem 3 (Dimension Reduction):            NOT STARTED
Problem 4 (Semi-Supervised Learning):       NOT STARTED
Problem 5 (Transfer Learning):              NOT STARTED
Final optimization:                         NOT STARTED
Report:                                     NOT STARTED
Video presentation:                         NOT STARTED
```

## Phase 2 summary (this stage)

- Extended the Phase 1 framework into a full reusable ML
  experimentation system: 2 new modules (`cleaning.py`, `torch_utils.py`),
  substantial additions to `preprocessing.py` (target/feature
  separation, bundled preprocessor, city-specific target scaling),
  `splitting.py` (reproducible split metadata, overlap verification),
  `evaluation.py` (multi-seed aggregation), `experiment_runner.py`
  (full artifact system: metrics/config/predictions/training-log
  saving, plus a leaderboard query function).
- Built a 79-test suite (`tests/`) covering every module, using
  synthetic data for speed except where testing against a real,
  known Phase 0 finding was more meaningful (e.g. the Amherst
  4-missing-row regression test). **All 79 tests pass.**
- Built and ran a framework demonstration
  (`scripts/framework_demo.py`) proving the full pipeline works
  end-to-end on synthetic data with a trivial Linear Regression model.
  **This demo is clearly not a project result** - its outputs live
  under `results/framework_demo/` and `figures/framework_demo/`,
  never under `problem1`-`problem5`.
- Found and fixed one real bug during this phase: adding a new column
  (`parameters`) to the results-history schema without a safety check
  would have silently misaligned every future row against the old
  header. Added a schema-mismatch guard to `save_result()` (and a
  test for it) so this can't happen silently again.
- **No ML models were trained on real project data. No problem-specific
  code was written. `results/experiment_history.csv` contains exactly
  one row: the labeled framework demo (not a real result).**

## Phase 1 summary

- Built the repository foundation: `src/` (8 reusable modules),
  `configs/`, `problems/` (5 placeholder folders), `results/` (schema
  only, no data), `models/`, `figures/`, `logs/`, `data/`,
  `requirements.txt`, `.gitignore`, `README.md`, `scripts/check_setup.py`.
- All 8 `src/` modules were smoke-tested against the real dataset
  (loading, splitting, scaling, encoding, feature engineering,
  metrics, plotting, config loading, result logging).
- **No ML models were trained. No problem-specific code was written.
  `results/experiment_history.csv` and `results/leaderboard.csv`
  contained headers only — zero real rows.**

## Phase 0 summary

- Read all 19 course files (18 PDF + 1 pptx) and the real project spec
  document → `COURSE_CONTEXT.md`, `TEACHER_EXPECTATIONS.md`.
- Inspected the actual Excel dataset (9 sheets, 5 cities) → 
  `DATASET_PROFILE.md`. Found one genuine anomaly (2012-03-22 zero-output
  across 4 cities) and one spec-vs-data discrepancy (stated sampling
  window vs. actual).
- Mapped course concepts to project needs, separating taught / useful-
  but-uncovered / unnecessary → `ML_METHOD_MAP.md`.
- Built a non-executed, per-problem experiment plan → `EXPERIMENT_PLAN.md`.
- **No code had been written. No models had been trained. No metrics
  existed yet anywhere in the project.**

## Known open item

- Problem 4's SSL algorithm choice (course-taught vs. pseudo-labeling)
  is still undecided — see `TEACHER_EXPECTATIONS.md`'s final section and
  `YOUR_PROJECT_NOTES.md`.

## Context on repository history

Before this Phase 0 work began, git history showed an earlier,
now-deleted implementation attempt (a `README.md`, an `ece571/` package
with `problem1.py`–`problem5.py`, and `results/` with computed metrics
and figures), removed by a commit titled "wipe." Per instruction, this
Phase 0 work treated the project as a fresh start and did not reuse or
verify those old results. If any of that old code/results should be
recovered from git history for reference in a later phase, that's a
decision for the project owner, not something a future agent should do
unprompted.
