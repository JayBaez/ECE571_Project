# Project Status

**Last updated:** Phase 0 completion (course/spec/data analysis stage).
**Future AI agents: update this file as the project progresses. Keep
entries short — status + one-line note, not a log of everything done.**

```
Phase 0 — Course material & spec analysis: COMPLETE
Phase 1 — Repository construction:          NOT STARTED
Phase 2 — ML framework:                     NOT STARTED
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

## Phase 0 summary (this stage)

- Read all 19 course files (18 PDF + 1 pptx) and the real project spec
  document → `COURSE_CONTEXT.md`, `TEACHER_EXPECTATIONS.md`.
- Inspected the actual Excel dataset (9 sheets, 5 cities) → 
  `DATASET_PROFILE.md`. Found one genuine anomaly (2012-03-22 zero-output
  across 4 cities) and one spec-vs-data discrepancy (stated sampling
  window vs. actual).
- Mapped course concepts to project needs, separating taught / useful-
  but-uncovered / unnecessary → `ML_METHOD_MAP.md`.
- Built a non-executed, per-problem experiment plan → `EXPERIMENT_PLAN.md`.
- **No code has been written. No models have been trained. No metrics
  exist yet anywhere in this project.**

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
