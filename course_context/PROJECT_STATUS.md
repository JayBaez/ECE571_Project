# Project Status

**Last updated:** Phase 11 completion (final presentation) — the entire ECE571 project is now COMPLETE.
**Future AI agents: update this file as the project progresses. Keep
entries short — status + one-line note, not a log of everything done.**

```
Phase 0 — Course material & spec analysis: COMPLETE
Phase 1 — Repository construction:          COMPLETE
Phase 2 — ML framework:                     COMPLETE
Phase 3 — Dataset validation:                COMPLETE
Problem 1 (Classification):                 COMPLETE
Problem 2 (Regression):                     COMPLETE
Problem 3 (Dimension Reduction):            COMPLETE
Problem 4 (Semi-Supervised Learning):       COMPLETE
Problem 5 (Transfer Learning):              COMPLETE
Final audit:                                COMPLETE
Final optimization:                         COMPLETE
Final report:                               COMPLETE
Presentation:                               COMPLETE
```

## Phase 11 summary (final presentation)

- Created `presentation/FINAL_PRESENTATION.pptx` (15 slides, ~11-14 min)
  via a purpose-built `pptxgenjs` script (`presentation/
  build_presentation.js`) — custom solar/energy color palette (navy +
  solar gold + teal/terracotta accents), 5 real project figures
  embedded directly (never recreated), 1 native comparison chart built
  for content with no existing figure (the Problem 5 learning-rate
  story), and a native cross-problem summary table.
- **Every number verified against `results/FINAL_EXPERIMENT_TABLE.csv`
  and the per-problem detail files twice** — once before writing slide
  content, once by extracting every number that actually appears on
  the built slides (via `markitdown`) and re-checking each against
  source data. Zero discrepancies found either time.
- Found and fixed two real layout bugs during visual QA (a translucent
  shape rendering as flat gray; a navy insight box overlapping its
  neighboring figure card on two slides) and one real content bug (a
  numbered list losing its numbers) — all caught by actually rendering
  and inspecting the slides, not assumed correct.
- `scripts/office/validate.py` reports all checks passed; `markitdown`
  content scan found no leftover placeholder text.
- Created `presentation/PROFESSOR_QUESTIONS.md` (26 questions, 9
  categories), `PRESENTATION_CHEAT_SHEET.md`, and `PRESENTATION_SCRIPT.md`
  (the latter generated programmatically from the pptx's own embedded
  speaker notes, guaranteeing the two never drift out of sync).
- Speaker notes trimmed once after an initial word-count/pace check
  ran over the 8-12 minute target; final estimate 11.3-14.0 minutes
  depending on speaking pace — reported honestly rather than claimed
  as precisely in-range.
- **No new experiments, no changed results, no fabricated numbers** —
  this phase was presentation design and verification only.

## Phase 10 summary (final report)

- Created `report/FINAL_REPORT.md` (the full 14-section course report)
  and `report/FINAL_REPORT.docx` (converted via a purpose-built
  markdown-to-docx script, `report/convert_report.js`, rather than
  manually retyping content — chosen specifically to avoid
  transcription risk on a report full of precise verified numbers).
- **Every number in the report was individually verified against
  saved results** (`results/FINAL_EXPERIMENT_TABLE.csv`, each
  problem's own results CSV, and the per-class/error-analysis/
  domain-shift detail files) — cross-checked once before writing and
  again after all editing passes, including a final spot-check
  focused on the most heavily-edited sections.
- 10 figures and 5 tables included, spanning all 5 problems plus one
  dataset-motivation figure; every figure has a numbered caption.
- Report length: 14 pages total (~13 pages of main content, excluding
  References) — above the 8-12 page target despite two rounds of
  prose trimming and layout tightening; the shortfall reflects the
  genuinely required content (5 full problem write-ups, 10 figures,
  5 tables) more than padding, and is reported honestly rather than
  hidden.
- Caught and fixed two real bugs in the conversion script during
  development: paragraph-per-source-line bloat (26 pages before the
  fix) and numbered-list items silently losing their numbers.
- AI assistance disclosure included (Section 15), accurately
  describing the project's actual AI-assisted workflow.
- **No new experiments, no changed results, no fabricated numbers or
  citations** — this phase was writing and verification only.

## Phase 9 summary (final audit)

- Created `course_context/FINAL_AUDIT.md` — a full leakage audit (all
  5 problems, SAFE), time-series split audit (SAFE), feature audit,
  metric audit, and seed/reproducibility audit, all performed by
  direct code inspection and live verification, not assumption.
- **Reran 5 representative experiments (one per problem) and matched
  every saved result exactly** — the strongest reproducibility
  evidence the project could offer (`FINAL_AUDIT.md`, Section 10).
- Determined, after auditing each problem's actual results against
  the grading rubric, that **none of the 5 problems needed new
  optimization experiments** — every problem's existing results were
  already strong, honestly investigated, and (where something looked
  weak, e.g. Problem 4's SSL gain or Problem 5's original negative
  transfer) already resolved with evidence during that problem's own
  phase. No new model training was performed this phase.
- Built the cross-problem deliverables: `results/FINAL_EXPERIMENT_TABLE.csv`
  / `FINAL_RESULTS.csv` / `FINAL_RESULTS.json` (30-row master table),
  `results/BEST_RESULTS.md`, `course_context/CLAIMS_TO_AVOID.md`,
  `course_context/PROJECT_STORY.md`, `course_context/
  FINAL_FIGURE_INVENTORY.md`, `course_context/FINAL_TABLE_INVENTORY.md`,
  `course_context/REPRODUCIBILITY_CHECKLIST.md`.
- Rewrote `README.md` (was still Phase-2-vintage, describing an empty
  framework with "no ML problems solved yet") to reflect all 5
  complete problems, and removed one confirmed-unused dependency
  (`tqdm`) from `requirements.txt`.
- **No fabricated numbers, no methodology changes** — every claim in
  `FINAL_AUDIT.md` and `BEST_RESULTS.md` traces to an existing result
  or a fresh rerun performed live this phase.

## Problem 5 summary

- Built `problems/problem5_transfer_learning/` (models.py,
  run_experiments.py), reusing Problem 2's exact Davis/Amherst dataset
  builders (same splits, same feature set, same leak-free zero-shot
  pattern) rather than rebuilding any of it.
- Ran zero-shot, few-shot (k=10/50/100), and transfer (k=10/50/100)
  Davis→Amherst experiments, 3 seeds each, plus freezing and target-
  normalization ablations — 27 real experiment rows in `results/
  problem5/problem5_results.csv`.
- **Found and fixed a genuine negative-transfer bug during
  development**: the initially-suggested smaller fine-tuning learning
  rate (1e-4) caused severe negative transfer (RMSE 140.9 vs. few-shot's
  38.2 at k=10) because it couldn't adapt the output scale from
  Davis's ~164kW calibration to Amherst's ~64kW in time. Traced to
  this specific cause, fixed by matching the pretraining LR (1e-3),
  and reported the whole investigation transparently rather than
  silently using the corrected number.
- **Finding: transfer gain was strongest exactly where labels are
  scarcest** — +21.4% RMSE improvement at k=10, shrinking to +0.9% at
  k=100 — the textbook transfer-learning shape, from real executed
  experiments.
- Normalization ablation showed explicit target normalization added
  no benefit once the LR was fixed; freezing ablation showed almost no
  difference between full and partial fine-tuning — both genuine,
  documented findings, not forced to show an effect.
- Domain-shift analysis flagged a striking Wind Speed anomaly (SMD=2.6)
  as possibly a sensor/data artifact rather than asserting it as
  confirmed climate fact.
- Saved 7 models (Davis pretrained + 3 few-shot + 3 transfer), all
  verified loadable.
- **No fabricated numbers** — every result in
  `course_context/PROBLEM5_REPORT.md` traces to an actual run.

## Problem 4 summary

- Built `problems/problem4_semi_supervised/` (pseudo_labeling.py,
  run_experiments.py), directly reusing Problem 1's exact Davis
  sky-condition dataset build (same chronological split, same
  leakage-safe features, same empirically-best model — Logistic
  Regression with balanced weight) rather than re-implementing any of
  it.
- Ran pseudo-labeling/self-training (primary SSL method) and Label
  Spreading (optional second method) at label fractions 10%/30%/50%,
  3 seeds each, against a supervised-only baseline on the SAME labeled
  subsets — 27 real experiment rows in `results/problem4/
  problem4_results.csv`.
- **Finding: SSL gain was slightly negative at every label fraction**
  (reported honestly, not adjusted). An offline diagnostic (hidden
  labels used only for this post-hoc check, never during training)
  explained why: pseudo-labels were 100% accurate but concentrated on
  the already-easy "Clear" class, adding redundant confirmation
  instead of new information about the harder classes.
- Found and used a real, evidence-based safeguard: uncapped
  self-training collapsed toward one class and performed worse than a
  capped version (tested directly, not assumed) — 1,000
  pseudo-labels/iteration was adopted based on that comparison.
- Fixed a `numpy.trapz`→`trapezoid` API break and added explicit
  `random_state` to the model factory for project-wide consistency.
- Saved 6 models (supervised + SSL at each fraction), all verified
  loadable.
- **No fabricated numbers** — every result in
  `course_context/PROBLEM4_REPORT.md` traces to an actual run.

## Problem 3 summary

- Built `problems/problem3_dimension_reduction/` (features.py,
  reduction.py, run_experiments.py), reusing the Phase 2 framework and
  Problem 1's leakage-safe feature set as ONE unified input for both
  downstream tasks (documented tradeoff: Problem 3's "raw" regression
  baseline is weaker than Problem 2's headline result, since
  irradiance features are deliberately excluded here — see
  `PROBLEM3_REPORT.md`, Section 4).
- Ran PCA (d=2/5/10, plus extra values for a clear elbow curve) and a
  small autoencoder (d=2/5/10, 3 seeds each) on Davis, then compared
  raw vs. every reduced representation on both Problem 1's
  sky-condition classifier and Problem 2's regressor — 54 real
  experiment rows in `results/problem3/problem3_results.csv`, central
  comparison table in `problem3_comparison_table.csv`.
- **Finding: dimension reduction hurt both downstream tasks at every
  tested dimension** (reported honestly, per the instructions'
  explicit allowance for this outcome) — traced partly to Cloud Type
  compressing poorly, confirmed via a small feature ablation. The
  autoencoder consistently outperformed PCA at every dimension in both
  reconstruction quality and downstream performance.
- A sandbox filesystem reset (disk space exhaustion) occurred mid-
  phase; `problems/problem3_dimension_reduction/` source was rebuilt
  from scratch and re-run — results matched exactly (bit-for-bit
  identical), confirming the rebuild was faithful.
- Saved all 6 fitted models (3 PCA, 3 autoencoder), verified loadable.
- **No fabricated numbers** — every result in
  `course_context/PROBLEM3_REPORT.md` traces to an actual run.

## Problem 2 summary

- Built `problems/problem2_regression/` (features.py, models.py,
  sequence.py, run_experiments.py) reusing the Phase 2 framework
  throughout, including target scaling functions built in Phase 2
  specifically for this kind of need (`fit_target_scaler()` etc.).
  Added R² to `src/evaluation.py`'s `regression_metrics()` (additive,
  non-breaking - existing tests unaffected, one new test added).
- Ran same-city (Davis, Amherst), cross-city zero-shot (Davis→Huron/
  Santa Barbara/La Jolla), 3yr-vs-6yr (Davis), and K=12 sequence
  (GRU) experiments — 108 real experiment rows in `results/problem2/
  problem2_results.csv`.
- **Cross-city zero-shot showed severe raw-scale failure (R² as low as
  -72), reported honestly rather than hidden** — then a labeled
  diagnostic proved the failure was almost entirely a scale mismatch,
  not a pattern-recognition failure (R² 0.55-0.84 once rescaled).
- **Caught and fixed a real methodology bug during development:** the
  first GRU hyperparameter search implementation scored candidates
  against the real test set, violating the project's own rule. Fixed
  before any tuned result was recorded.
- Saved best models for Davis, Amherst, cross-city, and the sequence
  model (`results/problem2/models/`), all verified loadable.
- **No fabricated numbers** — every result in
  `course_context/PROBLEM2_REPORT.md` traces to an actual run.

## Problem 1 summary

- Built `problems/problem1_classification/` (targets.py, features.py,
  models.py, run_experiments.py) reusing the Phase 2 framework
  throughout — no framework code needed rebuilding, only one small,
  well-justified extension (`torch_utils.make_dataloader()` gained a
  `y_dtype` parameter to support classification's Long-typed labels,
  alongside regression's existing Float default).
- Ran the full model progression (majority baseline → logistic
  regression → decision tree → random forest → gradient boosting →
  MLP) for both tasks (sky-condition, generation-regime), both cities
  (Davis, Amherst), at 3 seeds each — 198 real experiment rows in
  `results/problem1/problem1_results.csv`.
- Ran a small hyperparameter search (chronological inner validation),
  a class-weighting comparison, and a feature ablation study —
  **found and confirmed empirically** that the Solar Zenith Angle/
  DHI/DNI leakage risk flagged in Phase 3 is real: including those
  features pushed sky-condition accuracy from ~0.74–0.81 to ~0.98.
- Saved confusion matrices, per-class metrics, error analysis, and
  permutation-importance figures for the actual best model of each
  (task, city) combo — plus the models/preprocessors themselves
  (`results/problem1/models/`), all verified loadable.
- **No fabricated numbers** — every result in
  `course_context/PROBLEM1_REPORT.md` traces to an actual run.

## Phase 3 summary

- Built `scripts/run_eda.py`: a reusable, re-runnable EDA script
  covering all 34 requested analysis areas — sheet/column/dtype
  verification, missing values, duplicates, timestamp validation,
  Output Power distributions, irradiance relationships, Clear-Sky
  Index, Cloud Type, weather features, correlations, target-scale
  analysis, class-imbalance preview, and autocorrelation. Never
  modifies the raw Excel file (verified: file checksum unchanged
  after every run).
- Saved 19 tables to `results/eda/*.csv` and 11 figures to
  `figures/eda/*.png`.
- **Found one previously-undocumented, genuine data-quality issue:**
  `Relative Humidity`/`Wind Direction` appear swapped and/or
  mis-unitted for all of Davis 2013 and all of Huron 2012 (~14% of
  each city's data) — see `course_context/EDA_REPORT.md` and
  `DATASET_PROFILE.md`.
- Found and fixed a real internal contradiction in
  `DATASET_PROFILE.md` left over from Phase 0/1: its "Recommended
  defaults" section still said to skip the redundant `'14-'16` sheets,
  directly contradicting the correction earlier in the same file
  (which says both sheets are needed for the spec's 3yr-vs-6yr
  ablation). Fixed and now internally consistent.
- Created `course_context/LEAKAGE_MAP.md` (per-problem feature safety
  rules) and `course_context/EDA_REPORT.md` (full technical findings).
- Updated `DATASET_PROFILE.md` throughout with explicit
  [SPECIFIED]/[VERIFIED] labels per section, per this phase's
  instruction not to let an assumption pass as a fact.
- **No ML models were trained. No problem-specific code was written.
  No modeling decisions (final classifier, regressor, SSL algorithm,
  transfer strategy) were locked in.**

## Phase 2 summary

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

## Known open items

None currently — the three items previously listed here (Problem 4's
SSL algorithm, the sky-condition classifier's Solar Zenith Angle/DHI/
DNI feature question, and the Davis-2013/Huron-2012 Relative
Humidity/Wind Direction finding) were all resolved based on the
project owner's direct guidance. See `TEACHER_EXPECTATIONS.md`,
`EXPERIMENT_PLAN.md`, `LEAKAGE_MAP.md`, and `YOUR_PROJECT_NOTES.md`
for the resolutions.

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
