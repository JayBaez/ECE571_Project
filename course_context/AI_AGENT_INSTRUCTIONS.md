# Instructions for Future AI Coding Agents

This repository is a multi-stage ML course project. If you are an AI
agent (Claude Code or otherwise) picking up work here, read this file
first, then follow it.

## 1. Read these files, in this order, before writing any code

1. `course_context/COURSE_CONTEXT.md` — what was actually taught in the
   19-week course; use this to prefer course-aligned methods and
   terminology.
2. `course_context/TEACHER_EXPECTATIONS.md` — the real grading rubric
   and a REQUIRED/RECOMMENDED/OPTIONAL checklist per problem. This is
   the actual spec, not a guess.
3. `course_context/DATASET_PROFILE.md` — real, measured facts about the
   dataset, including two genuine issues to design around (the
   2012-03-22 zero-output anomaly across 4 cities, and the redundant-
   but-intentionally-needed 3yr/6yr sheet pairs).
4. `course_context/ML_METHOD_MAP.md` — which methods are taught vs.
   useful-but-uncovered vs. unnecessary, per problem.
5. `course_context/EXPERIMENT_PLAN.md` — the proposed (not yet executed)
   plan per problem.
6. `course_context/PROJECT_STATUS.md` — current progress; update this as
   you go.
7. `course_context/YOUR_PROJECT_NOTES.md` — the project owner's personal
   scratchpad; read when relevant to a decision you're about to make,
   since it may contain preferences or context not captured elsewhere.

## 2. Before modifying any existing code or file

- Inspect what's already there. Don't assume a file is empty or a
  script hasn't been run.
- Check `PROJECT_STATUS.md` for what phase/problem is actually in
  progress before starting new work.

## 3. Hard rules

- **Never fabricate results.** Any metric that appears in code output,
  `results.csv`/`results.json`, the report, or conversation must come
  from an actual executed run.
- **Preserve previous experiment results.** Don't overwrite or delete
  existing results files without asking the project owner first — add
  new rows/entries rather than replacing history, unless explicitly
  told to redo something.
- **Explain significant changes** before or as you make them — don't
  silently swap out a model, metric, split strategy, or preprocessing
  choice that a previous stage settled on.
- **Stop and report** after completing the specific stage/task you were
  asked to do. Don't cascade automatically into the next phase.
- **Ask permission before major methodological changes** — e.g.
  switching the SSL algorithm for Problem 4, changing the transfer
  learning approach for Problem 5, or altering the train/test split
  protocol.
- **Keep code understandable**, matching the project owner's stated
  background (basic Python + basic ML) — they need to explain every
  line to their professor per the course's AI-assistance policy
  (`TEACHER_EXPECTATIONS.md`, §"Vibe-Coding Policy"). Favor clarity over
  cleverness; avoid unnecessarily sophisticated architectures.
- **Prefer course-taught methods where they're a reasonable fit**
  (`ML_METHOD_MAP.md` Tier 1) before reaching for untaught techniques
  (Tier 2/3) — and when using an untaught technique, say so plainly so
  it can be flagged in the report's methods section.
- **Prioritize the grading rubric** (`TEACHER_EXPECTATIONS.md`) when
  deciding where to spend effort — "Best metric" (30 pts) and "Breadth
  of methods" (20 pts) are more than half the grade, so trying multiple
  methods per problem and iterating toward a best result matters more
  than polishing a single method.
- **No leakage.** Fit scalers/encoders/dimension-reduction on TRAIN
  only. Never shuffle time series randomly. Never let a label-derived
  quantity (e.g. `GHI`/`Clearsky GHI` for the sky-condition target)
  appear as an input feature for that target. See the "Major risks"
  section of each problem in `EXPERIMENT_PLAN.md` for problem-specific
  leakage traps.
- **Random seeds fixed and reported everywhere**; regression results
  reported as mean ± std over ≥3 seeds per the real spec.

## 4. Workflow to follow (per the project owner's stated process)

1. Inspect the current state of the relevant part of the repo.
2. Understand what's being asked and why.
3. Propose an approach.
4. Explain the proposal and trade-offs.
5. Get explicit approval before any substantive project decision
   (model family changes, split protocol changes, discarding prior
   results, expensive/long-running experiments).
6. Implement.
7. Run the code to verify it behaves correctly.
8. Report what happened — including actual numbers, not summaries that
   imply numbers exist if they don't.
9. Stop and wait for the next instruction.

## 5. When you finish a stage, update `PROJECT_STATUS.md`

Keep it short: change `NOT STARTED` → `IN PROGRESS` or `COMPLETE` for the
relevant line(s), and add a one- or two-line note under a dated entry
about what changed. Don't turn it into a full changelog — that's what
git commit history and `YOUR_PROJECT_NOTES.md` are for.
