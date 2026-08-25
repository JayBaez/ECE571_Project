# My Project Notes

A living scratchpad — edit this freely as the project progresses. The AI
agents will read it when relevant but won't rewrite it wholesale.

## Things I Need to Understand

- What Clear-Sky Index `k = GHI/Clearsky GHI` actually means physically
  (it's basically "how much of the theoretical clear-sky sunlight
  actually arrived") — this defines the Problem 1 sky-condition label.
- Why nRMSE matters: Davis's plant is ~3–3.5x the capacity of Huron/
  Santa Barbara/La Jolla, so a raw RMSE of "10 kW" means very different
  things in different cities. Need to be able to explain this simply.
- The difference between zero-shot and few-shot in Problem 5 (zero-shot
  = no target data at all during training; few-shot = a small handful
  of target samples).

## Important ML Concepts

- Course explicitly separates classical ML (Weeks 3–11: linear models,
  kNN, SVM, trees/forests, Naive Bayes, k-means/GMM, PCA, regularization)
  from deep learning (Weeks 13–15: MLP, CNN, RNN/LSTM/GRU).
- Semi-supervised methods actually taught: Transductive SVM, Co-training,
  graph-based label propagation. Pseudo-labeling (train → predict on
  unlabeled → add confident predictions back to training set) is common
  in practice but **not** one of the three taught methods — still
  undecided which way to go for Problem 4 (see Questions below).
- Transfer learning is not taught anywhere in the course — Problem 5
  will lean on general practice (fine-tuning a source-trained model on a
  little target data), not lecture content.

## Project Decisions

- (none made yet — this section fills in as we go)

## Ideas I Might Try Later

- 3-year vs. 6-year ablation for Problem 2 (spec §3.1, optional) — could
  be an easy "extra breadth" point since the sheets already exist for it.
- TCN or a small Transformer for the Problem 2 sequence task — spec
  explicitly suggests these alongside LSTM/GRU, but they're not taught,
  so only worth it if LSTM/GRU results are solid first.
- Co-training for Problem 4 using irradiance-family features vs.
  meteorological-family features as the two "views" — matches the
  taught algorithm's assumption of two reasonably independent feature
  groups.

## Questions for My Professor

- Does Problem 4's semi-supervised method need to be one of the three
  taught in Week12 (Transductive SVM / Co-training / graph-based), or is
  a simpler method like pseudo-labeling acceptable?
- The written spec describes the daily sampling window as "10:00 to
  ~14:30," but the actual data runs 10:00–15:00 (11 samples/day, not
  10). Worth flagging in case it affects grading expectations, or just
  a minor wording slip in the handout.
- Is Amherst the intended/expected target city for Problem 5, or is any
  3-year-only city acceptable? (Amherst is currently the *only* 3-year
  city in the data besides being the odd one out — worth confirming
  there isn't a different intended pairing.)

## Things I Need to Explain in the Presentation

- Every method used, including anything not covered in lecture
  (autoencoder for P3, transfer learning approach for P5, TCN/Transformer
  if used for P2) — per the course's AI-assistance policy, I need to be
  able to explain every line/method regardless of source.
- The 2012-03-22 data anomaly (all four CA/inland cities show zero
  Output Power despite normal irradiance that day) and how I handled it.
- The AI-assistance disclosure paragraph required by §8 of the spec —
  keep a running list below of what got AI help so this is easy to
  write later.

## Things Claude Recommended

- Keep the `'14-'16` and `'11-'16` sheets both in the loader (needed for
  the optional 3yr-vs-6yr ablation) rather than treating the shorter
  sheets as pure duplicates to discard.
- Exclude `GHI`/`Clearsky GHI` from features when predicting the
  sky-condition label (explicit spec rule); also consider ablating away
  `DHI`+`DNI`+`Solar Zenith Angle` together for that same target, since
  those three can approximately reconstruct `GHI`.
- Start with Problem 1 first (simplest full pipeline), then 2 → 3 → 4 →
  5, since 3 depends on 1+2 already existing.
- Initially use a reasonable model progression (baseline → classical →
  stronger classical → deep) per problem rather than trying an
  excessive number of models — breadth can expand later once the core
  pipeline works and if time/results justify going further.

## Things I Changed Personally

- (fill in as you make your own calls that diverge from a recommendation)

## Potential Improvements

- (ideas for later — extra models, extra cities, extra ablations)

## Things Not To Forget

- The 4 missing `Output Power` rows in Amherst (2020-07-06, 10:00–11:30)
  — decide drop vs. interpolate and note the choice in the report.
- The 2012-03-22 anomaly across Davis/Huron/Santa Barbara/La Jolla —
  decide whether to exclude it from training, and say why in the report.
- Fix and record random seeds from the very first experiment — don't
  add this retroactively.
- Keep the AI-assistance disclosure paragraph updated as work happens,
  not written from memory at the end.

## Notes About the Grading Rubric

- 100 pts total: Correctness & reproducibility (20) · Breadth of methods
  (20) · Best metric/leaderboard (30) · Analysis & insight (20) · Report
  + code + video clarity (10).
- **Best metric is the single biggest component (30 pts)** — and
  grading explicitly rewards trying multiple methods and reporting the
  best one, not just the first thing that worked.
- "Breadth" (20 pts) specifically wants multiple classical methods *and*
  at least one deep method, per problem, with an ablation table — not
  just variety for its own sake.
- "Analysis & insight" (20 pts) means every problem needs a short
  written explanation of *why* results came out the way they did, not
  just a metrics table.
- Deadline: 11:59:59 PM, 12/09/2026. Late penalty: −20 pts/day. No
  extensions per the spec — build in buffer time before the deadline.
