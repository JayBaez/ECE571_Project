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

## Repository Architecture (added end of Phase 1)

Beginner-friendly explanation of how the `src/` framework fits
together — the pipeline flows in one direction, left to right:

```
Excel file (course/*.xlsx)
    ↓  src/data_loader.py          — just reads the sheet, nothing else
Raw DataFrame
    ↓  src/preprocessing.py        — handle missing values
    ↓  src/feature_engineering.py  — add Clear-Sky Index, time features, lags
Cleaned + featured DataFrame
    ↓  src/splitting.py            — chronological / cross-city / random-subset
Train DataFrame, Test DataFrame
    ↓  src/preprocessing.py again  — fit_scaler/fit_encoder on TRAIN ONLY,
                                      then apply to both train and test
Scaled, encoded train/test data
    ↓  (a model — not built yet, this is Problem 1-5's job)
Predictions
    ↓  src/evaluation.py           — turn predictions into metrics (RMSE, F1, ...)
    ↓  src/visualization.py        — turn predictions/metrics into saved plots
    ↓  src/experiment_runner.py    — save_result() appends one row to
                                      results/experiment_history.csv
```

**Why it's split into 8 small files instead of one big one:** each
file does one job (loading data, OR cleaning it, OR splitting it, OR
scoring it, OR plotting it). When I'm building Problem 2's code later
and something's wrong with a metric, I know to look in
`evaluation.py`, not hunt through a 2,000-line file.

**The one rule that matters most:** anything that "learns" from data
(a scaler, an encoder, a PCA, a model) gets `fit()` on training data
only, then `apply()`/`transform()`/`predict()` on both train and test.
Never fit on the combined or test-only data — that's the #1 leakage
mistake called out throughout `EXPERIMENT_PLAN.md`.

**What's still missing (on purpose):** there's no actual model code
anywhere yet. `src/` only has the pipeline plumbing around a model —
Problems 1-5 (in `problems/problemN_*/`) will each add their own
model-specific code that plugs into this pipeline, rather than each
problem reinventing data loading/splitting/evaluation from scratch.

## Framework Architecture (added end of Phase 2)

Phase 2 filled in the "plumbing around a model" mentioned above.
Beginner-friendly summary of what's new:

**Two new files:**
- `src/cleaning.py` — detects and reports missing values, duplicate
  rows, and physically-suspicious values (e.g. negative irradiance),
  and prints a report like "Rows before: 180 / Missing Output Power: 4
  / Rows remaining: 180" so cleaning never happens silently.
- `src/torch_utils.py` — a generic training loop for PyTorch models
  (any of them — MLP, LSTM, GRU, autoencoder, whatever gets built in
  Problems 1-5). Handles batching, early stopping (stop once
  validation loss stops improving), and checkpointing (save the best
  version of the model seen so far). No actual neural network is
  defined yet — this file just knows how to *train* one once it
  exists.

**How an experiment will eventually be configured and saved:**
```
configs/my_experiment.yaml   (problem, model, seed, feature toggles, split type)
        ↓  experiment_runner.load_config()
config dict
        ↓  (load data, clean, engineer features, split, preprocess, train — as before)
metrics dict
        ↓  experiment_runner.create_experiment_dir()
results/problemN/EXPERIMENT_ID/
    metrics.json        ← the actual numbers
    config.yaml          ← exactly what was run
    predictions.csv       ← every prediction, for later re-plotting
    training_log.csv       ← per-epoch loss, if it was a neural net
        ↓  experiment_runner.save_result()
results/experiment_history.csv   ← one summary row added, others preserved
```

**How I'll find "what's the best result so far":** rather than a
separate `leaderboard.csv` file that could quietly get out of sync,
`experiment_runner.get_leaderboard(metric="rmse")` reads the always-
current `experiment_history.csv` and sorts it correctly — it knows
RMSE should sort low-to-high but balanced accuracy should sort
high-to-low, so I don't have to remember that myself.

**Multi-seed reporting:** `evaluation.aggregate_across_seeds()` takes
a list of metrics dicts (one per seed) and returns mean/std for each
metric, keeping every individual seed's number too — this is what
produces the "mean ± std over 3 seeds" the spec requires for Problem 2.

**City-scale differences (Davis vs. everyone else):** if a cross-city
or transfer experiment needs to combine cities with very different
Output Power scales, `preprocessing.fit_target_scaler()` /
`apply_target_scaler()` / `inverse_transform_target()` standardize the
target for training and convert predictions back to real kW before
computing RMSE/MAE — I only need this for cross-city work, not
same-city experiments.

**The framework demo:** `scripts/framework_demo.py` runs the *entire*
pipeline above on a small made-up dataset with a plain Linear
Regression model, just to prove all the pieces fit together. Its
output lives in `results/framework_demo/` and `figures/framework_demo/`
— never mix these up with real Problem 1-5 results.

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
