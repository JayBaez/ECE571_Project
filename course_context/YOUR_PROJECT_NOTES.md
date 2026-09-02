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

## What I Learned About The Dataset (Phase 3 EDA)

- The dataset really is clean in most ways: 0 duplicate rows anywhere,
  only 4 missing values total (Amherst, one date), all columns match
  the spec exactly. The two real problems are narrow and specific
  (below), not pervasive.
- GHI is clearly the strongest single predictor of Output Power, but
  only when checked **within one city at a time** — pooled across all
  five cities the correlation looks weak (0.43) purely because of the
  scale differences, not because the relationship is actually weak
  (per-city it's 0.75-0.97). This tripped me up when I first saw the
  pooled number — worth remembering when looking at any pooled stat.
- Full technical write-up: `course_context/EDA_REPORT.md`.

## Important Graphs

- `figures/eda/output_power_by_city_boxplot.png` — the city scale
  difference in one picture (Davis towers over the other four).
- `figures/eda/correlation_heatmap.png` — good one to screenshot for
  the report's "why GHI matters" discussion.
- `figures/eda/clear_sky_index_distribution.png` — shows the sky-
  condition thresholds (0.85, 0.4) aren't cutting through a weird
  spike in the distribution — reassuring that they're reasonable.
- `figures/eda/temporal_sampling.png` — simplest way to explain the
  "why only 11 readings a day" sampling design to the professor.

## Important Data Problems

1. **The 4 missing Amherst rows and the 2012-03-22 four-city zero-
   output anomaly** — already known from Phase 0, re-confirmed here.
2. **Relative Humidity / Wind Direction, Davis 2013 and Huron 2012**
   (~14% of each city's data) — investigated, and **resolved: not an
   error.** RH can legitimately exceed 100%; the small Wind Direction
   values for those two years are valid, likely just a different unit
   (radians vs. degrees). Keeping both columns; will normalize Wind
   Direction's units consistently if it's used as a feature.

## Things That Could Cause Leakage

Full checklist: `course_context/LEAKAGE_MAP.md`. The one I'm most
likely to forget: `DHI` + `DNI` + `Solar Zenith Angle` together can
basically reconstruct `GHI`, so even though the spec only says to
exclude `GHI`/`Clearsky GHI` from the sky-condition classifier, using
all three of those "safe" columns together is a backdoor around that
rule. **Resolved:** excluding all three from the primary sky-condition
model; running a labeled secondary ablation with them included to show
the leakage effect explicitly in the report.

## Things I Need to Understand Before Problem 1

- How to normalize `Wind Direction`'s units consistently across all
  years if I end up using it as a feature (Davis 2013/Huron 2012 are
  on a different scale than the rest — see above).
- How to cleanly report the GHI/DHI/DNI/Zenith leakage ablation as a
  secondary, clearly-labeled result without it being confused for the
  headline classifier result.

## Things I Might Ask The Professor

(See also "Questions for My Professor" above for the Phase 0 list.)

- (No new open question from Phase 3 — the Relative Humidity/Wind
  Direction question resolved without needing to ask.)

## Decisions Made (resolved, previously open)

1. **Davis-2013/Huron-2012 Relative Humidity/Wind Direction:** not an
   error — keeping both columns, normalize Wind Direction's units if
   used as a feature.
2. **Sky-condition classifier:** exclude Solar Zenith Angle/DHI/DNI
   from the primary model; run a secondary ablation with them included
   to demonstrate the leakage effect.
3. **Problem 4's SSL algorithm:** pseudo-labeling/self-training as the
   primary method; graph-based label propagation
   (`sklearn.semi_supervised.LabelPropagation`/`LabelSpreading`) added
   for breadth.

## Decisions Still Open

1. Whether to actually run the optional 3-year vs. 6-year ablation for
   Problem 2, or skip it if time is tight.

## Problem 1 — What I Learned

- **What "classification" means here:** instead of predicting a number
  (that's Problem 2), I'm predicting which of 3 labeled buckets a row
  falls into. Two separate targets: sky-condition (Clear/Partly
  Cloudy/Overcast) and generation-regime (Low/Medium/High power
  output).
- **Why GHI can't be used for sky-condition:** the label itself IS
  `GHI / Clearsky GHI` thresholded into 3 buckets. Using GHI as a
  feature would be like giving the model the answer key. I actually
  proved how big a deal this is: adding back the "risky" columns
  (DHI/DNI/Solar Zenith Angle, which can reconstruct GHI) made
  accuracy jump from ~0.74-0.81 to ~0.98 — a huge, very concrete
  demonstration of leakage, not just a theoretical worry.
- **Why Cloud Type is categorical:** it's a code (0=Clear, 1=Probably
  Clear, etc.), not a real number — treating "8" as "twice 4" would be
  meaningless. One-hot encoding turns it into several yes/no columns
  instead.
- **Why accuracy alone isn't enough:** Clear-sky rows dominate the
  data (~72% overall). A model that always guesses "Clear" gets ~72%
  accuracy while being useless. Balanced accuracy (average per-class
  recall) doesn't get fooled by this — that's exactly why my "majority
  baseline" always scores 0.333 (chance level for 3 balanced classes)
  instead of looking artificially good.
- **Why chronological splitting matters:** the test set has to be
  data the model genuinely hasn't seen yet, in time. For generation-
  regime specifically, I also learned the tercile *boundaries*
  themselves have to be computed from training data only — otherwise
  I'd be leaking test-period statistics into how the labels are even
  defined.
- **A finding I didn't expect:** no single model type won everywhere.
  Logistic regression beat Random Forest and the MLP for Davis
  sky-condition. Simpler isn't always worse.

## Problem 1 — Things I Need To Explain To My Professor

- Why I excluded DHI/DNI/Solar Zenith Angle from the primary
  sky-condition model even though the spec only explicitly forbids
  GHI/Clearsky GHI — and the ablation number (0.74→0.98) that proves
  this wasn't paranoia.
- Why generation-regime terciles are computed per-city and only on
  training data, and what happened when I checked the test-set
  distribution afterward (Amherst skewed to 38/35/27, not 33/33/33).
- Why I dropped (not interpolated) the 4 missing Amherst rows for the
  generation-regime task specifically — interpolating a target and
  training on it as if it were real would be fabricating a label.
- Why hyperparameter tuning barely changed anything here (all deltas
  within ±0.008) — worth being upfront about rather than only
  reporting the tuned numbers.
- Why I used one consistent model (Random Forest) for the feature
  ablation study instead of each combo's individual best model.

## Problem 2 — What I Learned

- **What regression means here:** instead of picking a category
  (Problem 1), I'm predicting an actual number — Output Power in kW.
- **RMSE:** root-mean-square error, in the same units as the target
  (kW), so "RMSE=15" literally means "typically off by around 15 kW,"
  but big misses count extra (squared before averaging), so it's more
  sensitive to occasional large errors than MAE.
- **MAE:** mean absolute error — the plain average size of a miss,
  also in kW, easier to explain to a non-technical audience, less
  swayed by outliers than RMSE.
- **nRMSE:** RMSE divided by the target's range (I kept using the same
  "divide by range" convention I set up back in Phase 2, documented
  clearly so it's consistent everywhere) — this is what makes RMSE
  numbers comparable ACROSS cities with very different scales.
- **Why Output Power's scale differs by city:** Davis's plant is much
  bigger than Huron/Santa Barbara/La Jolla's — I saw this cause a real
  problem when I tried zero-shot transfer (see below).
- **Why chronological splitting matters:** same reason as Problem 1 —
  the model can't be tested on data from before it "learned" to avoid
  cheating by seeing the future.
- **What zero-shot means:** training only on Davis, then testing
  directly on a totally different city with ZERO training on that
  city's own data. I saw this fail dramatically when I just used raw
  kW predictions (R² as bad as -72!), but once I checked whether it
  was just a SCALE problem (diagnostic only, not a real fix), it
  turned out the model actually understood the target cities' weather
  patterns fairly well — it was just predicting in the wrong "units"
  for that city's plant size.
- **Why sequence models are useful:** they use the recent past (not
  just the current moment) to predict what happens next — genuinely
  useful for real forecasting, where you don't get to see the future
  weather at the exact moment you're predicting.
- **What a 12-step window means:** the previous 12 readings (30 min
  apart = 6 hours of history) get fed in together to predict the next
  reading.
- **Why lagged Output Power can be legitimate for forecasting:** inside
  a 12-step window, every value is from BEFORE the thing being
  predicted — so including past Output Power readings as inputs isn't
  cheating, it's exactly what a real forecaster would have access to.
- **Why target normalization is tricky across cities:** I had to
  separate two totally different reasons to scale a target: (1)
  helping a neural network train faster/more stably [fine, just uses
  that one city's own data], vs. (2) rescaling zero-shot predictions
  using the TARGET city's own statistics [not fine for a real
  zero-shot claim, since a real deployment wouldn't have that data
  yet]. I built the sequence model's target scaling for reason (1) and
  used a clearly-labeled "diagnostic" for reason (2), which stayed out
  of the actual reported zero-shot number.

## Problem 2 — Things I Need To Explain To My Professor

- Why the cross-city zero-shot result looks terrible in raw RMSE
  (~125 kW) but is actually informative — the diagnostic
  scale-correction shows the model's underlying pattern-matching is
  good (R² 0.55–0.84), it's just outputting the wrong scale, which is
  itself an important, honest finding.
- Why the GRU sequence model "losing" to the best non-sequence model
  isn't really a fair fight — the non-sequence model gets to see the
  exact same-moment weather, while the GRU only gets the past. GRU vs.
  persistence (its real competitor) is where it clearly wins.
- Why I reduced Random Forest's default tree count partway through
  (200→100) — a documented, timed efficiency decision, not a quality
  compromise (RMSE barely changed).
- A mistake I caught myself: my first version of the GRU hyperparameter
  search accidentally scored candidates against the real test set. I
  found and fixed this before recording any tuned result — worth
  explaining as an example of catching my own methodology error.
- The 3-year vs. 6-year result (3yr looked better) is confounded by
  the two sheets not sharing a test period — I'm reporting it honestly
  but flagging that it doesn't cleanly prove "less data is better."

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
