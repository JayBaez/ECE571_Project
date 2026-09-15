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

## Problem 3 — What I Learned

- **What dimensionality reduction means:** squeezing many input
  columns (23 in my case) down into just a few numbers per row (2, 5,
  or 10), while trying to keep as much of the useful information as
  possible.
- **What PCA does:** finds the directions the data varies the most
  along, and re-describes each row using coordinates along those
  directions instead of the original columns. It's just geometry — it
  never looks at any label.
- **What explained variance means:** how much of the data's total
  spread/variation is captured by the components you kept. 100% would
  mean no information lost at all (using every original dimension).
- **What reconstruction error means:** compress a row down, then try
  to rebuild the original row from the compressed version — the
  difference between the original and the rebuilt version is the
  reconstruction error. Lower = the compression kept more.
- **What an autoencoder does:** the neural-network version of PCA — it
  learns to compress AND rebuild its own input, but because it can use
  nonlinear functions (ReLU etc.), it can often compress more
  efficiently than PCA can. I saw this directly: at every dimension I
  tested, the autoencoder reconstructed better than PCA.
- **What a latent/bottleneck representation is:** the small,
  compressed version of the data sitting in the middle of the
  autoencoder — the "2 numbers" (or 5, or 10) that everything else
  gets squeezed through.
- **Why the target can't be used when learning the representation:**
  PCA and the autoencoder are supposed to learn "what the DATA looks
  like," not "what makes the LABEL easy to predict" — using the label
  would be a different (supervised) technique entirely, and would
  make my downstream comparison meaningless (I'd be testing whether
  cheating helps, not whether compression helps).
- **Why a 2-D visualization isn't the same as a good predictive
  representation:** my 2-D plots showed the three sky-condition
  classes pretty mixed together — and sure enough, that's exactly
  where classification accuracy was worst (PCA-2: only 0.40 balanced
  accuracy, barely above random guessing at 0.33). A plot that "looks
  like it has some structure" doesn't guarantee a model can actually
  use that structure well.
- **Why dimensionality reduction can hurt performance:** in my case,
  it hurt BOTH downstream tasks, at every dimension I tried, even at
  d=10 using almost half the original columns. I traced this partly to
  Cloud Type — a category feature that turned out to compress poorly
  into a small number of continuous dimensions, even though it's one
  of the most useful features for predicting sky-condition.

## Problem 3 — Things I Need To Explain To My Professor

- Why raw features beat both PCA and the autoencoder at every
  dimension I tried, for both classification and regression — and why
  that's a legitimate, useful finding, not a failed experiment.
- Why I excluded irradiance features (GHI etc.) from the shared PCA/
  autoencoder input entirely, even though Problem 2's regression task
  has no leakage restriction on them — to keep ONE representation
  usable for both downstream tasks without reintroducing Problem 1's
  leakage risk.
- Why my Problem 3 "raw" regression baseline (RMSE 23.86) is worse
  than Problem 2's original Davis result (RMSE 15.17) — same reason as
  above, a deliberate, documented tradeoff, not an inconsistency.
- The feature ablation showing Cloud Type actually matters even after
  compression (removing it made both reconstruction AND downstream
  accuracy worse, despite technically raising explained variance).
- Why the autoencoder consistently beat PCA — its ability to learn
  nonlinear encodings, not just a lucky architecture choice.

## Problem 4 — What I Learned

- **What semi-supervised learning means:** training with a mix of a
  small amount of labeled data AND a larger pool of unlabeled data,
  hoping the unlabeled examples' input values (not their answers, which
  we don't have) still help the model somehow.
- **The difference between labeled and unlabeled data:** labeled = the
  model sees both X (features) and y (the true answer). Unlabeled = the
  model only sees X — the true y exists in my dataset (since I started
  from real labeled data and hid some labels on purpose) but the model
  is never allowed to see it.
- **What pseudo-labeling means:** train on the small labeled set, use
  that model to GUESS labels for the unlabeled data, keep only the
  guesses the model is very confident about, add those guesses to the
  training set as if they were real labels, and retrain.
- **How confidence thresholds work:** the model outputs a probability
  for each possible class; a pseudo-label only gets accepted if that
  probability is above a cutoff (I used 0.90, chosen by testing 0.80/
  0.90/0.95 on a held-out validation slice, not guessed).
- **Why pseudo-labeling can help:** more (even self-generated) labeled
  examples can, in principle, help a model learn patterns it wouldn't
  see from a tiny labeled set alone.
- **Why pseudo-labeling can also reinforce mistakes:** if the model is
  confidently WRONG about something, adding that wrong guess as a
  "real" label just teaches the model to be more confidently wrong. I
  saw a milder version of this risk directly: without a cap, my first
  pseudo-labeling round added 12,170 new labels and 98% of them were
  the same class ("Clear") — not wrong, but so lopsided it actually
  hurt performance slightly until I added a cap.
- **What label efficiency means:** how much performance you get out of
  a given amount of labeled data — a more "label-efficient" method
  needs fewer real labels to reach the same accuracy.
- **Why 10% labels is the most interesting case:** that's where
  unlabeled data theoretically has the most room to help, since the
  labeled set alone is smallest and weakest there.
- **What SSL gain means:** SSL's score minus the supervised-only
  score, at the same label fraction. Positive = unlabeled data helped.
  I got small NEGATIVE numbers at all three fractions — genuinely
  useful to know, not a failure of the experiment.
- **Why the test labels must remain hidden:** they're the one thing
  that has to stay completely untouched until the very end, so the
  final numbers actually mean something. I only used the training
  pool's hidden labels, once, for an "offline diagnostic" check
  clearly separated from anything that could influence the model.

## Problem 4 — Things I Need To Explain To My Professor

- Why SSL came out slightly WORSE than supervised-only at every label
  fraction, and why that's a legitimate, reportable finding rather
  than something to hide or "fix."
- The offline diagnostic finding that explains WHY: my pseudo-labels
  were 100% accurate (verified directly against the hidden true
  labels) but almost all for the easy "Clear" class — so they didn't
  add new information, just repeated what the model already knew.
- Why I added a cap on pseudo-labels per iteration after finding
  (empirically, by testing it) that uncapped self-training performed
  worse due to class imbalance in the added labels.
- Why Label Spreading (my optional second method) did notably worse
  than pseudo-labeling here — likely its KNN-graph approach struggling
  with a large unlabeled pool in a moderate-dimensional feature space.
- Why I reused Problem 1's exact train/test split and best model
  instead of picking new ones — keeps this a fair, apples-to-apples
  comparison to something already established.

## Problem 5 — What I Learned

- **What transfer learning means:** using knowledge a model already
  learned from one place (Davis) to help it learn faster/better
  somewhere new (Amherst), instead of starting completely from
  scratch every time.
- **Source domain vs. target domain:** the source is where the model
  learns first (Davis, lots of data) — the target is where it's
  applied and adapted (Amherst, very little data).
- **Zero-shot:** using the source-trained model directly on the
  target, with ZERO target training at all. Mine failed badly
  (RMSE=184 kW) — Davis's model just doesn't know Amherst's actual
  power scale.
- **Few-shot:** training a brand new model using only a tiny number of
  target examples (I used 10, 50, 100), with no help from the source
  city at all.
- **Fine-tuning:** starting from the ALREADY-TRAINED source model
  (not random weights) and continuing to train it a bit more, using
  just the small amount of target data — this is the actual "transfer
  learning" step.
- **Why Davis can potentially help Amherst:** they're both PV
  installations responding to weather/irradiance — the underlying
  physics (more sun → more power) should be similar even if the exact
  numbers differ.
- **Why different cities create domain shift:** Davis and Amherst have
  real climate differences (Davis is warmer and sunnier on average) —
  I measured this directly and found Wind Speed had a huge difference,
  which I flagged as possibly a sensor/data issue rather than assuming
  it was a real climate fact.
- **Why output-power scale matters so much:** Davis's plant produces
  roughly 2.7x Amherst's average power. If you don't account for this,
  a model "confidently" predicts numbers in completely the wrong range
  for the new city.
- **What negative transfer means:** when transfer learning actually
  makes things WORSE than not using it at all. I found this directly —
  my first attempt at fine-tuning (using a smaller learning rate, like
  the instructions suggested) caused severe negative transfer, because
  it couldn't adjust the output scale fast enough. Switching to a
  larger learning rate fixed it completely.
- **Why freezing layers can sometimes help:** the idea is that early
  layers learn general patterns (how weather relates to power in
  general) while later layers learn city-specific details — freezing
  the early layers protects the general knowledge while still letting
  the model adapt. In my case it made almost no difference either way,
  which is itself a useful thing to know.

## Problem 5 — Things I Need To Explain To My Professor

- Why my FIRST fine-tuning attempt caused severe negative transfer,
  and how I found and fixed the actual cause (learning rate, not a
  deeper transfer-learning problem) instead of just reporting the bad
  number or quietly changing my approach without explanation.
- Why the target-normalization ablation showed normalization DIDN'T
  help, once the real problem (learning rate) was already fixed —
  an interesting result showing the two ablations pointed to different
  root causes.
- Why the transfer gain shrinks as k grows (10 → 50 → 100) — makes
  sense, since the few-shot baseline gets progressively more capable
  on its own as it sees more real Amherst data.
- Why I flagged the Wind Speed domain-shift finding as possibly a
  data/sensor issue rather than presenting it as a confirmed climate
  fact — I don't have a way to verify this without checking the raw
  instrument documentation for both cities.
- Why the freezing ablation showed almost no difference — I explained
  this as the model not needing much adaptation in the early layers,
  not as the ablation "failing" to show anything.

## Final Project Findings

After finishing all 5 problems and a full audit, here's the overall
picture in plain language:

- **Strongest classification result:** predicting Davis's generation
  regime (Low/Medium/High power) reached 0.95 balanced accuracy — the
  strongest number anywhere in the project. Predicting sky-condition
  (Clear/Partly Cloudy/Overcast) was harder (0.77-0.82), since I
  deliberately excluded the strongest irradiance features to avoid
  leaking the label's own definition.
- **Strongest regression result:** same-city Davis power prediction,
  RMSE=15.17 kW, explaining 95.3% of the variance (R²=0.953).
- **What dimension reduction showed:** compressing my 23 features down
  to even 10 dimensions HURT both classification and regression,
  every time I tried it. The autoencoder did better than PCA at every
  size (it can learn curved, not just straight-line, compression), but
  neither one beat just using all the original features.
- **Whether SSL helped:** no — pseudo-labeling came out slightly WORSE
  than just using the labeled data alone, at every label amount I
  tried (10%/30%/50%). But I found out why: the pseudo-labels were
  100% correct, they just kept confirming the "easy" class (Clear
  skies) instead of teaching the model anything new about the harder
  classes.
- **Whether transfer learning helped:** yes, clearly — using Davis to
  help predict Amherst's power beat training on Amherst alone, at
  every amount of Amherst data I tried, with the biggest help
  (+21%) when Amherst data was scarcest (just 10 samples).
- **Most interesting result:** in Problem 5, my very first attempt at
  transfer learning actually made things WORSE, not better. I found
  the exact cause (my fine-tuning learning rate was too small to let
  the model adjust from Davis's power scale to Amherst's), fixed it,
  and transfer learning went from a failure to a clear win.
- **Most surprising result:** dimension reduction hurting performance
  was genuinely unexpected going in — usually you'd hope compression
  at least doesn't hurt. I traced part of the "why" to Cloud Type (a
  categorical feature) compressing poorly even though it's one of the
  most useful features I have.
- **Biggest limitation:** I kept every hyperparameter search small on
  purpose (3-5 options, not huge grids) — and found, repeatedly across
  three different problems, that tuning barely changed anything. A
  bigger search MIGHT find more, but I can't say for sure without
  trying it, and I chose not to given the project's own "don't
  overengineer" instructions.
- **What I should emphasize in the report:** the fact that my results
  are genuinely MIXED (transfer learning helped, dimension reduction
  hurt, SSL was a wash) is actually the strongest thing about this
  project — I didn't force every technique to "work," I tested them
  fairly and reported what actually happened.
- **What I should be prepared to explain to my professor:** the
  negative-transfer story in Problem 5 (what went wrong, how I found
  it, how I fixed it) is probably my best example of real scientific
  process, and I should be ready to walk through it step by step.

## Final Report — Things I Need To Know

- **The main project story:** I tested five ML paradigms on the same
  PV forecasting problem and reported honestly which ones helped.
  Supervised learning worked great, transfer learning clearly helped,
  but dimension reduction hurt and semi-supervised learning was a
  wash — and I can explain WHY in each case, not just report the
  numbers.
- **Strongest result:** same-city Davis regression, RMSE=15.17 kW,
  R²=0.953 — predicts 95% of the variance in Output Power from
  weather alone.
- **Weakest result:** raw-kW cross-city zero-shot transfer (Problem 2
  and the start of Problem 5) — RMSE around 125-184 kW. But I can
  explain this isn't a modeling failure, it's almost entirely a scale
  mismatch (Davis's plant is ~2.7x Amherst's average size).
- **Most interesting finding:** in Problem 5, the exact same setup
  produced severe negative transfer OR a clear win depending only on
  the fine-tuning learning rate. I found this, diagnosed it, and
  fixed it — a real example of the scientific process, not just a
  lucky good result.
- **Biggest limitation:** I kept every hyperparameter search small on
  purpose, and found repeatedly (Problems 1, 2, 4) that tuning barely
  changed results — worth mentioning honestly rather than implying I
  found the absolute best possible model.

**If the professor asks "Why did you choose this model?"** — For each
problem I compared several models on the SAME data/split/seeds and
picked whichever won on the primary metric (never picked in advance).
Several times a simpler model won (logistic regression beat every
ensemble for Davis sky-condition; untuned gradient boosting beat
tuned versions in Problem 2) — I can point to the actual numbers.

**If the professor asks "How did you prevent leakage?"** — Three
things: (1) any feature that could reconstruct or define a label was
excluded (GHI/Clearsky GHI/DHI/DNI/Solar Zenith Angle for
sky-condition — I proved this mattered by testing WITH them, accuracy
jumped to ~0.98); (2) every scaler/encoder was fit on training data
only, never test data; (3) for SSL and transfer learning, I checked
directly that hidden/target labels never entered training — I even
ran a full audit phase re-verifying this programmatically, not just
by reading my own code.

**If the professor asks "Why did you use chronological splitting?"**
— Because this is time-series data — randomly shuffling before
splitting would let the model "see the future" (train on next
Tuesday's readings, get tested on last Monday's), which would make
every metric optimistic and meaningless for real forecasting. First
80% of a city's timestamps = train, last 20% = test, always.

**If the professor asks "What did SSL actually do?"** — It didn't
help. I proved the pseudo-labels were 100% accurate (checked against
hidden true labels, after training only) but they mostly reinforced
the easy "Clear" class instead of helping with the hard classes — so
"correct but redundant," not "wrong."

**If the professor asks "What transferred between cities?"** — The
underlying weather-to-power relationship transferred well (confirmed
by a diagnostic rescaling that recovered R²=0.55-0.84 for the cross-
city case); what DIDN'T transfer automatically was the output scale,
which needed either fine-tuning (Problem 5) or explicit correction to
recover reasonable performance.

## Presentation Preparation

**30-second explanation:** I built a machine learning project that
predicts solar power output using five different ML techniques —
classification, regression, dimension reduction, semi-supervised
learning, and transfer learning — on the same weather/irradiance
dataset across five cities. My strongest result predicts power output
with 95% accuracy (R²=0.953) using just weather data. I also found
some techniques genuinely didn't help — dimension reduction hurt
performance, and semi-supervised learning barely moved the needle —
and I dug into why in both cases instead of hiding it.

**1-minute explanation:** [30-second version, plus:] The most
interesting part of the project happened in the transfer learning
problem — I tried to use a data-rich city (Davis, 6 years of data) to
help predict power for a data-poor city (Amherst, 3 years). My first
attempt actually made things WORSE than not using Davis at all, which
is called negative transfer. I dug into why, found it was because my
fine-tuning learning rate was too small to adjust the model's output
scale in time, fixed it, and transfer learning went from a failure to
a 21% improvement. I think that process — finding a real problem,
diagnosing the actual cause, and fixing it with evidence — is the
best example of what I learned doing this project.

- **Best result:** Davis same-city regression, RMSE=15.17 kW, R²=0.953.
- **Most interesting result:** The Problem 5 learning-rate story — same
  setup, severe failure (140.9 kW) or clear win (27.2 kW) depending
  only on the fine-tuning learning rate.
- **Biggest failure:** Dimension reduction hurt both downstream tasks
  at every tested dimension — traced to Cloud Type compressing poorly.
- **Biggest limitation:** Deliberately small hyperparameter searches
  throughout — repeatedly found tuning barely moved results, but never
  confirmed with a larger search.

**5 concepts I need to understand before presenting:**
1. Why chronological (not random) splitting matters for time-series data
2. Why GHI must be excluded from sky-condition classification (it defines the label)
3. What negative transfer means and how I found/fixed it in Problem 5
4. Why 100%-accurate pseudo-labels still didn't help SSL (redundancy, not error)
5. Why raw features beat every compressed representation in Problem 3

**10 questions the professor is most likely to ask:** see
`presentation/PROFESSOR_QUESTIONS.md` for the full list of 26,
organized by category (General, Dataset, Problems 1-5,
Reproducibility) — each with a short and a detailed answer.

## Report Revision — Important Note

- `report/FINAL_REPORT.md`/`.docx` were rewritten with a more detailed
  structure (explicit subsections per problem, a Title Page, a
  dedicated Key Findings section) — same facts and numbers as before,
  just reorganized to be more thorough. It's now 19 pages instead of
  14, which reflects the extra requested detail, not padding.
- **Action item for me:** the title page currently has `[YOUR NAME]`,
  `[PROFESSOR NAME]`, and `[DATE]` as placeholders — none of these are
  recorded anywhere in the project, so I need to fill them in myself
  before submitting.
- **Double-check the presentation too:** I noticed the presentation's
  title slide guessed "Prof. Puchovsky" as the professor's name — that
  name isn't actually documented anywhere in this ECE571 project (it
  looks like it was mistakenly pulled from an unrelated course). I
  should fix that slide with the correct name too, not just the report.

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
