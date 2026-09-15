# Presentation Cheat Sheet

Read this 10 minutes before presenting.

## PROJECT IN ONE SENTENCE
I tested five machine learning paradigms — classification, regression,
dimension reduction, semi-supervised learning, and transfer learning —
on the same solar power dataset, and reported honestly which ones
actually helped.

## DATASET IN ONE SENTENCE
Weather, irradiance, and measured Output Power for five U.S. cities,
30-minute readings, mostly 2011–2016 (Davis/Huron/Santa Barbara/La
Jolla) or 2018–2020 (Amherst).

## PROBLEM 1 IN ONE SENTENCE
Predicted sky-condition and generation-regime from weather features;
best result was logistic regression at 0.772 balanced accuracy for
Davis sky-condition.

## PROBLEM 2 IN ONE SENTENCE
Predicted continuous Output Power; gradient boosting on Davis reached
RMSE=15.17 kW (R²=0.953), the strongest result in the project.

## PROBLEM 3 IN ONE SENTENCE
Compressed the feature space with PCA and an autoencoder; raw,
uncompressed features beat every compressed version, at every
dimension tested.

## PROBLEM 4 IN ONE SENTENCE
Tested whether pseudo-labeling could beat supervised-only training with
few labels; it didn't — SSL gain was slightly negative at every label
fraction (10%/30%/50%).

## PROBLEM 5 IN ONE SENTENCE
Fine-tuned a Davis-pretrained model on a handful of Amherst samples;
transfer beat training from scratch at every sample count, most when
data was scarcest (+21.4% at k=10).

## BEST RESULT
Davis same-city regression, gradient boosting: RMSE=15.17 kW, R²=0.953.

## MOST INTERESTING RESULT
In Problem 5, the SAME setup produced either severe failure (RMSE
140.9 kW) or a clear win (RMSE 27.2 kW) depending only on the
fine-tuning learning rate — found, diagnosed, and fixed.

## BIGGEST FAILURE
Dimension reduction hurt both downstream tasks at every tested
dimension — traced to Cloud Type (a highly predictive categorical
feature) compressing poorly into continuous dimensions.

## BIGGEST LIMITATION
Deliberately small hyperparameter searches throughout (3-5 candidates)
— repeatedly found tuning barely moved results, but a larger search
was never run to confirm there's truly no more headroom.

## MAIN CONCLUSION
Different ML paradigms provide genuinely different, non-interchangeable
benefits — two clearly helped (supervised learning, transfer learning),
two produced honest negative results (dimension reduction, SSL) — and
understanding WHY a technique fails is as valuable as the number itself.

---

## 5 THINGS I ABSOLUTELY NEED TO REMEMBER

1. **GHI is excluded from sky-condition classification because it
   DEFINES the label** — proven by testing with it included (accuracy
   jumped to ~0.98).

2. **Chronological splitting, always** — random shuffling on time-series
   data lets the model see the future, which invalidates every metric.

3. **The Problem 5 learning-rate story is my best "process" story** —
   I found a real bug (negative transfer), diagnosed the exact cause
   (output scale, not a fundamental transfer-learning failure), and
   fixed it. Be ready to walk through this if asked about challenges.

4. **SSL's pseudo-labels were 100% accurate but redundant** — not
   "wrong," just concentrated on the easy class. This is a more
   precise (and more interesting) finding than "SSL failed."

5. **Every number on every slide traces to a saved CSV file** — if
   asked "are you sure about that number," the answer is yes, and I
   can point to `results/FINAL_EXPERIMENT_TABLE.csv`.
