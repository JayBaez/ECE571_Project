# Slide Outline

Built from `report/FINAL_REPORT.md` and the actual saved results in
`results/` — every number here traces to a saved results file, not
memory. Target: ~10 minutes, 14 slides. Per Part 3's guidance, time is
NOT split evenly — Problem 2 (the core task) and Problem 5 (the most
interesting finding) get the most depth.

**Note on report version:** the currently-pushed `report/FINAL_REPORT.md`
is the earlier, more consolidated version (5,853 words, 17 sections) —
a newer, more detailed draft (19 pages, with explicit Objective/
Methods/Results/Discussion subsections per problem) was built and
approved in an earlier session but has not yet been pushed to GitHub.
Both versions report identical underlying numbers and findings, so
this outline is unaffected either way — it draws directly from the
saved results files, not from either report's prose.

---

# Slide 1
**Title:** Photovoltaic (Solar) Power Prediction on Large-Scale Spatiotemporal Data
**Purpose:** Introduce the project, presenter, and context.
**Visual:** Simple title layout — no figure. Small solar/energy motif acceptable.
**Main talking point:** "This is a machine learning project that predicts solar power output using five different ML approaches on the same dataset."
**Time:** 0:30

# Slide 2
**Title:** Why Solar Power Forecasting?
**Purpose:** Motivate the problem before showing any data.
**Visual:** No existing project figure fits here (this is conceptual/motivational, not a data result) — recommend a simple text/icon layout, not a fabricated chart. Flagging this per Part 16: no figure is being invented for this slide.
**Main talking point:** PV output is intermittent (weather-dependent, not controllable like a gas plant); accurate forecasts help grid operators, storage systems, and new-site planning; this is fundamentally a weather-to-power ML problem.
**Time:** 0:45

# Slide 3
**Title:** The Dataset
**Purpose:** Show what data the project is built on, and introduce the central challenge (city scale differences).
**Visual:** `figures/eda/output_power_by_city_boxplot.png` — shows Output Power distribution by city, making the scale difference immediately visible.
**Main talking point:** 5 cities, 30-minute readings, mostly 2011-2016 (10:00-15:00 daytime window only), 22 columns (irradiance, weather, Cloud Type, Output Power). "Spatiotemporal" = data varies across both space (5 cities) and time (readings every 30 min, multiple years) — that's exactly why this dataset supports cross-city and forecasting experiments.
**Time:** 1:00

# Slide 4
**Title:** Experimental Design — Five ML Paradigms
**Purpose:** Introduce all five problems as one coherent investigation, not five separate assignments.
**Visual:** Simple diagram/table: 5 problem boxes (Classification, Regression, Dimension Reduction, Semi-Supervised Learning, Transfer Learning) — no existing project figure needed, a built diagram is appropriate here.
**Main talking point:** Each paradigm tested with a fair, matched baseline. Same-city vs. cross-city framing introduced here (used in Problems 2 and 5). Chronological splitting used everywhere — earliest 80% of a city's data trains, latest 20% tests, never shuffled.
**Time:** 1:00

# Slide 5
**Title:** Preprocessing Pipeline
**Purpose:** Show how raw data becomes model-ready data, and why leakage prevention matters.
**Visual:** Pipeline diagram (Raw data -> Missing-value handling -> Feature engineering -> Categorical encoding -> Scaling -> Chronological split -> ML models) — built diagram, not an existing figure.
**Main talking point:** Clear-Sky Index (GHI/Clearsky GHI) built for Problem 1's label; cyclical sine/cosine time features so hour 23 and hour 0 are numerically close; Cloud Type one-hot encoded (categorical, not ordered); K=12 lag/sequence window for Problem 2's forecasting sub-task; scalers fit on training data only.
**Time:** 1:00

# Slide 6
**Title:** Problem 1 — Classification
**Purpose:** Show the classification task, best model, and confusion matrix.
**Visual:** `figures/problem1/problem1_sky_condition_confusion_matrix_Davis.png`
**Main talking point:** Predict sky-condition (Clear/Partly Cloudy/Overcast) from weather — GHI excluded since it defines the label. Best model: logistic regression (balanced weight), 0.772 balanced accuracy — notably NOT an ensemble or neural net. Hardest class: Partly Cloudy (F1=0.566) — sits between the other two on the Clear-Sky Index scale, confused with both. Class imbalance (Clear=72% of data) is why balanced accuracy, not raw accuracy, is the headline metric.
**Time:** 1:00

# Slide 7
**Title:** Problem 2 — Regression (Core Task)
**Purpose:** The project's central result — give this the most depth of any single-problem slide.
**Visual:** `figures/problem2/problem2_predicted_vs_actual_davis.png` (primary); reference `figures/problem2/problem2_cross_city_comparison.png` numbers verbally or as a small inset if space allows.
**Main talking point:** Best model: gradient boosting, Davis same-city — RMSE=15.17 kW, MAE=8.31 kW, R²=0.953 — the strongest result in the whole project. Cross-city zero-shot (same model, applied directly to Huron/Santa Barbara/La Jolla with no retraining): RMSE ~ 125 kW — roughly 8x worse. Explain why: scale mismatch (Davis's plant averages 164 kW vs. 47-50 kW elsewhere), not a failure to learn the weather-power relationship — a diagnostic rescaling recovered R²=0.55-0.84.
**Time:** 1:15 (upper end of the 60-90s guideline, given this is the core task)

# Slide 8
**Title:** Problem 3 — Dimension Reduction
**Purpose:** Show whether compressing the feature space helped or hurt.
**Visual:** `figures/problem3/problem3_explained_variance.png` (primary) — optionally pair with `figures/problem3/problem3_downstream_classification.png` if two small figures fit cleanly.
**Main talking point:** PCA (linear) and an autoencoder (nonlinear) tested at d=2/5/10, both unsupervised (zero label access during fitting). Did reduction help? **No** — raw features beat every reduced representation at every dimension, on both downstream tasks (raw: 0.743 balanced accuracy / 23.86 kW RMSE vs. best reduced, Autoencoder d=10: 0.661 / 29.69 kW). Traced to Cloud Type compressing poorly despite being highly predictive.
**Time:** 1:00

# Slide 9
**Title:** Problem 4 — Semi-Supervised Learning
**Purpose:** Show whether unlabeled data helped when labels were scarce — especially at 10%.
**Visual:** `figures/problem4/problem4_label_efficiency_curve.png`
**Main talking point:** 10%/30%/50% of Davis's sky-condition labels kept; rest hidden (features visible, labels hidden) to simulate a partly-labeled deployment. Pseudo-labeling (train -> predict unlabeled -> keep confident predictions -> retrain) compared against supervised-only on the same labeled subset. **Did unlabeled data help, especially at 10%? No** — SSL gain was slightly negative at every fraction (10%: -0.0048 macro F1). Why: pseudo-labels were 100% accurate (checked after training only) but 98% were the easy "Clear" class — redundant, not wrong.
**Time:** 1:00

# Slide 10
**Title:** Problem 5 — Transfer Learning
**Purpose:** The most interesting result in the project — give this real depth, similar to Problem 2.
**Visual:** `figures/problem5/problem5_transfer_curve.png` (or `problem5_zero_shot_vs_transfer.png` if a simpler bar-style comparison presents better on a slide)
**Main talking point:** Source: Davis (data-rich). Target: Amherst (data-poor). Zero-shot (no Amherst training) fails badly, RMSE=184.17 kW — same scale-mismatch issue as Problem 2. Transfer (Davis-pretrained, fine-tuned on k Amherst samples) vs. few-shot (fresh model, same k samples): transfer wins at every k, most at k=10 (RMSE 30.03 vs. 38.22 kW, **+21.4%**). **Negative transfer did occur during development** — an early version with too-small a fine-tuning learning rate scored RMSE=140.9 kW, worse than few-shot — found, diagnosed (output scale couldn't adjust fast enough), and fixed.
**Time:** 1:15 (upper end of guideline — this is the most interesting finding)

# Slide 11
**Title:** Overall Results
**Purpose:** One clean summary table spanning all five problems.
**Visual:** Native table (not a chart) — Problem / Best Method / Main Result / Main Finding, one row per problem. Metrics are NOT combined into a single fake score, since balanced accuracy and RMSE aren't comparable quantities.
**Main talking point:** Two paradigms clearly helped (supervised learning, transfer learning); two produced honest negative results (dimension reduction, semi-supervised learning) — reported as real findings, not hidden.
**Time:** 1:00

# Slide 12
**Title:** Most Important Findings
**Purpose:** Highlight 3-4 findings, not a recap of every number already shown.
**Visual:** Simple numbered list, no figure needed.
**Main talking point (4 findings, all directly supported by results already shown):**
1. Same-city prediction was far easier than cross-city prediction (15-20 kW vs. ~125-184 kW RMSE) — driven mainly by output-scale mismatch, not a failure to learn weather patterns.
2. A simple linear model (logistic regression) beat every ensemble and neural net on Problem 1's classification task.
3. Semi-supervised learning did NOT provide useful gains even with as little as 10% labeled data — pseudo-labels were accurate but redundant.
4. Transfer learning clearly helped — but only after a real negative-transfer bug (a learning-rate issue) was found and fixed during development.
**Time:** 0:45

# Slide 13
**Title:** Limitations & Future Work
**Purpose:** Honest, clearly-labeled limitations, with future work clearly marked as NOT done.
**Visual:** Two-column layout, no figure needed.
**Main talking point:** Limitations — mid-day-only data (10:00-15:00, no nighttime), only 5 cities, city-specific power scale (a recurring complication), limited years for Amherst (3 vs. 6), deliberately small hyperparameter searches, no GPU during development. Future work (explicitly labeled as NOT completed) — larger hyperparameter search, additional cities for transfer learning (Huron/Santa Barbara/La Jolla currently zero-shot only), longer forecasting horizons, real weather-forecast inputs instead of historical readings, uncertainty estimation alongside point predictions.
**Time:** 0:45

# Slide 14
**Title:** Conclusion
**Purpose:** End with one clear, evidence-based takeaway — not a repeat of the abstract.
**Visual:** No figure — clean closing slide.
**Main talking point:** Machine learning predicts PV output very effectively within a city (R²=0.953), but cross-city transfer requires explicit handling of output-scale differences — raw cross-city/zero-shot transfer failed everywhere it was tried, while fine-tuning fixed it. Strongest result: Davis same-city regression (RMSE=15.17 kW). Most interesting finding: the Problem 5 learning-rate story. Biggest limitation: small hyperparameter searches throughout, never confirmed against a larger search. Closing line: "Whether an advanced ML technique helps has to be tested directly, not assumed — and that's the real finding of this project."
**Time:** 0:45

---

**Total estimated time: ~13 minutes** (sum of the above) — slightly over the 8-12 minute target. Recommend trimming Slide 2 and/or Slide 5 verbally (both are supporting context, not core results) if rehearsal runs long; every other slide ties directly to a required result.
