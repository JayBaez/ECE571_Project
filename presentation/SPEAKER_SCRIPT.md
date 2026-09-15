# Speaker Script

The complete spoken presentation, extracted directly from the speaker
notes embedded in `PV_ML_Final_Presentation.pptx` (guaranteed to match
exactly — not retyped). Read naturally, not word-for-word — this is
written the way a student would actually say it out loud.

**Total spoken words: 1725** (~11.5 min at 150 wpm, ~13.3 min at 130 wpm — slightly over the 1,000-1,500 word guideline; matches the outline's own ~13 minute estimate, so trim Slides 2 and/or 5 verbally if rehearsal runs long).

## Slide 1 — Title (0:30)

Hi, I'm [your name], and this is my ECE571 project — predicting solar power output using five different machine learning approaches on the same dataset. Rather than five separate assignments, I tried to tell one coherent story about what actually helps when forecasting photovoltaic power, and just as importantly, what doesn't. Frame this immediately as ONE investigation, not five disconnected exercises — that's the whole narrative thread for the next 10 minutes. Let's start with why this problem matters in the first place.

## Slide 2 — Why Solar Power Forecasting? (0:45)

Solar power is different from a fossil-fuel plant because you can't just turn it up when you need more — the sun decides. That creates real problems: grid operators need to know how much power is coming so they can plan other generation, storage systems need forecasts to know when to charge or discharge, and utilities planning a brand-new site often have little or no historical data. This is called 'intermittency' — a generation source you can forecast but not control. That's the reason PV forecasting is its own research area. Let's look at the actual dataset I used.

## Slide 3 — The Dataset (1:00)

This is the dataset — five cities, mostly six years of data at 30-minute intervals, 22 columns covering irradiance, weather, cloud type, and Output Power. The big number on the right — Davis's plant produces roughly three and a half times the average output of the smallest city. That's not a data quality issue, it's just different plant sizes, but it matters enormously later. 'Spatiotemporal' just means the data varies across both space — five different cities — and time — years of 30-minute readings. That combination is exactly what makes cross-city comparison and time-based forecasting possible. Now let's look at how I actually structured the experiments around this data.

## Slide 4 — Experimental Design (1:00)

I tested five ML paradigms on this dataset, each with its own fair baseline: classification, regression — the core task — dimension reduction, semi-supervised learning, and transfer learning. Two and five specifically compare same-city versus cross-city performance. And every single experiment uses chronological splitting: earliest eighty percent of a city's data trains, most recent twenty percent tests, never shuffled. Why chronological and not random? Because this is time-series data — random shuffling would let the model see the future relative to what it's being tested on, which would make every reported number meaningless. Before the results, let me quickly show the preprocessing pipeline that feeds all five problems.

## Slide 5 — Preprocessing Pipeline (1:00)

All five problems share one preprocessing pipeline: clean the raw data, engineer features like the Clear-Sky Index and cyclical time encodings, one-hot encode Cloud Type, scale everything, split chronologically, then train. A few specific decisions worth mentioning: the Clear-Sky Index — GHI divided by the theoretical clear-sky maximum — is what defines Problem 1's label. Time features use sine and cosine pairs so that, say, 11pm and midnight end up close together numerically instead of far apart. And Problem 2's forecasting sub-task uses the past 12 readings, about six hours, to predict the next one. Every scaler is fit on TRAINING data only, then applied unchanged to test data — this is the core leakage-prevention rule that holds everywhere in the project. Let's get into the actual results, starting with classification.

## Slide 6 — Problem 1 — Classification (1:00)

First result: classification. Predict whether the sky is Clear, Partly Cloudy, or Overcast, from weather features. I excluded GHI and related irradiance columns, because the label itself is defined by GHI divided by clear-sky GHI — including them would just leak the answer. The best model, at 0.772 balanced accuracy, was plain logistic regression with class weighting — not a random forest or neural network. Looking at the confusion matrix, Clear is easy, but Partly Cloudy is the hardest class, because it sits on the boundary between the other two. Balanced accuracy averages each class's recall equally, so a model that always guessed 'Clear' — the majority class at 72% of the data — can't fake a good score. Now, the core task of the whole project — regression.

## Slide 7 — Problem 2 — Regression (1:15)

This is the project's core task — predict actual Output Power in kilowatts. Gradient boosting on Davis's own data was the strongest result in the whole project: fifteen kilowatts of RMSE, an R-squared of point nine five three — it explains ninety-five percent of the variance using just weather. Applying that same Davis model directly to other cities without any adaptation failed badly — RMSE around 125 kilowatts. But that's mostly a scale problem, not a failure to learn weather patterns: rescaling the predictions by the target city's own average recovers R-squared of point five five to point eight four. RMSE penalizes big misses more heavily than MAE does, since it squares errors before averaging — that's why I report both. And nRMSE, normalized by the target's range, is what lets me compare error across cities with very different power scales. Next — did compressing the feature space help or hurt?

## Slide 8 — Problem 3 — Dimension Reduction (1:00)

Problem three asks: can I compress this feature space without hurting the tasks that depend on it? I tried PCA — the classical, linear method — and a small autoencoder — the deep, nonlinear method — at three sizes, fit with zero access to any label. The honest answer is no: raw, uncompressed features beat every reduced version, at every dimension, on both downstream tasks. That surprised me going in. I traced part of this to Cloud Type — a feature that turns out to be highly useful but doesn't compress well into a handful of continuous numbers. Explained variance tells you how much of the INPUT features' spread you kept — it says nothing about whether that spread includes the specific information needed to predict a label. That's exactly the gap this result exposes. Next — what if you barely have any labels at all?

## Slide 9 — Problem 4 — Semi-Supervised Learning (1:00)

For semi-supervised learning, I hid most of Davis's sky-condition labels — down to as few as ten percent — and tested whether pseudo-labeling, where the model labels the unlabeled data itself when confident enough, could beat just training on the small labeled set alone. The honest answer is no, especially interesting at the ten percent case — the gap is slightly negative at every label fraction. I checked why, using the hidden true labels purely as a diagnostic after training was already finished. The pseudo-labels were completely accurate — a hundred percent — but ninety-eight percent of the first batch were all the same, easy 'Clear' class. This means the method wasn't adding WRONG information — it was adding REDUNDANT information. That's a more precise, more useful finding than just 'SSL didn't work.' Now, the last paradigm, and the most interesting result in the project — what if a whole different city could help?

## Slide 10 — Problem 5 — Transfer Learning (1:15)

The last paradigm, and the most interesting result in the whole project. Can Davis, with six years of data, help Amherst, which only has three? I compared zero-shot — no Amherst training at all — few-shot — a fresh model trained only on a handful of Amherst samples — and transfer — the Davis-pretrained model fine-tuned on that same handful. Zero-shot failed badly, same scale problem as before. Transfer clearly beat training from scratch at every sample count, most at just ten samples — a twenty-one percent improvement.
IMPORTANT POINT — the real story: my very first version of this experiment made things WORSE, not better, because I used too small a learning rate during fine-tuning — actually what the assignment itself suggested. I dug into what the model was predicting, found its output was stuck near Davis's scale, and fixed it by matching the pretraining learning rate instead. One hyperparameter was the entire difference between failure and success.  Let's zoom out and see all five problems side by side.

## Slide 11 — Overall Results (1:00)

This table is the whole project in one place. Two paradigms clearly helped — supervised learning was strong throughout, and transfer learning gave a real boost when Amherst data was scarce. Two did NOT help, and I want to be upfront about that rather than bury it — dimension reduction hurt performance at every size I tried, and semi-supervised learning's gain was essentially zero. I deliberately did not combine these into one overall score — a balanced accuracy number and an RMSE in kilowatts simply aren't the same kind of quantity, so a fake combined ranking would be misleading, not helpful. Let's pull out the handful of findings that matter most.

## Slide 12 — Most Important Findings (0:45)

Four findings I'd want you to remember. First, same-city prediction was far easier than cross-city — mostly a scale problem, not a weather-learning problem. Second, a simple linear model beat every more complex model on classification — a genuinely useful, somewhat counterintuitive result. Third, semi-supervised learning didn't help even with just ten percent labels — the pseudo-labels were accurate but redundant. And fourth, transfer learning clearly helped, but only after I found and fixed a real negative-transfer bug during development. Notice these are genuinely mixed — two positive, two negative. That honesty is intentional; I didn't force every technique to look good. Let's talk about what this project doesn't cover.

## Slide 13 — Limitations & Future Work (0:45)

Quick, honest limitations: every reading falls in a ten-to-three daytime window, so nothing here describes nighttime behavior. Only five cities, and only one source-target pair tested for transfer learning. Plant-scale differences kept showing up as something I had to work around, not fully solve. And I kept hyperparameter searches small throughout, on purpose. The future-work items on the right are explicitly things I have NOT done — a larger search, more cities, longer horizons, real forecast inputs, better domain adaptation, uncertainty estimation. I want to be clear these are proposals, not completed work. Let's wrap up.

## Slide 14 — Conclusion (0:45)

So — machine learning predicts PV output very effectively within a city, but cross-city domain shift remains a major challenge — one that fine-tuning can address, but naive transfer cannot. My strongest result was Davis same-city regression, RMSE of fifteen point one seven kilowatts, R-squared of point nine five three. The most interesting finding was that single learning-rate story in Problem five. And the biggest limitation is that I kept every hyperparameter search deliberately small, so I can't say for certain there isn't more headroom out there. The real thread connecting all five problems — supervised learning and transfer learning clearly helped, dimension reduction and semi-supervised learning didn't, and understanding WHY in each case mattered as much as the numbers themselves. Thanks — happy to take questions.
