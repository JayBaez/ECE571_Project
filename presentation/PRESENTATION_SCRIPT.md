# Presentation Script

Word-for-word speaking script, matching the speaker notes embedded in
`FINAL_PRESENTATION.pptx` exactly (extracted directly from the built
file, not retyped, to guarantee they match). Read naturally, not
word-for-word if that feels stiff — the notes are written in spoken
language on purpose.

**Estimated speaking time: 11.3-14.0 minutes** (1815 words total, at 130-160 words/minute — target was 8-12 minutes; trim further if rehearsal runs long).

## Slide 1 — Title

**What to say:**

Hi everyone, I'm going to walk you through my ECE571 project: predicting solar power output using five different machine learning approaches on the same dataset. Rather than treating this as five separate assignments, I tried to tell one coherent story about what actually helps when you're trying to forecast photovoltaic power — and, just as importantly, what doesn't. Let's start with why this problem matters in the first place.

## Slide 2 — Why PV Forecasting Matters

**What to say:**

Solar power is different from a fossil-fuel plant because you can't just turn it up when you need more — the sun decides. That intermittency creates real problems: grid operators need to know how much power is coming so they can plan other generation, storage systems need forecasts to know when to charge or discharge, and utilities planning a brand-new site often have little or no historical data to work with. All three of those are fundamentally the same machine learning problem — relate weather to power output — just with different amounts of data available. That's the thread connecting all five parts of this project. Next, let's look at the actual dataset I used.

## Slide 3 — Dataset & Experimental Challenge

**What to say:**

This is the dataset: five cities, mostly six years of data at 30-minute intervals, 22 columns covering irradiance, weather, cloud type, and Output Power. The big number on the right — Davis's plant produces roughly three and a half times the average output of the smallest city. That's not a data quality issue, just different plant sizes, but it matters enormously later when I compare or transfer between cities. And because this is time-series data, every experiment splits chronologically — train on the earliest eighty percent, test on the most recent twenty — never a random shuffle, since that would let the model peek into the future. Now, the overall pipeline I built to run all five ML problems consistently.

## Slide 4 — One Shared Experimental Framework

**What to say:**

Rather than build five separate one-off pipelines, I built one shared framework: load the raw Excel data, clean it, engineer features like the clear-sky index and cyclical time encodings, split chronologically, scale and encode — fitting only on training data — then train and evaluate. That one pipeline feeds all five ML paradigms you see at the bottom: classification, regression — the core task — dimension reduction, semi-supervised learning, and transfer learning. Building it this way meant every paradigm was tested under the exact same leakage-prevention rules, so comparisons between them are fair. Let's go through each one, starting with classification.

## Slide 5 — Problem 1 — Supervised Classification

**What to say:**

The first problem is classification: predict whether the sky is Clear, Partly Cloudy, or Overcast, using weather features. Important detail — I deliberately excluded GHI and related irradiance columns, because the label itself is DEFINED by GHI divided by clear-sky GHI, so including them would just be leaking the answer. The best model here, at 0.772 balanced accuracy, was actually plain logistic regression with class weighting — not a random forest or neural network. Looking at the confusion matrix on the right, Clear conditions are easy to identify, but Partly Cloudy is the hardest class, because it sits on the boundary between the other two — it gets confused with both neighbors instead of having one dominant failure mode. That's a pattern that shows up again later. Next, the core task of the project: regression.

## Slide 6 — Problem 2 — Supervised Regression

**What to say:**

This is the project's core task: predict actual Output Power in kilowatts. Gradient boosting on Davis's own data was the strongest result in the whole project — fifteen kilowatts of RMSE, an R-squared of point nine five three, meaning it explains ninety-five percent of the variance using just weather and irradiance. You can see that in the scatter plot on the right — points cluster tightly around the perfect-prediction line. Two other things worth mentioning quickly: applying the Davis model directly to other cities without any adaptation failed badly, RMSE around 125 kilowatts — but that's mostly a scale problem I'll come back to in problem five. And a GRU sequence model, predicting the next reading from the previous twelve, clearly beat a naive persistence baseline, so it's learning real temporal structure. Now — did compressing the feature space help or hurt?

## Slide 7 — Problem 3 — Dimension Reduction

**What to say:**

Problem three asks: can I compress this feature space without hurting the tasks that depend on it? I tried PCA — the classical, linear method — and a small autoencoder — the deep, nonlinear method — at three latent sizes, fit with zero access to any label. The chart on the left is the honest answer: raw, uncompressed features beat every reduced version, at every dimension, on both tasks. That surprised me going in. The autoencoder did consistently beat PCA, since it can learn curved compressions instead of just straight lines — but neither closed the gap with using everything. I traced part of this to Cloud Type, one of the most useful individual predictors, which doesn't compress well into a handful of continuous numbers. Next: what if you barely have any labels at all?

## Slide 8 — Problem 4 — Semi-Supervised Learning

**What to say:**

For semi-supervised learning, I hid most of Davis's sky-condition labels — down to as few as ten percent — and tested whether pseudo-labeling, where the model labels the unlabeled data itself when confident enough, could beat just training on the small labeled set alone. The honest answer, shown in that curve, is no — the two lines are essentially on top of each other, slightly negative at every label fraction, not just the smallest. I checked WHY, using the hidden true labels purely as a diagnostic after training was already finished. The pseudo-labels were completely accurate, a hundred percent — but ninety-eight percent of the first batch were the same, easy "Clear" class. So the method wasn't adding wrong information, it was adding redundant information. Now, the last paradigm: what if a whole different city could help?

## Slide 9 — Problem 5 — Transfer Learning

**What to say:**

The last paradigm: can Davis, with six years of data, help Amherst, which only has three? I tested three approaches on Amherst's held-out test set: zero-shot, the Davis model applied directly with no Amherst training at all; few-shot, a fresh model trained only on a handful of Amherst samples; and transfer, the Davis-pretrained model fine-tuned on that same handful. Zero-shot failed badly, same scale problem as before. But transfer clearly beat training from scratch at every sample count, and the gain was biggest exactly where it matters most — just ten samples, a twenty-one percent improvement in RMSE. I'll be upfront: my first version of this experiment actually made things worse, because I used too small a learning rate during fine-tuning — I'll explain exactly what happened on the next slide, since it's the most interesting finding in the whole project.

## Slide 10 — Cross-Problem Results

**What to say:**

This table is the whole project in one place. Two paradigms clearly helped — plain supervised learning was strong throughout, and transfer learning gave a real, meaningful boost when Amherst data was scarce. Two paradigms did NOT help, and I want to be upfront about that rather than bury it — dimension reduction hurt performance at every size I tried, and semi-supervised learning's gain was essentially zero. I think that mix is actually the most honest and most useful outcome a project like this can produce — it tells you which techniques are worth reaching for on this kind of problem, and which aren't, based on real evidence instead of assuming every advanced technique automatically helps. Let's dig into the single most interesting result — one that almost didn't turn out this way.

## Slide 11 — Most Interesting Finding

**What to say:**

Here's the single most interesting thing I found in this project. My very first version of transfer learning used a smaller learning rate for fine-tuning than for pretraining — actually what the assignment itself suggested as a reasonable strategy. With that setup, transfer learning was WORSE than training from scratch on the same ten samples — 140.9 kilowatts of RMSE versus 38.2. That's negative transfer. Instead of accepting that number, I checked what the model was actually predicting, and found its average output was stuck near Davis's scale — about 164 kilowatts — because the learning rate was too small to shift it down to Amherst's roughly 64 kilowatt scale in time. Matching the fine-tuning learning rate to the pretraining rate instead dropped RMSE to 27.2, and transfer clearly won. That one hyperparameter was the entire difference between failure and success. Now, a real failure I want to walk through the same way.

## Slide 12 — Failure Analysis

**What to say:**

I want to spend a slide specifically on what didn't work, because that's just as important as the successes. The clearest failure is dimension reduction — compressing the feature space hurt performance on both classification and regression, at every size I tried, even when keeping ninety-five percent of the variance. I traced it to Cloud Type specifically: categorical, and one of the most useful individual features I have, but it doesn't survive being squeezed into a handful of continuous numbers. I proved this by removing it from the compression step and watching both reconstruction quality and downstream accuracy get worse, even though explained variance technically went UP. The bigger lesson: "compression is basically free" is an assumption you have to test, not assume. Problem four's SSL result is a different flavor of failure — redundant, not wrong — worth distinguishing. Let's talk about the project's limitations more broadly.

## Slide 13 — Limitations & Future Work

**What to say:**

Quickly, the honest limitations: five cities and mostly six years of data is enough to demonstrate these methods, but not enough to claim they generalize broadly. The different plant scales across cities kept showing up as a complication I had to work around, not something I fully solved. I kept hyperparameter searches deliberately small throughout, and there was no GPU available during development, though the code will use one automatically if run on GPU hardware. For future work, the most natural next steps are a larger hyperparameter search now that I have working baselines, digging into that Wind Speed anomaly I flagged rather than just noting it, and extending the transfer-learning approach to the other data-scarce cities that were only tested zero-shot in problem two. Let's wrap up with the big-picture takeaways.

## Slide 14 — Conclusion — Five Takeaways

**What to say:**

So, five takeaways. First, PV output can be predicted very effectively from weather and irradiance alone — an R-squared of point nine five three is strong by any standard. Second, the five ML paradigms genuinely behaved differently — there's no single technique that's universally best. Third, cross-city transfer is possible, but only once you handle the output-scale mismatch explicitly — raw zero-shot transfer failed everywhere I tried it. Fourth, transfer learning helped substantially here, while semi-supervised learning essentially didn't — two techniques that sound similar in spirit but produced opposite results. And fifth, maybe the most important lesson: understanding WHY something failed — a learning rate, a categorical feature, a redundant pseudo-label — was at least as valuable as the headline numbers themselves. That's the project. Happy to take questions.

## Slide 15 — Questions

**What to say:**

Thanks — I'm glad to answer questions about any of the five problems, the dataset, the leakage-prevention setup, or the reproducibility checks I ran.
