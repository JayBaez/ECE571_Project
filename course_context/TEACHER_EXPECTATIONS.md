# Teacher Expectations — Practical Checklist

**Source of this checklist:** the actual project specification ("Machine
Learning Course Project — Photovoltaic (Solar) Power Prediction on
Large-Scale Spatiotemporal Data"), provided directly, plus its §6 Grading
Scheme. This supersedes the earlier best-effort version of this file that
was built only from a paraphrase — everything below is quoted/derived from
the real document.

Legend: **REQUIRED** = explicitly called for in the spec · **RECOMMENDED**
= strongly implied / good practice for the stated goal · **OPTIONAL** =
spec explicitly says optional or "encouraged."

---

## The actual grading rubric (§6, 100 points)

| Component | Points | What earns it |
|---|---|---|
| Correctness & reproducibility | 20 | Code runs, fixed seeds, no leakage, splits documented |
| Breadth of methods | 20 | Multiple classical + at least one deep method **per problem**, ablation table |
| Best metric (leaderboard) | 30 | Your **best** RMSE / Accuracy / transfer gain etc. vs. baselines — biggest single component |
| Analysis & insight | 20 | Why methods work, failure modes, comparison to the paper's findings |
| Report + code + video clarity | 10 | Clear write-up and presentation |

Grade bands: A ≥ 90, B 80–89, C 70–79, D 60–69, F < 60. **Grading rewards
exploration and your best achieved metric, not just a single method** —
try multiple methods per problem; only the best result is graded, so
there's no penalty for an early attempt underperforming as long as a
later one is reported.

**Key implication for prioritization:** "Best metric" (30 pts) and
"Breadth of methods" (20 pts) together are half the grade — trying only
one model per problem, even if it's a good one, caps you well below
what breadth + iterating toward a best result would earn. "Analysis &
insight" (20 pts) means results tables alone aren't enough — each
problem needs a short discussion of *why* one method beat another and
where it failed.

## Global requirements

- **REQUIRED** — No test-set leakage: scalers fit on TRAIN only (spec
  §3.2 says this explicitly).
- **REQUIRED** — Time series never randomly shuffled; always split
  chronologically, or per the same-city/cross-city protocols in §3.1.
- **REQUIRED** — Random seeds fixed and reported; **regression results
  reported as mean ± std over ≥ 3 random seeds** (spec §3.2, §5.2).
- **REQUIRED** — Train/test split indices kept identical across all
  methods being compared for a given problem (spec §3.2).
- **REQUIRED** — Cloud Type treated as categorical (one-hot/embedding),
  never as a raw number (spec §2, confirmed independently in
  `DATASET_PROFILE.md`).
- **REQUIRED** — Drop or interpolate the ~4 missing `Output Power` rows
  in Amherst; never let NaN leak into training (spec §3.2 — matches the
  exact 4 rows found independently in `DATASET_PROFILE.md`).
- **REQUIRED** — Every problem should try **both classical and deep
  learning methods**; the best result (not the first attempt) is graded
  (spec §1, §4).
- **RECOMMENDED** — One reusable framework/pipeline across Problems 1–5
  rather than five separate scripts (matches your own stated
  architecture goal).
- **RECOMMENDED** — Prefer course-taught methods where reasonable (see
  `ML_METHOD_MAP.md`), balanced against the spec's explicit encouragement
  to try deep learning everywhere.

## Problem 1 — Supervised Classification

- **REQUIRED (exact definitions given by spec, §4):**
  - **Sky-condition class** (3-way) from Clear-Sky Index
    `k = GHI / Clearsky GHI`: Clear (k ≈ 0.85–1.0), Partly cloudy
    (0.4–0.85), Overcast (< 0.4).
  - **WARNING (spec, verbatim intent):** do not use `GHI` or
    `Clearsky GHI` as input features when predicting this label — that
    leaks the label definition directly. Use the other weather
    variables instead. **RESOLVED decision (confirmed via EDA —
    `EDA_REPORT.md`):** also **exclude `DHI`, `DNI`, and
    `Solar Zenith Angle` from the primary sky-condition model.**
    Together they can algebraically reconstruct `GHI` quite closely
    (`GHI ≈ DNI · cos(zenith) + DHI`), and `Solar Zenith Angle` alone
    was empirically confirmed to correlate with `GHI` at **-0.74** —
    strong enough that including them risks defeating the spirit of
    the spec's exclusion rule even while technically obeying its
    letter. **As a secondary, explicitly-labeled ablation** (not the
    headline result), also run the same classifier *with* those three
    columns included and report the accuracy difference — this turns
    the leakage risk into a demonstrated finding for the "Analysis &
    insight" rubric component (20 pts), rather than just a caveat.
  - **Generation-regime class** (3-way): bin `Output Power` into
    Low/Medium/High **terciles, computed per city** (not globally).
- **REQUIRED** — Confusion matrix figure.
- **REQUIRED** — Per-class metrics; discuss which classes are hardest
  and why (this is the "Analysis & insight" tie-in).
- **REQUIRED metric** — **Balanced accuracy** (mean of per-class recall)
  is the spec's preferred headline number because classes are imbalanced
  (clear-sky dominates) — prefer it over raw accuracy (spec §5.1).
- **RECOMMENDED** — Multiple classical models + at least one deep model
  (MLP) per the breadth requirement.

## Problem 2 — Supervised Regression (spec calls this the "core task")

- **REQUIRED** — Same-city: chronological 80/20 split; spec specifically
  suggests trying **Davis and Amherst**.
- **REQUIRED** — Cross-city (zero-shot): train on **Davis (6-year)**,
  test on **Huron / Santa Barbara / La Jolla** — note Amherst is *not*
  in this cross-city test group (it's reserved as the Problem 5 transfer
  target). Normalize per city.
- **REQUIRED** — Sequence variant: window of last **K=12** steps →
  predict next step's `Output Power` (very-short-term forecast).
- **REQUIRED metrics** — RMSE and MAE for every method/setup; **always
  report RMSE together with nRMSE** so cross-city numbers are
  comparable (spec §5.2) — directly relevant given the large per-city
  scale differences documented in `DATASET_PROFILE.md`.
- **REQUIRED** — Mean ± std over ≥ 3 seeds.
- **REQUIRED plot** — Learning-curve or prediction-vs-truth plot.
- **RECOMMENDED (3-year vs 6-year ablation, spec §3.1)** — Compare a
  model trained on a city's 3-year sheet vs. its 6-year sheet to see
  whether more history helps. This is *why* the `'14-'16` sheets exist
  alongside the `'11-'16` sheets — see the corrected guidance in
  `DATASET_PROFILE.md`.
- **RECOMMENDED preprocessing (spec §3.2, not mandatory but suggested):**
  Clear-Sky Index `k`, sin/cos encodings of Hour and Day-of-Year, lag
  features of `Output Power` (t-1, t-2) for sequence models.

## Problem 3 — Dimension Reduction for Classification/Regression

- **REQUIRED** — Reduce features to **d = 2 / 5 / 10** and feed the
  reduced representation into *both* the Problem-1 classifier and the
  Problem-2 regressor (this is explicitly a downstream-usefulness test,
  not reduction for its own sake).
- **REQUIRED comparison** — "Reduce-then-predict" vs. "predict on raw
  features" — the whole point of the problem per the spec.
- **REQUIRED** — Choose `d` via the explained-variance curve / elbow,
  and justify the choice in the report.
- **REQUIRED (intrinsic metrics)** — Reconstruction MSE for
  autoencoder/VAE; explained-variance ratio for PCA/Kernel PCA.
- **REQUIRED (downstream metrics)** — Balanced accuracy (from P1) and
  RMSE/MAE (from P2) on reduced features, compared to raw features.
- **OPTIONAL but encouraged** — 2-D t-SNE/UMAP scatter colored by
  sky-condition or Output-Power bin; trustworthiness/continuity metric
  for that embedding.
- **Course-alignment note (see `ML_METHOD_MAP.md`)** — PCA/Kernel PCA
  are directly taught (Week10); autoencoder/VAE are a reasonable
  extension of taught MLP material but not literally covered in the
  slides; t-SNE/UMAP are visualization-only tools, not taught at all.

## Problem 4 — Semi-Supervised Learning

- **REQUIRED** — Pick a base task: either Problem-2 regression or
  Problem-1 classification.
- **REQUIRED** — Withhold labels so only **p ∈ {10%, 30%, 50%}** of
  training samples are labeled; test on the full held-out test set.
- **REQUIRED plot** — Label-efficiency curve: chosen metric
  (spec suggests Macro-F1 or nRMSE) vs. labeled fraction, comparing your
  SSL model to a supervised-only model trained on the *same* p%.
- **REQUIRED comparison** — "Gain over supervised" at each fraction =
  `metric_SSL − metric_supervised_only`; explicitly check whether at
  **10% labels** the SSL method beats the 10%-only supervised baseline.
- **REQUIRED metric** — **AUC of the label-efficiency curve itself**
  (not classifier ROC-AUC) as a single-number summary of how fast
  performance rises as labels are added. *(This corrects an ambiguity
  in the earlier version of this file — "AUC" here means area under the
  metric-vs-%labels curve, not the usual classification ROC-AUC.)*
- **RESOLVED (was previously an open decision — see below)** — The
  spec doesn't name a required SSL algorithm, so any reasonable
  approach satisfies the letter of the spec. **Decision:** primary
  method is **pseudo-labeling/self-training** (simple, easy to
  implement and explain, satisfies every REQUIRED item above).
  **Breadth addition:** also run **graph-based label propagation**
  (scikit-learn's `LabelPropagation`/`LabelSpreading` — a direct,
  off-the-shelf implementation of the course-taught Week12 method) as
  a second SSL method, since the grading rubric's "Breadth of methods"
  (20 pts) rewards comparing more than one approach per problem, and
  this one is both taught in the course and cheap to add given
  scikit-learn already implements it. Transductive SVM and Co-training
  remain available as further optional additions if time allows, but
  aren't part of the primary plan.

## Problem 5 — Transfer Learning

- **REQUIRED** — SOURCE = **Davis (6-year, data-rich)**. TARGET =
  **Amherst (3-year, data-scarce)** or any 3-year-only city — spec
  explicitly frames this as the "cross-city challenge."
- **REQUIRED baselines** — (a) **Zero-shot**: source-trained model
  applied directly to target, no fine-tuning. (b) **Few-shot**: model
  trained only on a small target-labeled set, with spec-suggested sizes
  **k = 10 / 50 / 100** target samples.
- **REQUIRED** — Your own transfer method must be shown to beat *both*
  baselines on the target domain.
- **REQUIRED table** — Target-domain RMSE/accuracy for zero-shot,
  few-shot, and your transfer method, side by side.
- **REQUIRED discussion** — What transferred and what didn't; explicit
  **transfer gain** = `metric_target − metric_source_only_baseline`;
  report honestly if negative (negative transfer), don't hide it.
- **Course-alignment note** — Transfer learning as a topic is not
  covered anywhere in the 19 course files (see `ML_METHOD_MAP.md`) —
  this problem draws entirely on general ML practice (fine-tuning /
  feature reuse), which is fine per the spec but worth naming plainly
  in the report/presentation.

## Report, Code, Video, Reproducibility (§7 Submission Requirements)

- **REQUIRED — Report:** PDF or DOCX, **8–12 pages**: background, data
  description, methods, experimental setup, results tables/figures,
  discussion, limitations, and a reproducibility section (seeds,
  environment, how to run).
- **REQUIRED — Code:** runnable repository with `requirements.txt` (or
  `environment.yml`), a README explaining how to run each problem, and
  saved result artifacts (`results.csv`/`results.json` with final
  metrics).
- **REQUIRED — Video:** **8–12 minute** screencast — motivation,
  methods, key results, single most interesting finding — via a
  shareable link (YouTube unlisted / Google Drive / Panopto). Narrated;
  a silent slideshow is explicitly stated as not sufficient.
- **REQUIRED — Reproducibility:** fixed seeds, exact train/test split
  indices, package versions used.
- **Deadline:** 11:59:59 PM on 12/09/2026 (Wednesday). **Late penalty:**
  20 points/day until zero remain. No excuses/extensions per the spec.

## Vibe-Coding / AI-Assistance Policy (§8 — directly relevant since this
project is being built with AI coding assistance)

- **REQUIRED** — You must understand and be able to explain every
  line/method submitted — matches your own stated background/goal
  exactly.
- **REQUIRED** — Disclose AI assistance in the report: a short paragraph
  on what tools were used and for what parts. **Action item:** keep a
  running note of what Claude Code / other agents contributed, so this
  paragraph is easy to write later (see `YOUR_PROJECT_NOTES.md`).
- **REQUIRED** — AI-generated code must be modified/understood by you,
  correctly attributed; verbatim copying of another student's or a
  published solution is academic dishonesty.
- **REQUIRED** — Reported metrics must come from runs you (the team)
  actually executed and can reproduce — directly reinforces "never
  fabricate metrics."

## Superseded items from earlier versions of this file

The original version of this checklist (built without the real spec)
listed four "open questions worth confirming with the professor."
Three were answered directly by the spec: sky-condition/generation-
regime definitions (now exact, see Problem 1 above), and whether
PCA/KPCA alone is sufficient for Problem 3 (spec requires
reconstruction + explained variance regardless of whether an
autoencoder is added, so PCA/KPCA alone is *acceptable*). The fourth —
Problem 4's SSL algorithm choice — was left open through Phase 3 and
is now **resolved** (see Problem 4 above: pseudo-labeling as primary,
graph-based label propagation added for breadth). The Solar Zenith
Angle/DHI/DNI leakage question (Problem 1, above) is also now
**resolved** as of this update, based on the project owner's direct
guidance following the Phase 3 EDA findings. No open decisions remain
in this file as of this update — see `YOUR_PROJECT_NOTES.md` for
anything that comes up in later phases.
