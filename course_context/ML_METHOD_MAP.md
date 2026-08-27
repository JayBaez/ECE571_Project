# ML Method Map — Course Concept → Project Application

Format: **Course concept** (source deck) → possible project application → problem
number(s) → recommended model role. Grouped into three tiers as requested:
taught in course, useful but not covered, and probably unnecessary.

---

## Tier 1 — Taught in course, directly usable

| Course concept | Source | Project application | Problem | Recommended role |
|---|---|---|---|---|
| Logistic regression | Week03a | Binary/multi-class sky-condition or generation-regime classifier | P1 | Baseline classifier |
| Decision Tree (ID3/CART/C4.5, entropy, Gini) | Week06a | Interpretable classifier | P1 | Classical model, good for explaining to professor |
| Bagging / Random Forest | Week06b | Ensemble classifier and regressor | P1, P2 | Stronger classical model |
| Naive Bayes (incl. Gaussian NB) | Week07 | Simple generative classifier baseline | P1 | Baseline / sanity-check classifier |
| kNN (classification + regression variant) | Week04 | Simple non-parametric baseline | P1, P2 | Baseline |
| Linear SVM (hard/soft margin) | Week05 | Linear classifier with margin | P1 | Classical alternative to logistic regression |
| Linear Regression (closed form + gradient descent) | Week03b | Output Power regression | P2 | Baseline regressor |
| Multilayer Perceptron (MLP) | Week13 | Nonlinear classifier/regressor; also the natural building block for an autoencoder | P1, P2, P3 (as autoencoder) | Deep learning model |
| Vanilla RNN | Week15a | Sequence forecasting of Output Power | P2 (sequence, K=12) | Deep sequence model |
| LSTM, GRU | Week15b | Sequence forecasting with longer memory | P2 (sequence, K=12) | Stronger deep sequence model — likely the best fit for the K=12 sub-task |
| L1/L2 regularization, dropout, early stopping | Week11 | Prevent overfitting in any of the above models | P1, P2, P3 | Applied to all trained models, not a standalone method |
| PCA (max-variance / min-reconstruction views, explained variance) | Week10 | Direct dimensionality reduction | P3 | Primary method — directly matches "PCA/KPCA" in the spec |
| Kernel PCA | Week10 | Nonlinear dimensionality reduction | P3 | Secondary method — also directly matches the spec |
| Transductive SVM | Week12 | SSL classifier | P4 | One taught SSL option (needs a linear-SVM-style setup) |
| Co-training | Week12 | SSL classifier using two feature "views" | P4 | Taught SSL option — would need two reasonably independent feature groups (e.g., irradiance-based vs. weather-based) |
| Graph-based label propagation (min-cut / soft-cut / kNN graph) | Week12 | SSL classifier via similarity graph | P4 | **SELECTED** — taught SSL option, and scikit-learn implements it directly (`LabelPropagation`/`LabelSpreading`) — chosen as the course-aligned second method alongside pseudo-labeling (see Tier 2 below and `TEACHER_EXPECTATIONS.md`) |
| Train/test split, k-fold CV, bootstrapping | Week02 | Model validation methodology | All | Methodology, not a model |
| Accuracy, precision, recall, F1, confusion matrix, ROC/AUC | Week02 | Classification evaluation | P1, P4 (AUC) | Required metrics |
| MSE | Week02 | Regression evaluation | P2 | Base for RMSE/nRMSE |
| K-means clustering | Week08 | Exploratory clustering of days/regimes | Not required by any problem as stated | Optional exploratory tool only |
| Gaussian Mixture Model + EM | Week09 | Soft clustering alternative to k-means | Not required by any problem as stated | Optional exploratory tool only |

**Note on boosting:** "AdaBoost" is named once, only in passing, as an
example algorithm in the Week12 semi-supervised deck ("Naive Bayes,
logistic regression, SVM, Adaboost, etc."). There is **no dedicated
boosting lecture** in the 19 files — it is not taught in the depth that,
say, Random Forest or Decision Trees are. If you want a boosting model
(e.g., gradient boosting) for Problem 1 or 2, treat it as a
"useful but not covered" addition, not a course-taught technique.

## Tier 2 — Useful for this project, but not covered in the 19 files

| Concept | Project application | Problem | Notes |
|---|---|---|---|
| Autoencoder (as a specific named architecture/loss) | Learned nonlinear dimensionality reduction | P3 | Buildable from taught MLP material (Week13) + a reconstruction loss instead of a classification loss — a reasonable, explainable extension, but the professor may want to know it's not literally from the slides |
| Variational Autoencoder (VAE) | Probabilistic dimensionality reduction | P3 (optional per spec) | Meaningfully more complex than plain autoencoder; given your background, only worth it if PCA/KPCA/autoencoder feel insufficient |
| t-SNE / UMAP | Visualization of reduced embeddings | P3 (visualization only) | Not a "model" to train/compare — standard library tools (`sklearn.manifold.TSNE`, `umap-learn`), fine to use for plots without deep conceptual grounding from the course |
| Pseudo-labeling / self-training | Simple SSL approach | P4 | **SELECTED (primary method)** — not one of the three methods taught in Week12, but the spec (§4) doesn't mandate a specific algorithm, so this is fully compliant; simpler to implement/explain than the taught methods. Paired with graph-based label propagation (above) for breadth. |
| Transfer learning / fine-tuning (freezing layers, adapting a final layer, feature reuse) | Davis → Amherst adaptation | P5 | Not covered anywhere in the 19 files — this entire problem draws on general ML practice rather than lecture content |
| nRMSE as an explicit named metric | Cross-city regression comparison | P2 | **Now confirmed REQUIRED by the actual spec** (§5.2: "always report RMSE together with nRMSE") — trivial to compute (RMSE divided by a normalizer, e.g. target range or mean) once RMSE is understood (which is taught), but the metric name itself isn't in the slides |
| Gradient boosting (e.g. a scikit-learn `GradientBoostingClassifier/Regressor`) | Stronger classical model | P1, P2 (optional) | Natural "stronger classical" step beyond Random Forest, but not covered — Random Forest is the course-taught equivalent choice |
| TCN (Temporal Convolutional Network) / small Transformer | Sequence forecasting (K=12) | P2 (sequence variant) | **Explicitly named in the real spec's §3.2** alongside LSTM/GRU as example sequence models — legitimate "breadth" points, but neither is taught in the course (no TCN or attention/Transformer lecture exists) |

## Tier 3 — Probably unnecessary for this project

| Concept | Why it's a poor fit here |
|---|---|
| CNN (2D convolution, as taught in Week14) | Week14's CNN material is built around 2D image data (LeNet, image classification/age-estimation examples). This project's data is tabular time-series, not images — there's no natural 2D spatial grid to convolve over. **Update: the actual spec's §3.2 preprocessing recipe explicitly lists "LSTM/GRU/TCN/Transformer" as example sequence models for the K=12 forecasting sub-task (P2)**, so a 1D Temporal Convolutional Network (TCN) or a small Transformer are spec-suggested, not just a stretch — but neither TCN nor Transformer is taught in the 19 course files (2D CNN ≠ TCN, and no attention/Transformer lecture exists at all). LSTM/GRU (Week15) remain the only *course-taught* sequence models, and are a reasonable default; TCN/Transformer are legitimate "breadth of methods" additions (worth real grading points per §6) if time allows, at the cost of needing to self-teach/explain them independently of the slides. |
| Reinforcement Learning (Week16: Q-learning, DQN, Policy Gradient) | None of Problems 1–5 involve sequential decision-making, an agent, or a reward signal. There is no natural application here despite RL being thoroughly covered in the course. |
| Independent Component Analysis (ICA) | Mentioned briefly at the end of the PCA deck (Week10) as a contrast to PCA, not developed further. Not required by the Problem 3 description (which asks for PCA/KPCA and optionally autoencoder/VAE). Skip unless there's a specific reason to want statistically independent (vs. merely uncorrelated) components. |
| K-means / GMM as *required* deliverables | Genuinely taught in depth (Week08–09), but neither Problem 1–5 as described actually calls for a clustering deliverable. They remain useful only as optional exploratory tools (e.g., visually grouping "generation regimes" before defining Problem 1's regime labels), not as a graded component. |

## Quick per-problem model shortlist (cross-referenced with `EXPERIMENT_PLAN.md`)

- **Problem 1 (classification):** Logistic Regression / Naive Bayes / kNN (baselines) → Decision Tree / Random Forest / Linear SVM (classical) → MLP (deep). All taught.
- **Problem 2 (regression):** Linear Regression (baseline) → Random Forest Regression / kNN Regression (classical) → MLP (deep, same-city/cross-city) → LSTM/GRU (deep, sequence K=12, taught) → optionally TCN/Transformer (deep, sequence K=12, spec-suggested but not taught — good "breadth" addition if time allows).
- **Problem 3 (dim. reduction):** PCA → Kernel PCA (both taught) → Autoencoder (extension of taught MLP, not itself taught) → t-SNE/UMAP for visualization only (not taught, tool-only).
- **Problem 4 (semi-supervised):** Supervised baseline (any P1 classifier) → pseudo-labeling/self-training (primary method, spec-compliant, simple) → graph-based label propagation (taught, breadth addition). Decision resolved — see `TEACHER_EXPECTATIONS.md`.
- **Problem 5 (transfer learning):** Zero-shot (evaluate a Davis-trained model directly on Amherst) → few-shot (fine-tune on a small Amherst sample) — general practice, not covered by any lecture.
