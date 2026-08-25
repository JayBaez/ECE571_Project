# Course Context — ECE571 Machine Learning

This is a summary of the 19 course files found in `course/` (18 PDFs + 1 `.pptx`,
Week01 through Week16). It is based on direct extraction of the text content of
every file — not filenames or assumptions. Where a topic below is listed, it is
because it was actually found in the slides. Course appears to be from WPI
(Worcester Polytechnic Institute) — Week01 references WPI Matlab licensing.

Note on extraction quality: several decks (esp. Week03b Linear Regression,
Week14 CNN) render most equations as embedded vector graphics / LaTeX images
rather than text, so exact formulas didn't always extract cleanly. The topic
coverage and worked numeric examples below are still reliable; for exact
equation notation, the original PDFs are the source of truth.

---

## Week-by-week summary

**Week01 — ML Introduction.** Data+Machine+Learning framing. Three learning
types: supervised (classification & regression), unsupervised (clustering &
dimensionality reduction), reinforcement learning. Slide explicitly says RL
is "interesting but no time to be covered in this class" — **note:** a full
Week16 RL deck exists anyway (see below); treat RL as present in the
materials but likely not central to this project. Also introduces feature
engineering as manual preprocessing before modeling.

**Week02 — ML Basic Concepts.** Train/test split (hold-out, k-fold CV),
bootstrapping, MSE, accuracy/error rate, confusion matrix (TP/FP/TN/FN),
precision, recall, sensitivity/specificity, F1 score, ROC curve and AUC,
overfitting.

**Week03(a) — Linear Classification.** Perceptron (Rosenblatt), the
perceptron update rule, proof of convergence for linearly separable data,
general linear classifiers (affine decision boundary, multi-class /
Voronoi regions), the XOR limitation of linear classifiers (Minsky &
Papert), and logistic regression derived as a "differentiable perceptron"
(replacing `sign` with `tanh`/sigmoid so it's gradient-descent trainable).
Includes a full gradient derivation comparing perceptron vs. logistic
regression updates.

**Week03(b) — Linear Regression.** Least-squares linear model, closed-form
(normal equation) solution, multivariable linear regression, log-linear
regression, using nonlinear basis functions while keeping the model
"linear in the coefficients," batch vs. stochastic vs. mini-batch gradient
descent, and an explicit note that **inputs should always be normalized**
before gradient descent (unequal scales harm convergence).

**Week04 — K Nearest Neighbors.** Instance-based ("lazy") learning,
Voronoi-cell intuition, kNN for classification and for regression (with
optional distance weighting), choosing k (rule of thumb k≈√N, prefer odd
k to avoid ties), distance metrics, and speeding up kNN with K-D trees /
inverted lists. Explicit pros/cons and curse-of-dimensionality note.

**Week05 — Linear SVM.** Margin-based motivation, hard-margin SVM derived
as a constrained optimization (quadratic program), support vectors
defined precisely, soft-margin SVM with slack variables and the `C`
hyperparameter, and a pointer toward kernel tricks/dual formulation for
the nonlinear case (mentioned but not worked through in this deck).

**Week06(a) — Decision Trees.** ID3 top-down induction, entropy and
information gain (fully worked "Play Tennis" example), Gini impurity,
overfitting and pre-/post-pruning (subtree replacement), continuous
attribute handling via thresholds, missing-value handling, and a summary
naming ID3, CART, and C4.5 as the classic tree variants.

**Week06(b) — Bagging and Random Forest.** Bias/variance framing of why
decision trees generalize poorly alone (high variance), ensemble
aggregation (uniform and weighted averaging/voting; stacking mentioned
but explicitly "won't talk about this in this course"), bootstrapping,
bagging, out-of-bag (OOB) error estimation, and random forest (bootstrap
+ random feature subset per split).

**Week07 — Naive Bayes Classifier.** Bayes' rule, generative vs.
discriminative classifiers, the naive conditional-independence
assumption, MLE parameter estimation, Laplace/additive smoothing for
zero-count problems, Gaussian Naive Bayes for continuous features (fully
worked numeric example on the "Play Tennis" data), and missing-data
handling.

**Week08 — K-Means Clustering (start of unsupervised learning unit).**
Clustering motivation and applications, partitioning vs. hierarchical vs.
model-based vs. density-based clustering, the k-means algorithm and
optimization objective (distortion/SSE), sensitivity to initialization,
the elbow method for choosing k, and k-means++ seeding.

**Week09 — Gaussian Mixture Models.** Reviews k-means as hard assignment,
then motivates GMM as soft assignment. Full derivation of the GMM
log-likelihood and the EM algorithm (E-step computes responsibilities
`γ_ik`, M-step re-estimates `μ_k`, `Σ_k`, `π_k`). Practical notes: usually
initialize EM from k-means; can converge to a local optimum.

**Week10 — Principal Component Analysis.** Both the maximum-variance and
minimum-reconstruction-error views of PCA, derivation via the
covariance-matrix eigen-decomposition, explained variance / choosing the
number of components, PCA's relationship to the SVD, **Kernel PCA**
(derivation via the kernel/Gram matrix trick), and a short mention of
**ICA** as a contrast to PCA. No mention of autoencoders, VAEs, t-SNE, or
UMAP anywhere in this deck or elsewhere in the course materials.

**Week11 — Regularization.** L1 vs. L2 regularization (intuition:
L1 induces sparsity/feature selection, L2 is smooth), regularized cost
function, regularized linear regression (gradient descent and normal
equation, including the non-invertibility case), regularized logistic
regression, and deep-learning-specific regularization: L2 weight decay,
dropout ("inverted dropout," disabled at test time), data augmentation,
and early stopping.

**Week12 — Semi-Supervised Learning.** This deck is more advanced/academic
than the others (Maria-Florina Balcan's CMU slides). Explicitly covers
three named families with citations: **Transductive SVM** (Joachims '99,
maximizes margin over labeled + unlabeled data, NP-hard exactly, solved
heuristically by flipping unlabeled-point labels), **Co-training** (Blum
& Mitchell '98, two independent "views" of the features, each classifier
labels confident unlabeled points for the other), and **graph-based
methods** (build a similarity graph over labeled + unlabeled points, then
min-cut / soft-cut / spectral partitioning to propagate labels).
**Pseudo-labeling / self-training (train on labeled data, predict on
unlabeled data, add high-confidence predictions back to the training set)
is NOT explicitly named anywhere in this deck** even though it's a very
common, simple SSL technique in practice — see `ML_METHOD_MAP.md` for how
this affects Problem 4.

**Week13 — Multilayer Perceptron.** Motivation from perceptron's
limitations, one-hidden-layer architecture, softmax + cross-entropy for
multi-class classification (worked through the Iris dataset, one-hot
encoding), MSE as an alternative loss, and a fully worked forward-pass /
backward-pass / weight-update numeric example for a small 2-class MLP by
hand. Explicitly names MLP's limitations: "black box," non-convex
optimization / local minima.

**Week14 — Convolutional Neural Networks (Part 1 of what looks like a
2-part unit — only Part 1 was provided).** Selected slides from Sebastian
Raschka's deep learning course. Traditional (hand-engineered feature)
approaches vs. CNNs, the two core CNN ideas — **sparse connectivity** and
**parameter sharing** — LeNet-5 architecture (LeCun et al. 1998) with
conv/subsampling/fully-connected layers, parameter counting for conv vs.
fully-connected layers (motivating why CNNs scale to large images), and
feature visualization via the deconvnet approach (Zeiler & Fergus 2014).
Also references a specific research application (ordinal-regression CNNs
— CORAL/OR-CNN — for age estimation from face images) as a worked example
of a CNN-based *regression*-flavored task, which is a useful analogy for
this project's PV power regression problem.

**Week15(a) — Recurrent Neural Networks.** Motivation (variable-length
sequential data), the vanilla RNN cell and forward pass, weight sharing
across time ("unrolling"), applications (sentiment classification, image
captioning, language modeling), input/output topology taxonomy
(single-single, single-multiple, multiple-single, multiple-multiple —
this project's **sequence forecasting is a multiple-to-single or
multiple-to-multiple problem**), backpropagation through time (BPTT), and
the vanishing/exploding gradient problem (cites Pascanu et al. 2013).

**Week15(b) — Long Short-Term Memory (`.pptx`, not a PDF).** Builds
directly on Week15(a). Covers the vanishing-gradient motivation for gated
memory, the LSTM cell in detail (forget/input/output gates, cell state
`C_t`, why sigmoid/tanh are used), a side-by-side "RNN vs. LSTM"
comparison, peephole LSTM, **GRU** (as the simpler two-gate alternative
that merges cell and hidden state), Highway Networks / Residual Networks
(as a non-recurrent analogy to LSTM's gating), Grid LSTM, and several
applied examples (neural machine translation, sequence-to-sequence chat
models, speech recognition, attention-based image captioning). This is
the most directly relevant deck for Problem 2's sequence-forecasting
sub-task (K=12).

**Week16 — Reinforcement Learning.** Deep Q-Learning / DQN, Double DQN,
and Policy Gradient (REINFORCE). Thorough deck, but **RL has no natural
application to this project's five problems** — none of Problems 1–5
involve sequential decision-making or reward signals. Included here for
completeness only; see `ML_METHOD_MAP.md` ("probably unnecessary").

---

## Cross-cutting terminology and metrics actually taught

- **Classification metrics:** accuracy, error rate, precision, recall,
  sensitivity, specificity, F1, confusion matrix, ROC/AUC.
- **Regression-adjacent concepts:** MSE, closed-form least squares,
  gradient descent (batch/stochastic/mini-batch), normalization of
  inputs.
- **Model selection / validation:** hold-out split, k-fold cross-validation,
  bootstrapping, OOB error, regularization (L1/L2), early stopping.
- **Not explicitly taught anywhere in the 19 files:** autoencoders, VAEs,
  t-SNE, UMAP, transfer learning (as a named topic), pseudo-labeling /
  self-training. These all appear in the project's Problem 3–5
  descriptions, so where the project needs them they'll have to be
  introduced as natural extensions of taught material (e.g., an
  autoencoder is "just" the MLP architecture from Week13 trained with an
  unsupervised reconstruction loss) rather than pulled from a lecture.
  See `ML_METHOD_MAP.md` for the full breakdown.

## Course sequencing note

The order of the files (Week01 → Week16) roughly follows: ML basics →
linear models → classical supervised methods (kNN, SVM, trees,
ensembles, Naive Bayes) → unsupervised methods (k-means, GMM, PCA) →
regularization → semi-supervised learning → neural networks (MLP → CNN →
RNN/LSTM) → reinforcement learning. This project's Problems 1–5 line up
reasonably well against this sequence (classification → regression →
dimension reduction → semi-supervised → transfer learning), except that
transfer learning itself is never named as a topic.
