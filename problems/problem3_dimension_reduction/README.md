# Problem 3: Dimension Reduction

Compresses the feature space to d = 2, 5, and 10 dimensions, then checks how much that hurts the Problem 1 classifier and the Problem 2 regressor.

- **PCA** (linear) and a small **autoencoder** (nonlinear), both fit on training features only, with no label access.
- Downstream: sky-condition classification (random forest) and Output Power regression (gradient boosting), on raw features vs. each reduced representation.

## Files
- `features.py`: the single feature set used for both downstream tasks.
- `reduction.py`: PCA wrappers and the autoencoder.
- `run_experiments.py`: explained variance, reconstruction error, downstream comparison, 2-D visualizations (PCA, autoencoder, t-SNE), and a small feature ablation.

## Run
```bash
python problems/problem3_dimension_reduction/run_experiments.py
```
Run from the repo root. The raw dataset is not stored in the repo: place `Further Consolidated Data, HnL.xlsx` at `course/Further Consolidated Data, HnL.xlsx` first.

## Output
`results/problem3/` and `figures/problem3/`.

## Headline result
Raw features beat every PCA and autoencoder version at every dimension (0.743 balanced accuracy raw vs. 0.661 for the best reduced model).
