# Problem 1: Supervised Classification

Predicts a discrete label from weather and irradiance features, for Davis and Amherst.

- **Sky-condition** (Clear / Partly Cloudy / Overcast): derived from the Clear-Sky Index (GHI / Clearsky GHI) with fixed thresholds. GHI and clear-sky GHI are excluded from the features because they define the label.
- **Generation-regime**: second classification target built in `targets.py`.

## Files
- `targets.py`: builds the two labels.
- `features.py`: which columns are allowed as features per task, plus the ablation feature groups.
- `models.py`: classical scikit-learn models and the MLP classifier.
- `run_experiments.py`: model sweep (majority, logistic regression, decision tree, random forest, gradient boosting, MLP; 3 seeds), small hyperparameter search, class-weighting comparison, feature ablation, confusion matrices and error analysis.

## Run
```bash
python problems/problem1_classification/run_experiments.py
```
Run from the repo root. The raw dataset is not stored in the repo: place `Further Consolidated Data, HnL.xlsx` at `course/Further Consolidated Data, HnL.xlsx` first.

## Output
`results/problem1/` (results CSV, saved models) and `figures/problem1/`.

## Headline result
Logistic regression with balanced class weights, 0.772 balanced accuracy on Davis sky-condition.
