# Problem 2: Supervised Regression (core task)

Predicts continuous Output Power (kW).

- **Same-city**: train and test within Davis, and within Amherst.
- **Cross-city zero-shot**: Davis model applied to Huron, Santa Barbara, and La Jolla with no retraining.
- **3-year vs 6-year** training ablation (Davis).
- **Sequence forecasting**: GRU using the previous 12 readings (about 6 hours) to predict the next one, compared with a persistence baseline.

## Files
- `features.py`: allowed feature columns.
- `models.py`: classical regressors and the MLP regressor.
- `sequence.py`: K=12 sliding windows and the GRU.
- `run_experiments.py`: all stages above, plus learning curve, predicted-vs-actual plots, and error analysis. Metrics: RMSE, MAE, R2, nRMSE.

## Run
```bash
python problems/problem2_regression/run_experiments.py
```
Run from the repo root. The raw dataset is not stored in the repo: place `Further Consolidated Data, HnL.xlsx` at `course/Further Consolidated Data, HnL.xlsx` first.

## Output
`results/problem2/` (results CSVs, hyperparameter search JSONs) and `figures/problem2/`. The current script does not write model files.

## Headline result
Gradient boosting on Davis, same-city: RMSE 15.17 kW, R2 0.953.
