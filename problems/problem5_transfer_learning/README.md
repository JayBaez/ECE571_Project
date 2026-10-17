# Problem 5: Transfer Learning

Tests whether Davis (data-rich) can help Amherst (data-poor) for Output Power regression, all evaluated on the same held-out Amherst test set:

- **Zero-shot**: Davis-trained model applied directly.
- **Few-shot**: a fresh model trained only on k Amherst samples (k = 10, 50, 100).
- **Transfer**: the Davis-pretrained model fine-tuned on the same k samples.

## Files
- `models.py`: the transfer MLP (Input, Dense 128, Dense 64, Dense 1).
- `run_experiments.py`: all three comparisons, domain-shift analysis, a target-normalization ablation, and a layer-freezing ablation.

## Run
```bash
python problems/problem5_transfer_learning/run_experiments.py
```
Run from the repo root. The raw dataset is not stored in the repo: place `Further Consolidated Data, HnL.xlsx` at `course/Further Consolidated Data, HnL.xlsx` first.

## Output
`results/problem5/` (results CSV, saved models) and `figures/problem5/`.

## Headline result
Transfer beat few-shot at every k, most at k=10 (RMSE 30.03 kW, +21.4%). Zero-shot RMSE was 184.17 kW because of the plant-scale difference between the cities.
