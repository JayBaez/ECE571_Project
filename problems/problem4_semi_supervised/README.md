# Problem 4: Semi-Supervised Learning

Hides most of Davis's sky-condition labels (only 10%, 30%, or 50% stay labeled) and tests whether pseudo-labeling beats training on the labeled subset alone. Uses Problem 1's exact chronological train/test split.

## Files
- `pseudo_labeling.py`: self-training: train on the labeled subset, add confident predictions on the unlabeled pool as pseudo-labels, repeat.
- `run_experiments.py`: supervised baseline vs. pseudo-labeling at each label fraction and seed, label-efficiency curve, pseudo-label accuracy analysis, and an optional Label Spreading comparison.

## Run
```bash
python problems/problem4_semi_supervised/run_experiments.py
```
Run from the repo root. The raw dataset is not stored in the repo: place `Further Consolidated Data, HnL.xlsx` at `course/Further Consolidated Data, HnL.xlsx` first.

## Output
`results/problem4/` and `figures/problem4/`.

## Headline result
Semi-supervised gain was slightly negative at every label fraction. The pseudo-labels were accurate, but about 98% of the first round were the easy "Clear" class, so they added little new information.
