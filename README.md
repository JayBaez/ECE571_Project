# ECE571 PV Power Prediction Project

A reproducible Python implementation of all five project problems for the ECE571 Machine Learning Course Project: **Photovoltaic (Solar) Power Prediction on Large-Scale Spatiotemporal Data**.

## What is included

- **Problem 1:** supervised classification (sky condition and generation regime)
- **Problem 2:** supervised regression (same-city, cross-city, and sequence forecasting)
- **Problem 3:** PCA and autoencoder dimension reduction with downstream evaluation
- **Problem 4:** semi-supervised learning via pseudo-labeling
- **Problem 5:** transfer learning from Davis to Amherst with zero-shot and few-shot baselines
- Common chronological splitting, train-only preprocessing, feature engineering, metrics, plots, seeds, and result logging
- Each problem is a standalone runnable Python module

## Dataset

The supplied workbook is copied to `data.xlsx`. The implementation discovers sheets by city and year span, so it does not depend on hard-coded row counts.

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Run everything

```bash
python run_all.py --data data.xlsx --fast
```

`--fast` uses smaller models/epochs for a quick end-to-end run. Remove it for the fuller experiments.

## Run individual problems

```bash
python -m ece571.problem1 --data data.xlsx --city Davis --task sky
python -m ece571.problem1 --data data.xlsx --city Davis --task regime

python -m ece571.problem2 --data data.xlsx --city Davis --mode same_city
python -m ece571.problem2 --data data.xlsx --mode cross_city
python -m ece571.problem2 --data data.xlsx --city Davis --mode sequence

python -m ece571.problem3 --data data.xlsx --city Davis
python -m ece571.problem4 --data data.xlsx --city Davis --task classification
python -m ece571.problem5 --data data.xlsx --source Davis --target Amherst
```

Outputs are written under `results/` and `figures/`.

## Reproducibility

- Default seeds: 7, 21, 42
- Chronological 80/20 split for same-city experiments
- Scalers/encoders are fit on training data only
- Test labels are never used to train preprocessing or models
- Cross-city evaluation has zero target overlap for zero-shot experiments

## Project structure

```text
ece571_pv_ml_project/
├── data.xlsx
├── requirements.txt
├── README.md
├── run_all.py
├── ece571/
│   ├── common.py
│   ├── problem1.py
│   ├── problem2.py
│   ├── problem3.py
│   ├── problem4.py
│   └── problem5.py
├── tests/
├── results/
└── figures/
```

## Notes on the dataset

The workbook's Amherst sheet is named `Amhst 5hr-daily '18-'20`, while the course text calls the city Amherst. The loader maps this sheet to the canonical city name `Amherst`.

`Cloud Type` is treated as categorical. Output Power is never used as an input to the Problem 1 sky classifier. GHI/Clearsky GHI are excluded from sky-classifier inputs to prevent the label definition from leaking into the features.
