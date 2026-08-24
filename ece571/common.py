from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CITY_ALIASES = {
    'Amhst': 'Amherst', 'Amherst': 'Amherst', 'Davis': 'Davis',
    'Huron': 'Huron', 'Snt.Barb': 'SantaBarbara', 'SantaBarbara': 'SantaBarbara',
    'LaJolla': 'LaJolla', 'La Jolla': 'LaJolla'
}

BASE_NUMERIC = [
    'DHI','DNI','GHI','Clearsky DHI','Clearsky DNI','Clearsky GHI',
    'Dew Point','Solar Zenith Angle','Surface Albedo','Wind Speed',
    'Precipitable Water','Relative Humidity','Temperature','Pressure'
]
TIME_COLS = ['Year','Month','Day','Hour','Minute']
TARGET = 'Output Power'


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def load_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None)
    out = {}
    for raw_name, df in sheets.items():
        df = df.dropna(axis=1, how='all').copy()
        city_raw = raw_name.split(' ')[0]
        city = CITY_ALIASES.get(city_raw, city_raw)
        df['City'] = city
        df['Timestamp'] = pd.to_datetime(df[TIME_COLS])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        out[raw_name] = df
    return out


def choose_sheet(sheets: dict[str, pd.DataFrame], city: str, years: int | None = None) -> pd.DataFrame:
    candidates = [(n,d) for n,d in sheets.items() if d['City'].iloc[0] == city]
    if not candidates: raise ValueError(f'No sheet found for city={city}')
    if years == 6:
        six = [(n,d) for n,d in candidates if "'11-'16" in n]
        if six: return six[0][1].copy()
    if years == 3:
        three = [(n,d) for n,d in candidates if "'14-'16" in n or "'18-'20" in n]
        if three: return three[0][1].copy()
    return max(candidates, key=lambda x: len(x[1]))[1].copy()


def engineer_features(df: pd.DataFrame, add_lags: bool = False, target_lags: Iterable[int] = (1,2)) -> pd.DataFrame:
    d = df.copy()
    denom = d['Clearsky GHI'].replace(0, np.nan)
    d['ClearSkyIndex'] = (d['GHI'] / denom).clip(0, 2).fillna(0)
    d['HourFrac'] = d['Hour'] + d['Minute']/60.0
    d['HourSin'] = np.sin(2*np.pi*d['HourFrac']/24)
    d['HourCos'] = np.cos(2*np.pi*d['HourFrac']/24)
    d['DayOfYear'] = d['Timestamp'].dt.dayofyear
    d['DoySin'] = np.sin(2*np.pi*d['DayOfYear']/365.25)
    d['DoyCos'] = np.cos(2*np.pi*d['DayOfYear']/365.25)
    d['WindDirSin'] = np.sin(np.deg2rad(d['Wind Direction']))
    d['WindDirCos'] = np.cos(np.deg2rad(d['Wind Direction']))
    if add_lags:
        for lag in target_lags:
            d[f'OutputPower_lag{lag}'] = d[TARGET].shift(lag)
    return d


def chronological_split(df: pd.DataFrame, frac: float = .8):
    n = len(df); cut = int(n*frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def make_preprocessor(X: pd.DataFrame, categorical=('Cloud Type',)) -> ColumnTransformer:
    cats = [c for c in categorical if c in X.columns]
    nums = [c for c in X.columns if c not in cats]
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats)
    ], remainder='drop', verbose_feature_names_out=False)


def classification_metrics(y, pred):
    return {
        'accuracy': accuracy_score(y,pred),
        'balanced_accuracy': balanced_accuracy_score(y,pred),
        'report': classification_report(y,pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y,pred).tolist()
    }


def regression_metrics(y, pred):
    rmse = mean_squared_error(y,pred)**0.5
    mae = mean_absolute_error(y,pred)
    scale = np.mean(np.abs(y))
    return {'rmse': rmse, 'mae': mae, 'nrmse': rmse/scale if scale else np.nan}


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f: json.dump(obj,f,indent=2,default=float)


def append_results(rows, path='results/results.csv'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(path): df.to_csv(path,index=False,mode='a',header=False)
    else: df.to_csv(path,index=False)


def save_confusion(cm, labels, title, path):
    import seaborn as sns
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title); plt.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); plt.savefig(path,dpi=180); plt.close()


def save_prediction_plot(y, pred, title, path):
    plt.figure(figsize=(10,4)); plt.plot(np.asarray(y), label='Truth'); plt.plot(np.asarray(pred), label='Prediction', alpha=.8)
    plt.title(title); plt.xlabel('Test sample'); plt.ylabel('Output Power (kW)'); plt.legend(); plt.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); plt.savefig(path,dpi=180); plt.close()
