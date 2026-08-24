from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from .common import *

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data.xlsx'); ap.add_argument('--city',default='Davis'); ap.add_argument('--fast',action='store_true'); args=ap.parse_args()
  sheets=load_workbook(args.data); df=choose_sheet(sheets,args.city); d=engineer_features(df).dropna(subset=[TARGET]);
  drop=TIME_COLS+['Timestamp','City',TARGET,'DayOfYear','HourFrac']; X=d.drop(columns=[c for c in drop if c in d]);
  # Remove categorical feature for PCA; Cloud Type can be retained downstream but PCA needs numeric inputs.
  X=X.drop(columns=['Cloud Type'],errors='ignore'); cut=int(.8*len(X)); prep=Pipeline([('imp',__import__('sklearn').impute.SimpleImputer(strategy='median')),('scale',StandardScaler())]); Ztr=prep.fit_transform(X.iloc[:cut]); Zte=prep.transform(X.iloc[cut:]);
  maxd=min(20,Ztr.shape[1]); pca_full=PCA(n_components=maxd).fit(Ztr); cum=np.cumsum(pca_full.explained_variance_ratio_)
  plt.figure(figsize=(7,4)); plt.plot(range(1,maxd+1),cum,marker='o'); plt.xlabel('Components'); plt.ylabel('Cumulative explained variance'); plt.title('PCA explained variance'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig('figures/p3_explained_variance.png',dpi=180); plt.close()
  rows=[]
  for dcomp in [2,5,10]:
    if dcomp>Ztr.shape[1]: continue
    pca=PCA(n_components=dcomp).fit(Ztr); A=pca.transform(Ztr); B=pca.transform(Zte)
    reg=HistGradientBoostingRegressor(max_iter=150 if not args.fast else 60,random_state=42).fit(A,d[TARGET].iloc[:cut]); pred=reg.predict(B); m=regression_metrics(d[TARGET].iloc[cut:],pred)
    rows.append({'problem':3,'city':args.city,'representation':f'PCA-{dcomp}','explained_variance':float(pca.explained_variance_ratio_.sum()),**m})
  append_results(rows); save_json(rows,'results/p3_pca.json'); print(rows)

if __name__=='__main__': main()
