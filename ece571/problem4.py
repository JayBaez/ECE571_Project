from __future__ import annotations
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from .common import *

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data.xlsx'); ap.add_argument('--city',default='Davis'); ap.add_argument('--task',choices=['classification','regression'],default='classification'); ap.add_argument('--fast',action='store_true'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); set_seed(args.seed)
  sheets=load_workbook(args.data); df=choose_sheet(sheets,args.city); d=engineer_features(df).dropna(subset=[TARGET]); cut=int(.8*len(d)); tr=d.iloc[:cut].copy(); te=d.iloc[cut:].copy()
  drop=TIME_COLS+['Timestamp','City',TARGET,'DayOfYear','HourFrac']; Xtr=tr.drop(columns=[c for c in drop if c in tr]); Xte=te.drop(columns=[c for c in drop if c in te]);
  if args.task=='classification':
    q=tr[TARGET].quantile([1/3,2/3]).values; ytr=pd.cut(tr[TARGET],[-np.inf,*q,np.inf],labels=['Low','Medium','High']).astype(str).values; yte=pd.cut(te[TARGET],[-np.inf,*q,np.inf],labels=['Low','Medium','High']).astype(str).values; Model=HistGradientBoostingClassifier(max_iter=40 if args.fast else 150, random_state=args.seed)
  else: ytr=tr[TARGET].values; yte=te[TARGET].values; Model=HistGradientBoostingRegressor(max_iter=100 if args.fast else 250,random_state=args.seed)
  prep=make_preprocessor(Xtr); A=prep.fit_transform(Xtr); B=prep.transform(Xte); rng=np.random.default_rng(args.seed); rows=[]
  for frac in [.1,.3,.5]:
    n=max(2,int(frac*len(A))); idx=rng.choice(len(A),n,replace=False)
    sup=Model; sup.fit(A[idx],ytr[idx]); ps=sup.predict(B)
    if args.task=='classification': base=balanced_accuracy_score(yte,ps)
    else: base=regression_metrics(yte,ps)['nrmse']
    # Pseudo-label all unlabeled samples using the supervised seed model, then retrain with confidence-filtered labels.
    seed_model=Model; seed_model.fit(A[idx],ytr[idx]); pu=seed_model.predict(A)
    if args.task=='classification':
      proba=seed_model.predict_proba(A); conf=proba.max(1); keep=np.ones(len(A),bool); keep[idx]=False; keep &= conf>=np.quantile(conf[keep],.65) if keep.any() else keep
    else:
      keep=np.ones(len(A),bool); keep[idx]=False; keep &= np.abs(pu-np.mean(ytr[idx])) < 3*np.std(ytr[idx])+1e-6
    Xssl=np.vstack([A[idx],A[keep]]); yssl=np.concatenate([ytr[idx],pu[keep]]); ssl=Model; ssl.fit(Xssl,yssl); pred=ssl.predict(B)
    metric=balanced_accuracy_score(yte,pred) if args.task=='classification' else regression_metrics(yte,pred)['nrmse']
    rows.append({'problem':4,'city':args.city,'task':args.task,'labeled_fraction':frac,'supervised_metric':base,'ssl_metric':metric,'gain':metric-base})
  append_results(rows); save_json(rows,f'results/p4_{args.task}.json'); print(rows)

if __name__=='__main__': main()
