from __future__ import annotations
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from .common import *
from .problem2 import feature_frame

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data.xlsx'); ap.add_argument('--source',default='Davis'); ap.add_argument('--target',default='Amherst'); ap.add_argument('--fast',action='store_true'); args=ap.parse_args()
    sheets=load_workbook(args.data); src=choose_sheet(sheets,args.source,6); tgt=choose_sheet(sheets,args.target,3)
    Xs,ys=feature_frame(src); Xt,yt=feature_frame(tgt)
    prep=make_preprocessor(Xs); A=prep.fit_transform(Xs); B=prep.transform(Xt)
    rows=[]
    source_mean,source_std=ys.mean(),ys.std()+1e-8
    source_model=HistGradientBoostingRegressor(max_iter=60 if args.fast else 250,learning_rate=.06,random_state=42)
    source_model.fit(A,(ys-source_mean)/source_std)
    source_pred=source_model.predict(B)*source_std+source_mean
    rows.append({'problem':5,'source':args.source,'target':args.target,'setup':'zero_shot',**regression_metrics(yt,source_pred)})
    rng=np.random.default_rng(42)
    for k in [10,50,100]:
        idx=rng.choice(len(B),min(k,len(B)),replace=False); Xk=B[idx]; yk=yt.iloc[idx].values
        baseline=HistGradientBoostingRegressor(max_iter=40 if args.fast else 180,learning_rate=.06,random_state=100+k)
        baseline.fit(Xk,yk); bp=baseline.predict(B)
        rows.append({'problem':5,'source':args.source,'target':args.target,'setup':f'few_shot_{k}',**regression_metrics(yt,bp)})
        # Transfer: source model supplies a learned target-domain prior; target model learns only the residual from k labels.
        residual=yk-source_pred[idx]
        adapt=Ridge(alpha=10).fit(Xk,residual)
        tp=source_pred+adapt.predict(B)
        rows.append({'problem':5,'source':args.source,'target':args.target,'setup':f'transfer_{k}',**regression_metrics(yt,tp)})
    append_results(rows); save_json(rows,'results/p5_transfer.json'); print(rows)
if __name__=='__main__': main()
