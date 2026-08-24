from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from .common import *

def feature_frame(df):
    d=engineer_features(df,add_lags=True).dropna(subset=[TARGET,'OutputPower_lag1','OutputPower_lag2'])
    drop=TIME_COLS+['Timestamp','City',TARGET,'DayOfYear','HourFrac']
    return d.drop(columns=[c for c in drop if c in d]), d[TARGET]

def fit_eval(Xtr,ytr,Xte,yte,fast=False):
    models={
      'ridge':Ridge(alpha=10),
      'random_forest':RandomForestRegressor(n_estimators=25 if fast else 300,random_state=42,n_jobs=-1,max_features=.8),
      'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=40 if fast else 300,learning_rate=.06,random_state=42),
      'mlp':MLPRegressor(hidden_layer_sizes=(128,64),max_iter=40 if fast else 300,random_state=42,early_stopping=True)
    }
    rows=[]; best=None
    for name,model in models.items():
      pipe=Pipeline([('prep',make_preprocessor(Xtr)),('model',model)])
      pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); m=regression_metrics(yte,pred)
      row={'model':name,**{k:float(v) for k,v in m.items()}}; rows.append(row)
      if best is None or m['rmse']<best[0]: best=(m['rmse'],name,pred,m)
    return rows,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data.xlsx'); ap.add_argument('--city',default='Davis'); ap.add_argument('--mode',choices=['same_city','cross_city','sequence'],default='same_city'); ap.add_argument('--fast',action='store_true'); args=ap.parse_args()
    sheets=load_workbook(args.data)
    if args.mode=='same_city':
      df=choose_sheet(sheets,args.city); X,y=feature_frame(df); cut=int(.8*len(X)); rows,best=fit_eval(X.iloc[:cut],y.iloc[:cut],X.iloc[cut:],y.iloc[cut:],args.fast)
      for r in rows:r.update(problem=2,setup='same_city',city=args.city)
      append_results(rows); save_prediction_plot(y.iloc[cut:],best[2],f'Problem 2 {args.city} best: {best[1]}',f'figures/p2_{args.city}_prediction.png')
      save_json({'best_model':best[1],'metrics':best[3],'rows':rows},f'results/p2_{args.city}_same_city.json')
      print(rows)
    elif args.mode=='cross_city':
      source=choose_sheet(sheets,'Davis',6); Xs,ys=feature_frame(source); Xs_tr=Xs; ys_tr=ys
      rows=[]
      for target in ['Huron','SantaBarbara','LaJolla']:
        dt=choose_sheet(sheets,target,6); Xt,yt=feature_frame(dt); rr,best=fit_eval(Xs_tr,ys_tr,Xt,yt,args.fast)
        for r in rr:r.update(problem=2,setup='cross_city',source='Davis',city=target)
        rows.extend(rr)
      append_results(rows); save_json(rows,'results/p2_cross_city.json'); print(rows)
    else:
      print('Sequence mode: use the dedicated GRU/LSTM implementation in sequence.py')
      from .sequence import run_sequence
      run_sequence(sheets,args.city,args.fast)

if __name__=='__main__': main()
