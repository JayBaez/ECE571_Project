from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from .common import *

SKY_BINS = [-np.inf, .4, .85, np.inf]
SKY_LABELS = ['Overcast','Partly cloudy','Clear']

def make_task(df, task):
    d = engineer_features(df)
    if task == 'sky':
        d['label'] = pd.cut(d['ClearSkyIndex'], bins=SKY_BINS, labels=SKY_LABELS).astype(str)
        # GHI and all clear-sky GHI information are excluded to prevent direct label leakage.
        drop = TIME_COLS + ['Timestamp','City',TARGET,'label','GHI','Clearsky GHI','ClearSkyIndex','DayOfYear','HourFrac']
    elif task == 'regime':
        q = d[TARGET].quantile([1/3,2/3]).values
        d['label'] = pd.cut(d[TARGET], bins=[-np.inf,*q,np.inf], labels=['Low','Medium','High']).astype(str)
        drop = TIME_COLS + ['Timestamp','City',TARGET,'label']
    else: raise ValueError('task must be sky or regime')
    X = d.drop(columns=[c for c in drop if c in d])
    y = d['label']
    return X,y

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data.xlsx'); ap.add_argument('--city',default='Davis'); ap.add_argument('--task',choices=['sky','regime'],default='sky'); ap.add_argument('--fast',action='store_true'); ap.add_argument('--seeds',default='7,21,42'); args=ap.parse_args()
    sheets=load_workbook(args.data); df=choose_sheet(sheets,args.city)
    X,y=make_task(df,args.task); le=LabelEncoder(); y_enc=le.fit_transform(y); Xt,Xv=X.iloc[:int(.8*len(X))],X.iloc[int(.8*len(X)):]; yt,yv=y_enc[:len(Xt)],y_enc[len(Xt):]
    models={
      'logistic': LogisticRegression(max_iter=1000),
      'random_forest': RandomForestClassifier(n_estimators=120 if args.fast else 300, random_state=42,n_jobs=-1,class_weight='balanced'),
      'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=80 if args.fast else 200, random_state=42),
      'mlp': MLPClassifier(hidden_layer_sizes=(64,32),max_iter=80 if args.fast else 250,random_state=42,early_stopping=True)
    }
    rows=[]; best=None
    for name,model in models.items():
      pipe=Pipeline([('prep',make_preprocessor(Xt)),('model',model)])
      pipe.fit(Xt,yt); pred=pipe.predict(Xv); m=classification_metrics(yv,pred)
      row={'problem':1,'city':args.city,'task':args.task,'model':name,'balanced_accuracy':m['balanced_accuracy'],'accuracy':m['accuracy']}
      rows.append(row)
      if best is None or m['balanced_accuracy']>best[0]: best=(m['balanced_accuracy'],name,pred,m)
    append_results(rows)
    save_confusion(best[3]['confusion_matrix'], list(le.classes_), f'Problem 1 {args.task}: {best[1]}', f'figures/p1_{args.city}_{args.task}_confusion.png')
    save_json({'best_model':best[1],'metrics':best[3]},f'results/p1_{args.city}_{args.task}.json')
    print(rows)

if __name__=='__main__': main()
