from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .common import *

class GRURegressor(nn.Module):
    def __init__(self,n_features,hidden=64):
        super().__init__(); self.gru=nn.GRU(n_features,hidden,batch_first=True); self.fc=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.fc(self.gru(x)[0][:,-1])

def run_sequence(sheets,city='Davis',fast=False,seed=42,K=12):
    set_seed(seed); df=choose_sheet(sheets,city); d=engineer_features(df).dropna(subset=[TARGET]).copy()
    cols=[c for c in d.columns if c not in TIME_COLS+['Timestamp','City',TARGET,'DayOfYear','HourFrac','Cloud Type']]
    vals=d[cols].astype(float).replace([np.inf,-np.inf],np.nan).interpolate().bfill().ffill().values
    y=d[TARGET].values.astype(float)
    cut=int(.8*len(d)); mean=vals[:cut].mean(0); std=vals[:cut].std(0)+1e-8; vals=(vals-mean)/std
    def windows(a,b):
      X=[];Y=[]
      for i in range(a+K,b): X.append(vals[i-K:i]); Y.append(y[i])
      return np.asarray(X,np.float32),np.asarray(Y,np.float32)
    Xtr,Ytr=windows(0,cut); Xte,Yte=windows(max(0,cut-K),len(d));
    if fast and len(Xtr)>3000:
        sel=np.linspace(0,len(Xtr)-1,3000).astype(int); Xtr,Ytr=Xtr[sel],Ytr[sel]

    ymean=Ytr.mean(); ystd=Ytr.std()+1e-8; Ytrn=(Ytr-ymean)/ystd
    dev='cuda' if torch.cuda.is_available() else 'cpu'; model=GRURegressor(Xtr.shape[-1]).to(dev); opt=torch.optim.Adam(model.parameters(),lr=.001); loss=nn.MSELoss(); epochs=5 if fast else 40
    xb=torch.tensor(Xtr,device=dev); yb=torch.tensor(Ytrn[:,None],device=dev)
    for _ in range(epochs):
      model.train(); opt.zero_grad(); pred=model(xb); l=loss(pred,yb); l.backward(); opt.step()
    model.eval();
    with torch.no_grad(): pred=model(torch.tensor(Xte,device=dev)).cpu().numpy().ravel()*ystd+ymean
    m=regression_metrics(Yte,pred); append_results([{'problem':2,'setup':'sequence_GRU','city':city,'model':'GRU','seed':seed,**m}]); save_prediction_plot(Yte,pred,f'Problem 2 sequence GRU - {city}','figures/p2_sequence_gru.png'); save_json(m,'results/p2_sequence_gru.json'); print(m)
