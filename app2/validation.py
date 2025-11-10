
from __future__ import annotations
import math, numpy as np, pandas as pd
from datetime import datetime
from . import models as M, metrics as MX

def time_splits(df: pd.DataFrame, n_folds: int = 6):
    # split by month chunks
    if df.empty: return []
    by_month = df.groupby([df.index.year, df.index.month])
    months = sorted({(y,m) for y,m in zip(df.index.year, df.index.month)})
    if len(months) < n_folds+1:
        n_folds = max(1, len(months)-1)
    folds = []
    for i in range(n_folds):
        train_months = months[: i+1]
        test_month = months[i+1] if i+1 < len(months) else None
        if test_month is None: break
        tr_idx = df.index.map(lambda t: (t.year, t.month) in train_months)
        te_idx = df.index.map(lambda t: (t.year, t.month) == test_month)
        folds.append((df.index[tr_idx], df.index[te_idx]))
    return folds

def walk_forward(prices: pd.DataFrame, horizon: int=1):
    X, y = M.dataset(prices, horizon=horizon)
    bundle = M.load()
    results = []
    for tr_idx, te_idx in time_splits(X, n_folds=6):
        Xtr, ytr = X.loc[tr_idx], y.loc[tr_idx]
        Xte, yte = X.loc[te_idx], y.loc[te_idx]
        # fit fresh models on train
        info = M.fit_offline_multi({'_': prices.loc[tr_idx]}, horizon=horizon)
        bundle = M.load()  # reload
        p = M.predict_proba(bundle, Xte, prices['close'].astype(float))
        # build equity proxy: long with size=1 for evaluation of AUC-like metrics
        pnl = ( (p - 0.5).clip(lower=0) * yte.replace({0:-1,1:1}) ).rename('pnl')  # rough proxy
        eq = pnl.cumsum() + 1.0
        results.append({'auc_like': M.auc(bundle, Xte, yte, prices['close']), 'ret_like': float(eq.iloc[-1]-1.0)})
    return results
