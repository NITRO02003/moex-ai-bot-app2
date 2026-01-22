from __future__ import annotations
import datetime as dt, pandas as pd
from . import data as D, models as M

def run(symbols, days: int = 3, horizon: int=1):
    end = dt.date.today().isoformat()
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    prices = {s: D.fetch_range(s, start, end) for s in symbols}
    for s, df in prices.items():
        D.save_csv(s, df)
    model = M.load_model()
    Xs, ys = [], []
    for s, df in prices.items():
        if df is None or df.empty: continue
        X, y = M.build_dataset(df, horizon=horizon)
        if len(X): Xs.append(X); ys.append(y)
    if not Xs: return
    X = pd.concat(Xs).sort_index(); y = pd.concat(ys).reindex(X.index).fillna(0).astype(int)
    M.partial_fit(model, X, y)
