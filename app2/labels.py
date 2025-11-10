
from __future__ import annotations
import numpy as np, pandas as pd

def y_updown(close: pd.Series, horizon: int=1) -> pd.Series:
    fwd = close.pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype('int8')

def y_return(close: pd.Series, horizon: int=1) -> pd.Series:
    return close.pct_change(horizon).shift(-horizon)

def clean_xy(X: pd.DataFrame, y: pd.Series):
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0).astype('float32')
    y = y.reindex(X.index).fillna(0).astype('int8')
    return X, y
