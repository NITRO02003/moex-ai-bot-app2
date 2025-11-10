
from __future__ import annotations
import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from .paths import MODELS_DIR
from . import features as F, labels as L

MODEL_PATH = MODELS_DIR / 'ai_strategy.pkl'

def regime_by_vol(close: pd.Series, lb: int = 96) -> pd.Series:
    r = close.pct_change().rolling(lb).std()
    med = r.rolling(lb).median()
    return (r > med).rename('volatile')  # True = volatile

def dataset(prices: pd.DataFrame, horizon: int=1):
    X = F.build(prices)
    X = X[F.final_columns(X.columns)]
    y = L.y_updown(prices['close'].astype(float), horizon=horizon)
    X, y = L.clean_xy(X, y)
    m = X.index.intersection(y.index)
    return X.loc[m], y.loc[m]

def make_models(seed=42):
    base = SGDClassifier(loss='log_loss', alpha=1e-4, random_state=seed)
    rf = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=seed, n_jobs=-1)
    return base, rf

def fit_offline_multi(prices_by_symbol: dict[str, pd.DataFrame], horizon: int=1, path=MODEL_PATH):
    Xs, ys = [], []
    for sym, df in prices_by_symbol.items():
        if df is None or df.empty or 'close' not in df.columns: continue
        X, y = dataset(df, horizon=horizon)
        if len(X): Xs.append(X); ys.append(y)
    if not Xs: raise RuntimeError('no data to train')
    X = pd.concat(Xs).sort_index(); y = pd.concat(ys).reindex(X.index).fillna(0).astype(int)
    base, rf = make_models()
    base.partial_fit(X.values, y.values, classes=np.array([0,1]))
    rf.fit(X.values, y.values)
    joblib.dump({'base': base, 'rf': rf, 'cols': list(X.columns)}, path)
    return {'n': len(X), 'cols': len(X.columns)}

def load(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    base, rf = make_models()
    return {'base': base, 'rf': rf, 'cols': []}

def _proba(model, X: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(X.values)[:,1]
    except Exception:
        z = model.decision_function(X.values)
        return 1/(1+np.exp(-z))

def predict_proba(bundle, X: pd.DataFrame, close: pd.Series) -> pd.Series:
    cols = bundle.get('cols', list(X.columns))
    X = X[[c for c in cols if c in X.columns]].copy()
    # pad missing columns
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[cols]
    p_base = _proba(bundle['base'], X)
    p_rf   = _proba(bundle['rf'], X)
    reg = regime_by_vol(close).reindex(X.index).fillna(False)
    p = np.where(reg.values, 0.6*p_rf + 0.4*p_base, 0.8*p_base + 0.2*p_rf)
    return pd.Series(p, index=X.index, name='p_up')

def auc(bundle, X, y, close) -> float:
    p = predict_proba(bundle, X, close)
    idx = X.index.intersection(y.index)
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y.loc[idx], p.loc[idx]))
