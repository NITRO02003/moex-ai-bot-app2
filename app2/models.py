# models.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
from __future__ import annotations
import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from .paths import MODELS_DIR
from . import features as F, labels as L

MODEL_PATH = MODELS_DIR / 'ai_strategy.pkl'

def dataset(prices: pd.DataFrame, horizon: int = 1):
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

def fit_offline_multi(prices_by_symbol: dict[str, pd.DataFrame], horizon: int = 1, path=MODEL_PATH):
    Xs, ys = [], []
    for sym, df in prices_by_symbol.items():
        if df is None or df.empty or 'close' not in df.columns:
            continue
        try:
            X, y = dataset(df, horizon=horizon)
            if len(X) > 10:
                Xs.append(X); ys.append(y)
        except Exception as e:
            print(f"Warning: Failed to process {sym}: {e}")
            continue

    if not Xs:
        raise RuntimeError('no data to train')

    X = pd.concat(Xs).sort_index()
    X = X[~X.index.duplicated(keep='last')]
    y = pd.concat(ys).sort_index()
    y = y[~y.index.duplicated(keep='last')]
    y = y.reindex(X.index).fillna(0).astype(int)

    base, rf = make_models()
    base.partial_fit(X.values, y.values, classes=np.array([0, 1]))
    rf.fit(X.values, y.values)

    joblib.dump({'base': base, 'rf': rf, 'cols': list(X.columns)}, path)
    return {'n': len(X), 'cols': len(X.columns), 'symbols': list(prices_by_symbol.keys())}

def load(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Warning: Failed to load model from {path}: {e}")

    base, rf = make_models()
    return {'base': base, 'rf': rf, 'cols': []}

def partial_fit(bundle, X_new, y_new):
    """ИНКРЕМЕНТАЛЬНОЕ ОБУЧЕНИЕ - НОВАЯ ФУНКЦИЯ"""
    try:
        if hasattr(bundle.get('base'), 'partial_fit'):
            bundle['base'].partial_fit(X_new.values, y_new.values, classes=np.array([0, 1]))
        return True
    except Exception as e:
        print(f"Partial fit failed: {e}")
        return False

def load_model(path=MODEL_PATH):
    """АЛИАС ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ"""
    return load(path)

def _proba(model, X: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(X.values)[:, 1]
    except Exception:
        try:
            z = model.decision_function(X.values)
            return 1 / (1 + np.exp(-z))
        except Exception:
            return np.full(len(X), 0.5)

def predict_proba(bundle, X: pd.DataFrame, close: pd.Series) -> pd.Series:
    cols = bundle.get('cols', [])
    if not cols or X.empty:
        return pd.Series(0.5, index=X.index, name='p_up')

    try:
        available_cols = [c for c in cols if c in X.columns]
        if not available_cols:
            return pd.Series(0.5, index=X.index, name='p_up')

        X = X[available_cols].copy()
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols]

        p_base = _proba(bundle['base'], X)
        p_rf = _proba(bundle['rf'], X)

        rv = close.pct_change().rolling(96, min_periods=10).std()
        med = rv.rolling(96, min_periods=10).median()
        reg = (rv > med).reindex(X.index).fillna(False)

        p = np.where(reg.values, 0.6 * p_rf + 0.4 * p_base, 0.8 * p_base + 0.2 * p_rf)
        return pd.Series(p, index=X.index, name='p_up')
    except Exception as e:
        print(f"Warning: Predict failed: {e}")
        return pd.Series(0.5, index=X.index, name='p_up')

def auc(bundle, X, y, close) -> float:
    try:
        p = predict_proba(bundle, X, close)
        idx = X.index.intersection(y.index)
        if len(idx) == 0:
            return 0.5
        return float(roc_auc_score(y.loc[idx], p.loc[idx]))
    except Exception:
        return 0.5