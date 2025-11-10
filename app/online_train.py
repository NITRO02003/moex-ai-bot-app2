
# app/online_train.py
from __future__ import annotations
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .features import build_features_15m, FINAL_FEATURE_SET
from .utils import ensure_dtindex_utc

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def build_xy(prices: pd.DataFrame, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    close = prices["close"].astype(float)
    feats = build_features_15m(prices)
    feats = feats[[c for c in feats.columns if c in FINAL_FEATURE_SET]]
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    y = np.where(fwd_ret > 0, 1, 0).astype("int8")
    X = feats.iloc[:-horizon].copy() if horizon>0 else feats.copy()
    y = pd.Series(y, index=feats.index).iloc[:-horizon] if horizon>0 else pd.Series(y, index=feats.index)
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0).astype("float32")
    return X, y

def partial_fit_model(X: pd.DataFrame, y: pd.Series, model_path: str = os.path.join(MODELS_DIR, "ai_strategy.pkl")) -> None:
    from sklearn.linear_model import SGDClassifier
    import joblib
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            model = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)
    else:
        model = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)
    if not hasattr(model, "classes_"):
        model.partial_fit(X.values, y.values, classes=np.array([0,1], dtype=np.int64))
    else:
        model.partial_fit(X.values, y.values)
    joblib.dump(model, model_path)

def online_train_from_dataframes(data: Dict[str, pd.DataFrame], horizon: int = 1) -> None:
    frames_X = []
    frames_y = []
    for sym, df in data.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        df = ensure_dtindex_utc(df)
        X, y = build_xy(df, horizon=horizon)
        if len(X) and len(y):
            frames_X.append(X)
            frames_y.append(y)
    if not frames_X:
        return
    X_all = pd.concat(frames_X).sort_index()
    y_all = pd.concat(frames_y).reindex(X_all.index).fillna(0).astype(int)
    partial_fit_model(X_all, y_all)
