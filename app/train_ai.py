# train_ai.py
from __future__ import annotations
import os, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

from .utils import load_all, slice_by_date
from .config import config
from .features import build_features_10m, FINAL_FEATURE_SET, make_signal_labels, make_regime_labels

OUT_MODELS = "models"
os.makedirs(OUT_MODELS, exist_ok=True)
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)


def build_dataset():
    # Используем правильный путь к данным
    data_dir = getattr(config.bt_cfg, 'data_dir', 'data')  # Защита от отсутствия атрибута
    data = load_all(data_dir, config.symbols_cfg.symbols)

    frames = []
    for s, df in data.items():
        # Применяем фильтрацию по датам если указаны в конфиге
        start_date = getattr(config.bt_cfg, 'start_date', None)
        end_date = getattr(config.bt_cfg, 'end_date', None)
        if start_date or end_date:
            df = slice_by_date(df, start_date, end_date)

        feats = build_features_10m(df)
        feats["symbol"] = s
        feats["close"] = df["close"]
        frames.append(feats)

    if not frames:
        raise RuntimeError("No data")

    ds = pd.concat(frames).sort_index()
    X = ds[FINAL_FEATURE_SET].copy()
    y_sig = make_signal_labels(ds["close"])
    y_reg = make_regime_labels(ds["close"])

    # align
    mask = (~X.isna().any(axis=1)) & (~y_sig.isna()) & (~y_reg.isna())
    X = X[mask];
    y_sig = y_sig[mask];
    y_reg = y_reg[mask]

    return X.astype("float32"), y_sig.astype("int8"), y_reg.astype("int8")


def train_and_save():
    X, y_sig, y_reg = build_dataset()

    # Signal model (binary good/bad)
    clf_sig = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=None, class_weight={0: 1.0, 1: 1.5}, n_jobs=-1,
                                      random_state=42))
    ])
    clf_sig.fit(X, y_sig)
    joblib.dump(clf_sig, os.path.join(OUT_MODELS, "ai_strategy.pkl"))

    # Risk model (regime -> risk score mapping via proba of high vol)
    clf_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42))
    ])
    clf_reg.fit(X, y_reg)
    joblib.dump(clf_reg, os.path.join(OUT_MODELS, "ai_risk.pkl"))

    # quick metrics
    pred_sig = clf_sig.predict(X)
    pred_sig_p = clf_sig.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_sig, pred_sig_p)
    acc = accuracy_score(y_sig, pred_sig)
    with open(os.path.join(OUT_DIR, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"signal_auc": float(auc), "signal_acc": float(acc)}, f, indent=2)


if __name__ == "__main__":
    train_and_save()