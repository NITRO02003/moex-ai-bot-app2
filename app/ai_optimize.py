# app/ai_optimize.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
os.makedirs(OUT_DIR, exist_ok=True)

def auto_pick_thresholds(scores: pd.Series) -> float:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if s.empty: return 0.5
    return float(s.median())

def grid_from_series(series: pd.Series, steps: int = 10):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return []
    lo, hi = float(s.quantile(0.05)), float(s.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi: return []
    return list(np.linspace(lo, hi, steps))
