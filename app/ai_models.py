
# app/ai_models.py
from __future__ import annotations
from typing import Optional
import os
import pandas as pd
from .logging_utils import get_logger

logger = get_logger("AI_Models")

class _JoblibLoader:
    def __init__(self, path: str):
        self.path = path
        self.model = None
        self._try_load()
    def _try_load(self):
        if not os.path.exists(self.path): return
        try:
            import joblib
            self.model = joblib.load(self.path)
        except Exception as e:
            logger.warning(f"joblib load failed: {e}; trying pickle")
            import pickle
            with open(self.path, "rb") as f:
                self.model = pickle.load(f)

class AIStrategyModel:
    def __init__(self, path: str = "models/ai_strategy.pkl", threshold: float = 0.52):
        self.loader = _JoblibLoader(path)
        self.threshold = threshold
    def available(self) -> bool:
        return self.loader.model is not None
    def predict_series(self, feats: pd.DataFrame) -> pd.Series:
        if feats is None or feats.empty or not self.available():
            logger.warning("AIStrategyModel: Invalid input or model not available")
            return pd.Series(0.0, index=feats.index if feats is not None else pd.DatetimeIndex([], tz="UTC"))
        m = self.loader.model
        try:
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(feats)
                p1 = proba[:,1] if proba.ndim==2 and proba.shape[1]>=2 else proba.ravel()
                sig = (p1 > self.threshold).astype("int8") - (p1 < 1-self.threshold).astype("int8")
                return pd.Series(sig.astype(float), index=feats.index)
            elif hasattr(m, "predict"):
                y = m.predict(feats)
                s = pd.Series(y, index=feats.index).clip(-1,1).astype(float)
                return s
        except Exception as e:
            logger.error(f"AIStrategyModel prediction failed: {e}")
            return pd.Series(0.0, index=feats.index)  # conservative zero
        return pd.Series(0.0, index=feats.index)

class AIRiskModel:
    def __init__(self, model_path: str = "models/ai_risk.pkl", file_path: str = "out/ai_risk_scores.csv"):
        self.loader = _JoblibLoader(model_path)
        self.file_path = file_path
        self.series: Optional[pd.Series] = None
        self._load_file()
    def _load_file(self):
        if os.path.exists(self.file_path):
            try:
                df = pd.read_csv(self.file_path)
                if "dt" in df.columns and "risk_score" in df.columns:
                    idx = pd.to_datetime(df["dt"], errors="coerce", utc=True)
                    s = pd.to_numeric(df["risk_score"], errors="coerce").clip(0.0,1.0)
                    self.series = pd.Series(s.values, index=idx).sort_index()
            except Exception as e:
                logger.warning(f"Failed to read ai_risk_scores.csv: {e}")
                self.series = None
    def available(self) -> bool:
        return self.loader.model is not None or (self.series is not None and not self.series.empty)
    def get_series(self, index: pd.DatetimeIndex, feats: Optional[pd.DataFrame] = None) -> pd.Series:
        if self.loader.model is not None and feats is not None and not feats.empty:
            m = self.loader.model
            try:
                if hasattr(m, "predict_proba"):
                    proba = m.predict_proba(feats)
                    p1 = proba[:,1] if proba.ndim==2 and proba.shape[1]>=2 else proba.ravel()
                    s = pd.Series(p1, index=feats.index).clip(0.0,1.0)
                    return s.reindex(index).ffill().fillna(1.0)
                elif hasattr(m, "predict"):
                    y = m.predict(feats)
                    s = pd.Series(y, index=feats.index).clip(0.0,1.0)
                    return s.reindex(index).ffill().fillna(1.0)
            except Exception as e:
                logger.error(f"AIRiskModel prediction failed: {e}")
        if self.series is not None and not self.series.empty:
            return self.series.reindex(index).ffill().fillna(1.0)
        return pd.Series(1.0, index=index)
