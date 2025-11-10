# app/models/catboost_model.py
from .base import BaseModel
from catboost import CatBoostClassifier
import numpy as np, pandas as pd

class CatModel(BaseModel):
    def __init__(self, params=None):
        self.m = CatBoostClassifier(
            depth=6, learning_rate=0.05, loss_function="Logloss",
            iterations=800, verbose=False, **(params or {})
        )
    def fit(self, X, y):
        self.m.fit(X, y)
        return self
    def predict_proba(self, X) -> pd.Series:
        p = self.m.predict_proba(X)[:,1]
        return pd.Series(p, index=X.index)
