# strategy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from .features import build_features_10m  # ИЗМЕНИЛИ НА 10m

@dataclass
class FeatureStrategy:
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features_10m(df)  # ИЗМЕНИЛИ НА 10m

class ParallelStrategyExecutor:
    def __init__(self, strategy: FeatureStrategy):
        self.strategy = strategy
    def compute_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        res = {}
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            feats = self.strategy.features(df)
            res[sym] = feats
        return res