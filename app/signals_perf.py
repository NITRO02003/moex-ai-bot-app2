
# app/signals_perf.py
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
from .ai_models import AIRiskModel
from .logging_utils import get_logger

logger = get_logger("signals_perf")

def build_signals_matrix(ai_signals: Dict[str, pd.Series], union_index: pd.DatetimeIndex) -> pd.DataFrame:
    """No forward-fill: missing bar == no signal (0)."""
    df = pd.DataFrame(index=union_index)
    for sym, s in ai_signals.items():
        df[sym] = s.reindex(union_index)
    return df.fillna(0.0)

def build_risk_series(ai_risk: AIRiskModel, union_index: pd.DatetimeIndex, union_feats: Optional[pd.DataFrame] = None) -> pd.Series:
    s = ai_risk.get_series(union_index, feats=union_feats)
    return s.ffill().fillna(1.0)
