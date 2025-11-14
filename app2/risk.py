
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class RiskParams:
    per_trade_risk: float = 0.001
    vol_lb: int = 48

def realized_vol(close: pd.Series, lb: int = 48) -> pd.Series:
    ret = close.astype(float).pct_change().rolling(lb).std()
    return ret.replace(0.0, np.nan).ffill().bfill().fillna(0.0)

def position_size(close: pd.Series, p: float, equity: float, rp: RiskParams) -> pd.Series:
    vol = realized_vol(close, rp.vol_lb).replace(0.0, np.nan).ffill().bfill()
    dollar_risk = max(rp.per_trade_risk * equity, 1.0)
    denom = (vol * max(p, 1e-9)).replace(0.0, np.nan).ffill().bfill()
    size = dollar_risk / denom
    return size.clip(0, np.inf).reindex(close.index).ffill().bfill()

def from_config():
    try:
        from .config import config
        return RiskParams(per_trade_risk=getattr(config.risk_cfg, "per_trade_risk", 0.001),
                          vol_lb=getattr(config.risk_cfg, "vol_lb", 48))
    except Exception:
        return RiskParams()
