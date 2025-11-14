from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class RiskConfig:
    atr_period: int = 14
    atr_stop_mult: float | None = None
    rr_take: float | None = None
    risk_per_trade: float | None = None

def compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def atr_position_size(atr: float, equity: float, risk_per_trade: float, stop_mult: float) -> float:
    if atr is None or np.isnan(atr) or atr <= 0 or risk_per_trade is None or stop_mult is None or stop_mult <= 0:
        return 1.0
    risk_amount = max(equity * float(risk_per_trade), 1e-9)
    stop_dist = float(atr) * float(stop_mult)
    if stop_dist <= 0:
        return 1.0
    qty = risk_amount / stop_dist
    return float(max(qty, 0.0))
