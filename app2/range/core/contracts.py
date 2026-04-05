from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class TradeRecord:
    """
    A serializable record representing a completed trade.

    This dataclass mirrors the legacy Trade structure from backtest.py but
    lives in a separate module to decouple data representation from
    backtest execution logic.
    """
    symbol: str
    side: int  # +1 or -1
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_rel: float
    bars_held: int
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    post_circuit_breaker: Optional[bool] = None

    # Optional entry snapshot analytics; may be NaN/None if not available.
    entry_geo_class: Optional[str] = None
    entry_geo_valid_box: Optional[bool] = None
    entry_geo_h_pct: Optional[float] = None
    entry_atr_pct: Optional[float] = None
    entry_slope_pct_per_bar: Optional[float] = None
    entry_dist_L_pct: Optional[float] = None
    entry_dist_U_pct: Optional[float] = None
