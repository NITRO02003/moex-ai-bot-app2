
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from . import strategy as S, metrics as MX, risk as R

@dataclass
class BtParams:
    commission: float = 0.0005
    slippage_bps: float = 1.0
    horizon: int = 1

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def signal_and_size(prices: pd.DataFrame, bundle: dict, rp: R.RiskParams, equity0: float,
                    th_long: Optional[float], th_short: Optional[float]) -> Tuple[pd.Series, pd.Series]:
    return S.signal_and_size(prices, bundle, rp, equity0, th_long=th_long, th_short=th_short, use_filters=True)

def run_symbol(prices: pd.DataFrame, model_bundle: dict, rp: R.RiskParams, bt: BtParams,
               equity0: float = 1_000_000.0, threshold: Optional[float] = None,
               th_long: Optional[float] = None, th_short: Optional[float] = None) -> Dict[str, Any]:
    side, size = signal_and_size(prices, model_bundle, rp, equity0, th_long=th_long, th_short=th_short)
    close = prices["close"].astype(float).reindex(side.index).ffill()
    pnl = pd.Series(0.0, index=side.index)
    shares = 0.0
    commission_paid = 0.0; turnover = 0.0; trades = 0

    for ts in side.index:
        p = float(close.loc[ts])
        desired = float(size.loc[ts]) * float(side.loc[ts]) / max(p, 1e-9)
        d = desired - shares
        if d != 0.0:
            turn = abs(d) * p
            cost = turn * bt.commission + turn * (bt.slippage_bps/10_000.0)
            pnl.loc[ts] += (-d) * p - cost
            commission_paid += cost; turnover += turn; trades += 1
            shares = desired

    eq = pnl.cumsum() + equity0
    mx = MX.equity_stats(eq)
    mx.update(dict(trade_count=trades, turnover=float(turnover), commission_paid=float(commission_paid)))
    return dict(side=side, size=size, pnl=pnl, equity=eq, metrics=mx, timestamp=_now_iso())

def save_report(symbol: str, res: Dict[str, Any]) -> None:
    from .paths import REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{symbol}_backtest.json"
    payload = {
        "timestamp": _now_iso(),
        "metrics": res.get("metrics", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        import json; json.dump(payload, f, ensure_ascii=False, indent=2)
