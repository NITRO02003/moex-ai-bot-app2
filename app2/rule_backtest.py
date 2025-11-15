from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from . import metrics as MX


@dataclass
class RuleBtParams:
    commission: float = 0.0005
    slippage_bps: float = 1.0
    risk_per_trade: float = 0.005
    atr_len: int = 14
    sl_atr_mult: float = 1.5
    tp_mult: float = 3.0


def _atr(px: pd.DataFrame, n: int) -> pd.Series:
    c = px["close"].astype(float)
    h = px["high"].astype(float)
    l = px["low"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            (h - l),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    return atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_rule_symbol(
    prices: pd.DataFrame,
    side: pd.Series,
    rp: RuleBtParams,
    equity0: float = 1_000_000.0,
) -> Dict[str, Any]:
    if prices is None or prices.empty:
        return {
            "pnl": pd.Series(dtype="float64"),
            "equity": pd.Series([equity0], index=pd.DatetimeIndex([])),
            "metrics": {},
        }

    px = prices.copy()
    if "close" not in px.columns or "high" not in px.columns or "low" not in px.columns:
        raise ValueError("prices must have close/high/low columns")

    c = px["close"].astype(float)
    atr = _atr(px, rp.atr_len)

    side = side.reindex(c.index).fillna(0).astype(int)

    idx = c.index
    n = len(idx)

    equity = float(equity0)
    position = 0.0
    entry_price = 0.0
    commission_paid = 0.0
    trades = 0
    turnover = 0.0

    pnl_per_bar = np.zeros(n, dtype="float64")
    prev_c = c.shift(1).bfill()

    for i, ts in enumerate(idx):
        price = float(c.iat[i])
        s = int(side.iat[i])

        if position != 0.0 and i > 0:
            pnl_bar = position * (price - float(prev_c.iat[i]))
            pnl_per_bar[i] += pnl_bar
            equity += pnl_bar

        if i > 0:
            prev_side = int(side.iat[i - 1])
        else:
            prev_side = 0

        close_position = False
        open_new = False

        if position != 0.0 and s == 0:
            close_position = True
        elif position != 0.0 and np.sign(position) != s and s != 0:
            close_position = True
            open_new = True
        elif position == 0.0 and s != 0:
            open_new = True

        if close_position and position != 0.0:
            exit_price = price * (1.0 - np.sign(position) * rp.slippage_bps / 10000.0)
            trade_pnl = position * (exit_price - entry_price)
            trade_comm = rp.commission * abs(position) * exit_price
            commission_paid += trade_comm
            trade_pnl -= trade_comm
            pnl_per_bar[i] += trade_pnl
            equity += trade_pnl
            turnover += abs(position) * exit_price
            trades += 1
            position = 0.0
            entry_price = 0.0

        if open_new and s != 0:
            risk_dollars = max(equity * rp.risk_per_trade, 1.0)
            atr_i = float(atr.iat[i]) if float(atr.iat[i]) > 0 else 1.0
            dollar_risk_per_unit = atr_i * rp.sl_atr_mult
            size_units = risk_dollars / dollar_risk_per_unit
            size_units = float(size_units)
            if size_units <= 0 or not np.isfinite(size_units):
                continue
            side_sign = 1.0 if s > 0 else -1.0
            entry_price = price * (1.0 + side_sign * rp.slippage_bps / 10000.0)
            position = side_sign * size_units
            entry_notional = abs(position) * entry_price
            trade_comm = rp.commission * entry_notional
            commission_paid += trade_comm
            pnl_per_bar[i] -= trade_comm
            equity -= trade_comm
            turnover += entry_notional

    pnl = pd.Series(pnl_per_bar, index=idx, name="pnl")
    eq = pnl.cumsum() + float(equity0)

    mx = MX.summary_from_pnl(pnl, equity0=equity0)
    mx.update(
        dict(
            trade_count=int(trades),
            turnover=float(turnover),
            commission_paid=float(commission_paid),
        )
    )
    return dict(side=side, pnl=pnl, equity=eq, metrics=mx)
