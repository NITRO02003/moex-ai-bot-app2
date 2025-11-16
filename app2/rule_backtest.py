"""Simple backtest engine for rule‑based strategies.

This module connects raw direction signals (long/short/flat) to a
position sizing and trade execution logic.  It uses a very basic ATR
based sizing rule: risk a fixed fraction of equity on each trade
relative to the distance to the stop, where the stop is set at a
multiple of ATR.  The backtester handles commissions and slippage
linearly and computes per‑bar PnL and summary metrics.

Note that this backtester does not simulate intrabar dynamics; it
assumes that entries and exits occur at the bar close with the
specified slippage and commission.  As such, it is intended for
comparative research rather than exact simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from . import metrics as MX


@dataclass
class RuleBtParams:
    """Parameters for the rule‑based backtest."""

    commission: float = 0.0005
    slippage_bps: float = 1.0
    # Fraction of equity risked per trade.  Reduced from 0.5% to 0.2%
    # to limit position sizes and drawdowns when used by default.  A
    # smaller per‑trade risk helps mitigate losses from false
    # signals and high volatility.
    risk_per_trade: float = 0.002
    atr_len: int = 14
    sl_atr_mult: float = 1.5
    tp_mult: float = 3.0  # take profit multiple of ATR (not used here but reserved)


def _compute_atr(px: pd.DataFrame, n: int) -> pd.Series:
    """Compute the Average True Range (ATR) as a ratio to price."""
    c = px["close"].astype(float)
    h = px["high"].astype(float)
    l = px["low"].astype(float)
    prev_c = c.shift(1).bfill()
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    return atr


def run_rule_symbol(px: pd.DataFrame,
                    side: pd.Series,
                    rp: RuleBtParams,
                    equity0: float = 1_000_000.0) -> Dict[str, object]:
    """Run a simple long/short backtest for a single symbol.

    Parameters
    ----------
    px : pandas.DataFrame
        OHLC dataframe with index aligned to ``side``.
    side : pandas.Series
        Series of trade directions (``+1`` for long, ``-1`` for short, ``0`` for flat).
    rp : RuleBtParams
        Backtest parameter dataclass controlling commission, slippage and risk size.
    equity0 : float, optional
        Initial capital.  Default is 1,000,000.

    Returns
    -------
    dict
        A dictionary containing the equity curve, PnL series and summary metrics.
    """
    # Ensure alignment
    side = side.reindex(px.index).fillna(0).astype(int)
    atr = _compute_atr(px, rp.atr_len)
    # Convert slippage bps to fraction
    slip_frac = rp.slippage_bps / 10_000.0
    # Containers
    equity = np.full(len(px), equity0, dtype=float)
    pnl = np.zeros(len(px), dtype=float)
    turnover = 0.0
    commission_paid = 0.0
    in_pos = False
    pos_dir = 0
    pos_size = 0.0
    entry_price = 0.0
    # Iterate bars
    for i in range(len(px)):
        # Mark to market PnL on open position
        if in_pos:
            price = px["close"].iat[i]
            price_diff = price - px["close"].iat[i - 1]
            pnl[i] += pos_dir * pos_size * price_diff
        # Determine if position should change
        target_dir = side.iat[i]
        if target_dir != pos_dir:
            # Exit existing position if any
            if in_pos:
                exit_price = px["close"].iat[i] * (1 - slip_frac * pos_dir)
                pnl[i] += pos_dir * pos_size * (exit_price - px["close"].iat[i])
                commission = rp.commission * abs(pos_size * exit_price)
                pnl[i] -= commission
                commission_paid += commission
                turnover += abs(pos_size * exit_price)
                in_pos = False
                pos_dir = 0
                pos_size = 0.0
            # Enter new position
            if target_dir != 0:
                # Risk per trade in dollars
                eq = equity[i - 1] if i > 0 else equity0
                risk_dollars = rp.risk_per_trade * eq
                # Use ATR to determine stop distance
                this_atr = atr.iat[i]
                stop_dist = max(this_atr * rp.sl_atr_mult * px["close"].iat[i], 1e-9)
                size = risk_dollars / stop_dist
                # Limit size to not exceed equity (no leverage)
                max_notional = eq  # 1x gross
                notional = min(size * px["close"].iat[i], max_notional)
                size = notional / px["close"].iat[i]
                entry_price = px["close"].iat[i] * (1 + slip_frac * target_dir)
                pos_size = size
                pos_dir = target_dir
                in_pos = True
                commission = rp.commission * abs(pos_size * entry_price)
                pnl[i] -= commission
                commission_paid += commission
                turnover += abs(pos_size * entry_price)
                # Immediately mark entry slippage in PnL
                pnl[i] += pos_dir * pos_size * (px["close"].iat[i] - entry_price)
        # Update equity
        equity[i] = equity0 + pnl[: i + 1].sum()
    equity_series = pd.Series(equity, index=px.index)
    pnl_series = pd.Series(pnl, index=px.index)
    stats = MX.summary_from_pnl(pnl_series, equity0)
    stats["trade_count"] = int((abs(np.diff(side.values)) > 0).sum())
    stats["turnover"] = turnover
    stats["commission_paid"] = commission_paid
    return {
        "equity_curve": equity_series,
        "pnl": pnl_series,
        "metrics": stats,
    }