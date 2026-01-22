from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RiskState:
    equity: float
    max_equity: float
    max_dd: float
    consec_losses: int
    last_day: Optional[pd.Timestamp]
    daily_equity_start: float
    daily_dd: float
    daily_disabled: bool
    stopped_by_circuit_breaker: bool
    circuit_breaker_time: Optional[pd.Timestamp]


def init_risk_state(equity0: float) -> RiskState:
    return RiskState(
        equity=equity0,
        max_equity=equity0,
        max_dd=0.0,
        consec_losses=0,
        last_day=None,
        daily_equity_start=equity0,
        daily_dd=0.0,
        daily_disabled=False,
        stopped_by_circuit_breaker=False,
        circuit_breaker_time=None,
    )


def on_new_day(state: RiskState, day: pd.Timestamp) -> None:
    state.daily_equity_start = state.equity
    state.daily_dd = 0.0
    state.daily_disabled = False
    state.consec_losses = 0
    state.last_day = day


def calc_entry_price(df_sig: pd.DataFrame, i: int) -> Tuple[Optional[int], Optional[float]]:
    entry_idx = i + 1
    if entry_idx >= len(df_sig):
        return None, None
    col = "open" if "open" in df_sig.columns else "close"
    price = float(df_sig[col].iloc[entry_idx])
    if not np.isfinite(price) or price <= 0:
        return None, None
    return entry_idx, price


def build_weekend_boundary(index: pd.Index) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        return np.zeros(len(index), dtype=bool)
    if len(index) <= 1:
        return np.zeros(len(index), dtype=bool)
    weekday = index.weekday
    trading_idx = np.where(weekday < 5)[0]
    boundary = np.zeros(len(index), dtype=bool)
    if len(trading_idx) <= 1:
        return boundary
    curr_idx = trading_idx[:-1]
    next_idx = trading_idx[1:]
    curr_days = index[curr_idx].normalize()
    next_days = index[next_idx].normalize()
    day_gap = (next_days - curr_days).days
    boundary[curr_idx[day_gap >= 2]] = True
    return boundary


def calc_qty(equity: float, entry_price: float, risk_pct_per_trade: float, sl_pct: float) -> float:
    risk_capital = equity * risk_pct_per_trade
    sl_dist = entry_price * sl_pct
    if sl_dist <= 0:
        return 0.0
    qty = risk_capital / sl_dist
    if qty <= 0:
        return 0.0
    return float(qty)


def calc_sl_tp(entry_price: float, sl_pct: float, tp_pct: float, side: int) -> Tuple[float, float]:
    if side > 0:
        return entry_price * (1.0 - sl_pct), entry_price * (1.0 + tp_pct)
    return entry_price * (1.0 + sl_pct), entry_price * (1.0 - tp_pct)


def eval_exit(
    position_side: int,
    price: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    sl_price: float,
    tp_price: float,
    sig: int,
    bars_held: int,
    max_bars_in_trade: int,
) -> Tuple[bool, str, float]:
    hit_sl = False
    hit_tp = False
    exit_price = price

    if position_side > 0:
        if bar_open <= sl_price:
            return True, "gap_sl", bar_open
        if bar_open >= tp_price:
            return True, "gap_tp", tp_price
    else:
        if bar_open >= sl_price:
            return True, "gap_sl", bar_open
        if bar_open <= tp_price:
            return True, "gap_tp", tp_price

    if position_side > 0:
        if bar_low <= sl_price < bar_high:
            hit_sl = True
            exit_price = sl_price
        elif bar_high >= tp_price > bar_low:
            hit_tp = True
            exit_price = tp_price
    else:
        if bar_high >= sl_price > bar_low:
            hit_sl = True
            exit_price = sl_price
        elif bar_low <= tp_price < bar_high:
            hit_tp = True
            exit_price = tp_price

    need_close = hit_sl or hit_tp
    if not need_close:
        if sig == -position_side and sig != 0:
            need_close = True
        if bars_held >= max_bars_in_trade:
            need_close = True

    if not need_close:
        return False, "unknown", exit_price

    exit_reason = "unknown"
    if hit_sl:
        exit_reason = "sl"
    elif hit_tp:
        exit_reason = "tp"
    elif sig == -position_side and sig != 0:
        exit_reason = "opposite_signal"
    elif bars_held >= max_bars_in_trade:
        exit_reason = "timeout"

    return True, exit_reason, exit_price


def update_after_exit(
    state: RiskState,
    pnl: float,
    ts: pd.Timestamp,
    daily_dd_limit_pct: Optional[float],
    max_consecutive_losses: Optional[int],
) -> bool:
    post_cb = False
    state.equity += pnl
    if state.daily_equity_start != 0:
        state.daily_dd = (state.equity - state.daily_equity_start) / state.daily_equity_start

    state.max_equity = max(state.max_equity, state.equity)
    if state.equity < state.max_equity and state.max_equity > 0:
        dd = (state.equity - state.max_equity) / state.max_equity
        state.max_dd = min(state.max_dd, dd)

    if pnl < 0:
        state.consec_losses += 1
    else:
        state.consec_losses = 0

    if daily_dd_limit_pct is not None and state.daily_dd <= -abs(float(daily_dd_limit_pct)):
        state.daily_disabled = True

    if max_consecutive_losses is not None and state.consec_losses >= max_consecutive_losses:
        post_cb = True
        state.stopped_by_circuit_breaker = True
        state.circuit_breaker_time = ts
        state.daily_disabled = True

    return post_cb
