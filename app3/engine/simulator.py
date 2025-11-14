from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .risk import RiskConfig, compute_atr, atr_position_size

@dataclass
class SimConfig:
    fees_bps: float = 1.0
    slippage_bps: float = 0.0
    cooldown_bars: int = 20
    min_hold_bars: int = 10
    qty: float = 1.0
    # use default_factory to avoid mutable default
    risk: RiskConfig = field(default_factory=RiskConfig)

def simulate(bars: pd.DataFrame, signals: pd.DataFrame, cfg: SimConfig) -> tuple[pd.Series, pd.Series, List[Dict[str, Any]]]:
    """
    Fill-based simulator with optional ATR stops/takes and ATR sizing.
    - Position changes only at signal times
    - PnL realized on exit/flip/TP/SL
    - SL/TP evaluated per bar using High/Low crossing (discrete approximation)
    """
    df = bars[["dt","open","high","low","close"]].copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    sig = signals[["dt","side"]].copy().drop_duplicates().sort_values("dt").reset_index(drop=True)
    s = pd.merge_asof(sig, df[["dt"]], on="dt", direction="forward")
    s = s.dropna(subset=["dt"]).reset_index(drop=True)

    atr_ser = compute_atr(df, period=cfg.risk.atr_period) if (cfg.risk.atr_stop_mult or cfg.risk.risk_per_trade) else pd.Series([np.nan]*len(df))

    equity = []
    cash = 1_000_000.0
    position = 0               # -1/0/+1
    qty = float(cfg.qty)       # updated per-trade if ATR sizing is on
    entry_price: Optional[float] = None
    entry_atr: Optional[float] = None
    stop_price: Optional[float] = None
    take_price: Optional[float] = None

    fills: List[Dict[str, Any]] = []
    pnl_list: List[float] = []

    fees = cfg.fees_bps / 1e4
    slippage = cfg.slippage_bps / 1e4

    bars_since_fill = np.inf
    bars_in_position = 0

    def exec_price(px: float, side: int, is_entry: bool) -> float:
        adj = (1 + slippage) if (side > 0) else (1 - slippage) if (side < 0) else 1.0
        trade_px = float(px) * adj
        fee_mult = (1 - fees) if (is_entry and side < 0) else (1 + fees) if (is_entry and side > 0) else 1.0
        return trade_px * fee_mult

    # helper to set SL/TP
    def set_protective_levels(side: int, entry_px: float, atr_value: float) -> tuple[Optional[float], Optional[float]]:
        if cfg.risk.atr_stop_mult is None or atr_value is None or np.isnan(atr_value) or atr_value <= 0:
            return None, None
        stop_dist = cfg.risk.atr_stop_mult * float(atr_value)
        if side > 0:
            sl = entry_px - stop_dist
            tp = entry_px + (cfg.risk.rr_take * stop_dist) if (cfg.risk.rr_take and cfg.risk.rr_take > 0) else None
        elif side < 0:
            sl = entry_px + stop_dist
            tp = entry_px - (cfg.risk.rr_take * stop_dist) if (cfg.risk.rr_take and cfg.risk.rr_take > 0) else None
        else:
            sl = None; tp = None
        return sl, tp

    sig_idx = 0
    n_sigs = len(s)

    for i, row in df.iterrows():
        dt = row["dt"]; o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        current_atr = float(atr_ser.iloc[i]) if len(atr_ser) == len(df) else np.nan

        # Evaluate SL/TP intrabar for open positions BEFORE processing new signals at this bar
        if position != 0 and entry_price is not None:
            exit_reason = None
            exit_px = None

            if stop_price is not None:
                # Long: stop if low <= stop ; Short: stop if high >= stop
                if (position > 0 and l <= stop_price) or (position < 0 and h >= stop_price):
                    exit_reason = "stop"
                    exit_px = stop_price

            if exit_reason is None and take_price is not None:
                # Long: take if high >= take ; Short: take if low <= take
                if (position > 0 and h >= take_price) or (position < 0 and l <= take_price):
                    exit_reason = "take"
                    exit_px = take_price

            if exit_reason is not None:
                price_x = exec_price(exit_px, position, is_entry=False)
                pnl = (price_x - entry_price) * position * qty
                cash += pnl
                pnl_list.append(pnl)
                fills.append({"dt": dt, "type": exit_reason, "side": position, "price": price_x, "pnl": pnl})
                position = 0
                entry_price = None
                stop_price = None
                take_price = None
                entry_atr = None
                bars_since_fill = 0
                bars_in_position = 0

        # Process signals that occur at/before this bar's dt
        while sig_idx < n_sigs and s.loc[sig_idx, "dt"] <= dt:
            proposed = int(np.sign(s.loc[sig_idx, "side"]))
            sig_idx += 1

            can_enter = (bars_since_fill >= cfg.cooldown_bars)
            can_exit_or_flip = (bars_in_position >= cfg.min_hold_bars)

            if proposed == position:
                continue

            # Exit/Flip
            if position != 0 and proposed != position:
                if can_exit_or_flip and entry_price is not None:
                    price_x = exec_price(c, position, is_entry=False)
                    pnl = (price_x - entry_price) * position * qty
                    cash += pnl
                    pnl_list.append(pnl)
                    fills.append({"dt": dt, "type": "exit", "side": position, "price": price_x, "pnl": pnl})
                    position = 0
                    entry_price = None
                    stop_price = None
                    take_price = None
                    entry_atr = None
                    bars_since_fill = 0
                else:
                    continue  # skip early flip

            # Entry
            if position == 0 and proposed != 0 and can_enter:
                # ATR sizing
                if cfg.risk.risk_per_trade is not None and cfg.risk.atr_stop_mult is not None and current_atr and current_atr > 0:
                    qty = atr_position_size(current_atr, cash, cfg.risk.risk_per_trade, cfg.risk.atr_stop_mult)
                else:
                    qty = float(cfg.qty)
                price_e = exec_price(c, proposed, is_entry=True)
                entry_price = price_e
                entry_atr = current_atr if (current_atr and current_atr > 0) else None
                position = proposed
                stop_price, take_price = set_protective_levels(position, entry_price, entry_atr if entry_atr else np.nan)
                fills.append({"dt": dt, "type": "entry", "side": position, "price": price_e, "qty": qty, "atr": entry_atr})
                bars_since_fill = 0

        # advance counters
        if position != 0:
            bars_in_position += 1
        bars_since_fill += 1

        # mark-to-market
        mtm = 0.0
        if position != 0 and entry_price is not None:
            mtm = (c - entry_price) * position * qty
        equity.append((dt, cash + mtm))

    # force close at the end
    if position != 0 and entry_price is not None:
        c = float(df.iloc[-1]["close"]); dt = df.iloc[-1]["dt"]
        price_x = exec_price(c, position, is_entry=False)
        pnl = (price_x - entry_price) * position * qty
        cash += pnl
        pnl_list.append(pnl)
        fills.append({"dt": dt, "type": "exit", "side": position, "price": price_x, "pnl": pnl})

    eq_ser = pd.Series([e[1] for e in equity], index=[e[0] for e in equity], dtype=float)
    pnl_ser = pd.Series(pnl_list, dtype=float) if pnl_list else pd.Series([], dtype=float)
    return eq_ser, pnl_ser, fills
