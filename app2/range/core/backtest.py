import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..range_v3 import RangeV3Params
# The following imports were used solely for entry-model gating in the previous
# implementation.  That logic has been moved to ``entry_gating.py``.  We no
# longer import feature engineering or slope helpers here.
from .entry_gating import apply_entry_ai_filter as _apply_entry_ai_filter
from .execution import run_trades_from_signals as run_trades_from_signals_impl

# Reporting utilities: convert trades to DataFrames and write outputs
from .reporting import trades_to_df as _report_trades_to_df, save_symbol_outputs, save_portfolio_outputs

# Default list of compact features for entry gating.  Still required for CLI default argument.
from ..baseline_ml import COMPACT_FEATURES

from ...parallel import parallel_map
from .engine import run_core_for_symbol
from .metrics import build_symbol_metrics, compute_pnl_rel, compute_trade_pnl
from .portfolio import build_portfolio_stats
from .risk import (
    RiskState,
    build_weekend_boundary,
    calc_entry_price,
    calc_qty,
    calc_sl_tp,
    eval_exit,
    init_risk_state,
    on_new_day,
    update_after_exit,
)

# analytics helpers
from .snapshots import build_entry_snapshot_frame


def _parse_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# Import the canonical trade dataclass from the contracts module.  This
# centralizes the definition of trade records and decouples data structures
# from the backtest orchestration logic.  Importing it as Trade preserves the
# original naming used throughout the module.
from .contracts import TradeRecord as Trade
# Import data and configuration helpers from the dedicated I/O module.  These
# functions encapsulate loading of range parameters, symbol discovery and OHLCV
# data retrieval.  Centralizing them in ``data_io`` removes low-level file I/O
# concerns from this orchestration layer and supports easier testing.
from .data_io import (
    _load_range_config,
    _list_available_symbols,
    _find_data_path,
    _load_ohlcv,
)

# Import the runner orchestrator to delegate multi-symbol backtest execution.
from .runner import run_range_backtest as _run_range_backtest



def _run_trades_from_signals(
    symbol: str,
    df_sig: pd.DataFrame,
    params: RangeV3Params,
    equity0: float,
    entry_allow_mask: Optional[pd.Series] = None,
    no_hold_weekend: bool = False,
) -> Tuple[List[Trade], Dict[str, Any]]:
    # Delegate the execution logic to the implementation in the execution module.
    # This early return bypasses the heavy logic below and decouples trade loop
    # from the backtest orchestrator. All code following this return is preserved
    # only for backward compatibility and will never execute in this configuration.
    return run_trades_from_signals_impl(
        symbol,
        df_sig,
        params,
        equity0,
        entry_allow_mask=entry_allow_mask,
        no_hold_weekend=no_hold_weekend,
    )

    trades: List[Trade] = []
    position_side = 0
    entry_idx: Optional[int] = None
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    qty = 0.0
    entry_reason: Optional[str] = None
    entry_geo_class: Optional[str] = None
    entry_geo_valid_box: Optional[bool] = None
    entry_geo_h_pct: Optional[float] = None
    entry_atr_pct: Optional[float] = None
    entry_slope_pct_per_bar: Optional[float] = None
    entry_dist_L_pct: Optional[float] = None
    entry_dist_U_pct: Optional[float] = None
    state: RiskState = init_risk_state(equity0)
    last_price: Optional[float] = None

    prices = df_sig["close"]
    if "v3_signal" in df_sig.columns:
        signals = df_sig["v3_signal"]
    else:
        signals = pd.Series(0, index=df_sig.index)
    signals = signals.fillna(0).astype(int)
    weekend_boundary = build_weekend_boundary(df_sig.index) if no_hold_weekend else None
    # --- analytics: precompute entry snapshot features (does NOT affect trading logic) ---
    # Delegate calculation of geometry, ATR, slope and distance features to snapshots module.
    snapshot_df = build_entry_snapshot_frame(df_sig, params)
    geo_class = snapshot_df["geo_class"]
    geo_valid_box = snapshot_df["geo_valid_box"]
    geo_h_pct = snapshot_df["geo_h_pct"]
    atr_pct = snapshot_df["atr_pct"]
    slope_pct_per_bar = snapshot_df["slope_pct_per_bar"]
    dist_L_pct = snapshot_df["dist_L_pct"]
    dist_U_pct = snapshot_df["dist_U_pct"]
    # Base range levels for entry distance calculation.  These columns are
    # populated by the state machine (rolling range levels).  If absent,
    # distance to L/U at entry cannot be computed.
    base_L = df_sig["v3_L"] if "v3_L" in df_sig.columns else None
    base_U = df_sig["v3_U"] if "v3_U" in df_sig.columns else None

    max_consecutive_losses = getattr(params, "max_consecutive_losses", None)
    daily_dd_limit_pct = getattr(params, "daily_dd_limit_pct", None)

    for i, (ts, sig) in enumerate(signals.items()):
        day = ts.date()
        if state.last_day is None or day != state.last_day:
            on_new_day(state, day)
        next_ts = df_sig.index[i + 1] if i + 1 < len(df_sig.index) else None
        weekend_exit = False
        if no_hold_weekend and weekend_boundary is not None:
            if weekend_boundary[i]:
                weekend_exit = True
        price = float(prices.iloc[i])
        if np.isfinite(price) and price > 0:
            last_price = price

        if position_side != 0 and weekend_exit:
            exit_price = last_price if last_price is not None else price
            if np.isfinite(exit_price) and exit_price > 0:
                pnl = compute_trade_pnl(entry_price, exit_price, qty, position_side)
                pnl_rel = compute_pnl_rel(pnl, state.equity)
                post_cb = update_after_exit(
                    state,
                    pnl,
                    ts,
                    daily_dd_limit_pct,
                    max_consecutive_losses,
                )
                trades.append(
                    Trade(
                        symbol=symbol,
                        side=position_side,
                        entry_time=df_sig.index[entry_idx] if entry_idx is not None else ts,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        pnl=pnl,
                        pnl_rel=pnl_rel,
                        bars_held=i - (entry_idx or 0),
                        entry_reason=entry_reason,
                        exit_reason="weekend_exit",
                        post_circuit_breaker=post_cb,
                        entry_geo_class=entry_geo_class,
                        entry_geo_valid_box=entry_geo_valid_box,
                        entry_geo_h_pct=entry_geo_h_pct,
                        entry_atr_pct=entry_atr_pct,
                        entry_slope_pct_per_bar=entry_slope_pct_per_bar,
                        entry_dist_L_pct=entry_dist_L_pct,
                        entry_dist_U_pct=entry_dist_U_pct,
                    )
                )
                position_side = 0
                entry_idx = None
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
                qty = 0.0
                entry_reason = None
                entry_geo_class = None
                entry_geo_valid_box = None
                entry_geo_h_pct = None
                entry_atr_pct = None
                entry_slope_pct_per_bar = None
                entry_dist_L_pct = None
                entry_dist_U_pct = None
                continue

        if not np.isfinite(price) or price <= 0:
            continue

        if position_side != 0:
            bars_held = i - (entry_idx or 0)
            need_close = False

            # --- price-based SL/TP checks on current bar ---
            if "high" in df_sig.columns:
                bar_high = float(df_sig["high"].iloc[i])
            else:
                bar_high = price
            if "low" in df_sig.columns:
                bar_low = float(df_sig["low"].iloc[i])
            else:
                bar_low = price
            if "open" in df_sig.columns:
                bar_open = float(df_sig["open"].iloc[i])
            else:
                bar_open = price

            need_close, exit_reason, exit_price = eval_exit(
                position_side,
                price,
                bar_open,
                bar_high,
                bar_low,
                sl_price,
                tp_price,
                sig,
                bars_held,
                params.max_bars_in_trade,
            )
            if not need_close and weekend_exit:
                need_close = True
                exit_reason = "weekend_exit"
                exit_price = price

            if need_close:
                pnl = compute_trade_pnl(entry_price, exit_price, qty, position_side)
                pnl_rel = compute_pnl_rel(pnl, state.equity)
                post_cb = update_after_exit(
                    state,
                    pnl,
                    ts,
                    daily_dd_limit_pct,
                    max_consecutive_losses,
                )

                trades.append(
                    Trade(
                        symbol=symbol,
                        side=position_side,
                        entry_time=df_sig.index[entry_idx] if entry_idx is not None else ts,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        pnl=pnl,
                        pnl_rel=pnl_rel,
                        bars_held=bars_held,
                        entry_reason=entry_reason,
                        exit_reason=exit_reason,
                        post_circuit_breaker=post_cb,
                        entry_geo_class=entry_geo_class,
                        entry_geo_valid_box=entry_geo_valid_box,
                        entry_geo_h_pct=entry_geo_h_pct,
                        entry_atr_pct=entry_atr_pct,
                        entry_slope_pct_per_bar=entry_slope_pct_per_bar,
                        entry_dist_L_pct=entry_dist_L_pct,
                        entry_dist_U_pct=entry_dist_U_pct,
                    )
                )
                position_side = 0
                entry_idx = None
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
                qty = 0.0
                entry_reason = None
                entry_geo_class = None
                entry_geo_valid_box = None
                entry_geo_h_pct = None
                entry_atr_pct = None
                entry_slope_pct_per_bar = None
                entry_dist_L_pct = None
                entry_dist_U_pct = None

                continue

        # open new position (execution at open_{t+1} per contract)
        if position_side == 0 and sig != 0:
            if state.daily_disabled:
                continue
            if entry_allow_mask is not None:
                allow = bool(entry_allow_mask.iloc[i])
                if not allow:
                    continue
            if no_hold_weekend:
                if ts.weekday() >= 5:
                    continue
                if weekend_boundary is not None and weekend_boundary[i]:
                    continue
                if next_ts is not None and next_ts.weekday() >= 5:
                    continue
            entry_idx_candidate, entry_price_candidate = calc_entry_price(df_sig, i)
            if entry_idx_candidate is None or entry_price_candidate is None:
                continue
            sl_pct = params.sl_pct
            tp_pct = params.tp_pct
            qty = calc_qty(state.equity, entry_price_candidate, params.risk_pct_per_trade, sl_pct)
            if qty <= 0:
                continue
            position_side = sig
            entry_idx = entry_idx_candidate
            entry_price = entry_price_candidate
            # Entry reason and snapshot (analytics)
            entry_reason = "v3_long" if sig > 0 else "v3_short"

            if geo_class is not None:
                v = geo_class.iloc[i]
                entry_geo_class = str(v) if pd.notna(v) else None
            else:
                entry_geo_class = None

            if geo_valid_box is not None:
                v = geo_valid_box.iloc[i]
                entry_geo_valid_box = bool(v) if pd.notna(v) else None
            else:
                entry_geo_valid_box = None

            v = geo_h_pct.iloc[i] if geo_h_pct is not None else np.nan
            entry_geo_h_pct = float(v) if np.isfinite(v) else None

            v = atr_pct.iloc[i] if atr_pct is not None else np.nan
            entry_atr_pct = float(v) if np.isfinite(v) else None

            v = slope_pct_per_bar.iloc[i] if slope_pct_per_bar is not None else np.nan
            entry_slope_pct_per_bar = float(v) if np.isfinite(v) else None

            entry_dist_L_pct = None
            entry_dist_U_pct = None
            if base_L is not None and base_U is not None:
                L = base_L.iloc[i]
                U = base_U.iloc[i]
                if pd.notna(L) and pd.notna(U):
                    H = float(U - L)
                    if np.isfinite(H) and H > 0:
                        entry_dist_L_pct = float((entry_price - float(L)) / H)
                        entry_dist_U_pct = float((float(U) - entry_price) / H)
            sl_price, tp_price = calc_sl_tp(entry_price, sl_pct, tp_pct, position_side)

    pnl_vals = [t.pnl for t in trades]
    metrics = build_symbol_metrics(
        symbol=symbol,
        pnls=pnl_vals,
        equity0=equity0,
        max_dd=state.max_dd,
        equity_final=state.equity,
        circuit_breaker_hit=state.stopped_by_circuit_breaker,
        circuit_breaker_time=(str(state.circuit_breaker_time) if state.circuit_breaker_time is not None else None),
    )
    return trades, metrics



def _trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    """Delegate conversion of trades to DataFrame to the reporting module.

    This thin wrapper preserves the original name while forwarding to
    ``reporting.trades_to_df``. Keeping this stub simplifies migration
    until all internal calls are updated to use the new reporting module.
    """
    # Import locally to avoid cyclic import issues at module import time
    from .reporting import trades_to_df
    return trades_to_df(trades)


def _run_symbol_task(
    args: tuple[
        str,
        str,
        float,
        RangeV3Params,
        str,
        str,
        str,
        str,
        float,
        float,
        str,
        str,
        float,
        float,
        float,
        bool,
        List[str],
    ],
) -> tuple[str, Dict[str, Any] | None, pd.DataFrame]:
    (
        symbol,
        interval,
        equity0,
        params,
        out_prefix,
        tag,
        entry_model_path,
        entry_model_mode,
        entry_model_threshold,
        entry_model_top_pct,
        entry_model_trend_path,
        entry_model_trend_mode,
        entry_model_trend_threshold,
        entry_model_trend_top_pct,
        entry_trend_slope_k,
        no_hold_weekend,
        entry_feature_cols,
    ) = args
    try:
        df = _load_ohlcv(symbol, interval)
    except FileNotFoundError:
        return symbol, None, pd.DataFrame()

    sig_df, debug_info = run_core_for_symbol(df, params)
    entry_allow_mask, entry_ai_stats = _apply_entry_ai_filter(
        sig_df,
        params,
        entry_model_path,
        entry_model_mode,
        entry_model_threshold,
        entry_model_top_pct,
        entry_model_trend_path,
        entry_model_trend_mode,
        entry_model_trend_threshold,
        entry_model_trend_top_pct,
        entry_trend_slope_k,
        entry_feature_cols,
    )
    trades, metrics = _run_trades_from_signals(
        symbol,
        sig_df,
        params,
        equity0,
        entry_allow_mask=entry_allow_mask,
        no_hold_weekend=no_hold_weekend,
    )
    if entry_ai_stats:
        metrics.update(entry_ai_stats)
    # Save per-symbol outputs (stats, trades, snapshots, debug) using the reporting module
    save_symbol_outputs(
        out_prefix=out_prefix,
        symbol=symbol,
        interval=interval,
        tag=tag,
        metrics=metrics,
        trades=trades,
        debug_info=debug_info,
        sig_df=sig_df,
    )

    trades_df = _trades_to_df(trades)
    print(
        f"[range-v3] {symbol}: trades={len(trades)} "
        f"pf={metrics['pf']:.3f} win_rate={metrics['win_rate']:.3f}"
    )
    return symbol, metrics, trades_df


def main(args):
    # Offload orchestration to the runner.  This function will build per-symbol
    # tasks, run them (optionally in parallel), aggregate results and write
    # out portfolio-level outputs.  Any return value is ignored.
    return _run_range_backtest(args)
