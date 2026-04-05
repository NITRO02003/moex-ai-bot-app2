import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..range_v3 import RangeV3Params
from ..features import add_basic_range_features
from ..feature_sweep import add_derived_features
from ..baseline_ml import COMPACT_FEATURES
from .blocks import calc_slope_mask
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



def _run_trades_from_signals(
    symbol: str,
    df_sig: pd.DataFrame,
    params: RangeV3Params,
    equity0: float,
    entry_allow_mask: Optional[pd.Series] = None,
    no_hold_weekend: bool = False,
) -> Tuple[List[Trade], Dict[str, Any]]:
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
    window = int(getattr(params, "range_window_bars", 20) or 20)

    # Geometry (preferred) or fallback to v3_L/v3_U/v3_M columns if present
    geo_L = geo_U = geo_H = geo_M = geo_valid_box = geo_class = None
    try:
        from .geometry import compute_geometry  # local import to avoid circular deps

        df_geo = compute_geometry(df_sig, params)
        geo_L = df_geo.get("geo_L")
        geo_U = df_geo.get("geo_U")
        geo_H = df_geo.get("geo_H")
        geo_M = df_geo.get("geo_M")
        geo_valid_box = df_geo.get("geo_valid_box")
        geo_class = df_geo.get("geo_class")
    except Exception:
        pass

    base_L = geo_L if geo_L is not None else (df_sig["v3_L"] if "v3_L" in df_sig.columns else None)
    base_U = geo_U if geo_U is not None else (df_sig["v3_U"] if "v3_U" in df_sig.columns else None)
    base_M = geo_M if geo_M is not None else (df_sig["v3_M"] if "v3_M" in df_sig.columns else df_sig["close"])
    base_H = geo_H if geo_H is not None else ((base_U - base_L) if (base_L is not None and base_U is not None) else None)

    # ATR pct (causal) from OHLC
    close = df_sig["close"]
    prev_close = close.shift(1)
    tr1 = (df_sig["high"] - df_sig["low"]).abs()
    tr2 = (df_sig["high"] - prev_close).abs()
    tr3 = (df_sig["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    atr_pct = atr / close.replace(0, np.nan).abs()

    # Slope pct per bar of midline (causal)
    slope = (base_M - base_M.shift(window)) / float(window)
    slope_pct_per_bar = slope / close.replace(0, np.nan).abs()

    # Range height pct proxy (causal)
    geo_h_pct = None
    if base_H is not None:
        geo_h_pct = base_H / close.replace(0, np.nan).abs()

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


def _select_entry_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    work = df.replace([np.inf, -np.inf], np.nan).copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    return work[feature_cols]


def _build_entry_features(df_sig: pd.DataFrame) -> pd.DataFrame:
    feats = add_basic_range_features(df_sig, ma_len=20, ma_mode="ema")
    feats = add_derived_features(feats)
    return feats


def _build_trend_mask(
    df_sig: pd.DataFrame,
    params: RangeV3Params,
    trend_slope_k: float,
) -> pd.Series:
    if trend_slope_k <= 0:
        return pd.Series(False, index=df_sig.index)
    window = int(getattr(params, "range_window_bars", 20) or 20)
    slope_pct, _mask = calc_slope_mask(df_sig["close"], window, trend_slope_k)
    return (slope_pct >= trend_slope_k).fillna(False)


def _compute_threshold(
    prob_series: pd.Series,
    mask: pd.Series,
    mode: str,
    threshold: float,
    top_pct: float,
) -> float:
    if mode == "top_pct":
        pct = min(max(float(top_pct), 0.0), 1.0)
        if mask.any():
            return float(prob_series[mask].quantile(1.0 - pct))
        return 1.0
    return float(threshold)


def _apply_entry_ai_filter(
    df_sig: pd.DataFrame,
    params: RangeV3Params,
    entry_model_path: str,
    entry_model_mode: str,
    entry_model_threshold: float,
    entry_model_top_pct: float,
    trend_model_path: str,
    trend_model_mode: str,
    trend_model_threshold: float,
    trend_model_top_pct: float,
    trend_slope_k: float,
    entry_feature_cols: Optional[List[str]] = None,
) -> tuple[Optional[pd.Series], Dict[str, object]]:
    if (
        (not entry_model_path or entry_model_mode == "off")
        and (not trend_model_path or trend_model_mode == "off")
    ):
        return None, {}
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        raise SystemExit("CatBoost is required for entry-model gating") from exc

    feature_cols = entry_feature_cols or COMPACT_FEATURES
    feats = _build_entry_features(df_sig)
    x_all = _select_entry_features(feats, feature_cols)
    signal_mask = df_sig["v3_signal"].fillna(0).astype(int) != 0

    trend_mask = (
        _build_trend_mask(df_sig, params, trend_slope_k)
        if trend_model_path and trend_model_mode != "off"
        else pd.Series(False, index=df_sig.index)
    )
    base_mask = ~trend_mask

    allow_base = pd.Series(True, index=df_sig.index)
    base_threshold = None
    if entry_model_path and entry_model_mode != "off":
        model = CatBoostClassifier()
        model.load_model(entry_model_path)
        prob_base = pd.Series(model.predict_proba(x_all)[:, 1], index=df_sig.index)
        base_threshold = _compute_threshold(
            prob_base, signal_mask & base_mask, entry_model_mode, entry_model_threshold, entry_model_top_pct
        )
        allow_base = prob_base >= base_threshold

    allow_trend = pd.Series(True, index=df_sig.index)
    trend_threshold = None
    if trend_model_path and trend_model_mode != "off":
        model = CatBoostClassifier()
        model.load_model(trend_model_path)
        prob_trend = pd.Series(model.predict_proba(x_all)[:, 1], index=df_sig.index)
        trend_threshold = _compute_threshold(
            prob_trend, signal_mask & trend_mask, trend_model_mode, trend_model_threshold, trend_model_top_pct
        )
        allow_trend = prob_trend >= trend_threshold

    allow_mask = (trend_mask & allow_trend) | (base_mask & allow_base)
    stats = {
        "entry_ai_mode": entry_model_mode,
        "entry_ai_threshold": base_threshold,
        "entry_ai_top_pct": float(entry_model_top_pct),
        "entry_ai_trend_mode": trend_model_mode,
        "entry_ai_trend_threshold": trend_threshold,
        "entry_ai_trend_top_pct": float(trend_model_top_pct),
        "entry_ai_trend_slope_k": float(trend_slope_k),
        "entry_ai_signal_bars": int(signal_mask.sum()),
        "entry_ai_trend_bars": int(trend_mask.sum()),
        "entry_ai_allowed": int((allow_mask & signal_mask).sum()),
        "entry_ai_blocked": int((~allow_mask & signal_mask).sum()),
    }
    return allow_mask, stats

def _trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "qty",
                "pnl",
                "pnl_rel",
                "bars_held",
                "entry_reason",
                "exit_reason",
                "post_circuit_breaker",
                "entry_geo_class",
                "entry_geo_valid_box",
                "entry_geo_h_pct",
                "entry_atr_pct",
                "entry_slope_pct_per_bar",
                "entry_dist_L_pct",
                "entry_dist_U_pct",
            ]
        )
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "pnl": t.pnl,
                "pnl_rel": t.pnl_rel,
                            "bars_held": t.bars_held,
                "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
                "post_circuit_breaker": t.post_circuit_breaker,
                "entry_geo_class": t.entry_geo_class,
                "entry_geo_valid_box": t.entry_geo_valid_box,
                "entry_geo_h_pct": t.entry_geo_h_pct,
                "entry_atr_pct": t.entry_atr_pct,
                "entry_slope_pct_per_bar": t.entry_slope_pct_per_bar,
                "entry_dist_L_pct": t.entry_dist_L_pct,
                "entry_dist_U_pct": t.entry_dist_U_pct,
            }
        )
    return pd.DataFrame(rows)


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

    base = f"{out_prefix}_{symbol}_{interval}_{tag}"
    stats_path = f"{base}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    trades_df = _trades_to_df(trades)
    trades_path = f"{base}_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    snaps_path = f"{base}_snapshots.csv"
    sig_df.to_csv(snaps_path)

    debug_path = f"{base}_debug.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)

    print(
        f"[range-v3] {symbol}: trades={len(trades)} "
        f"pf={metrics['pf']:.3f} win_rate={metrics['win_rate']:.3f}"
    )
    return symbol, metrics, trades_df


def main(args):
    symbols: List[str] = list(args.symbols)
    interval: str = args.interval
    equity0: float = float(args.equity0)
    cfg_path: str = args.config_range
    out_prefix: str = args.out_prefix
    tag: str = getattr(args, "tag", "rangeV3")
    n_jobs = getattr(args, "n_jobs", None)
    entry_model_path = str(getattr(args, "entry_model_path", "") or "")
    entry_model_mode = str(getattr(args, "entry_model_mode", "off"))
    entry_model_threshold = float(getattr(args, "entry_model_threshold", 0.5))
    entry_model_top_pct = float(getattr(args, "entry_model_top_pct", 0.3))
    entry_model_trend_path = str(getattr(args, "entry_model_trend_path", "") or "")
    entry_model_trend_mode = str(getattr(args, "entry_model_trend_mode", "off"))
    entry_model_trend_threshold = float(getattr(args, "entry_model_trend_threshold", 0.5))
    entry_model_trend_top_pct = float(getattr(args, "entry_model_trend_top_pct", 0.3))
    entry_trend_slope_k = float(getattr(args, "entry_trend_slope_k", 0.0))
    no_hold_weekend = bool(getattr(args, "no_hold_weekend", False))
    entry_feature_cols = _parse_list(getattr(args, "entry_feature_include", ""))
    if not entry_feature_cols:
        entry_feature_cols = COMPACT_FEATURES

    params_cfg = _load_range_config(cfg_path)
    params = RangeV3Params(params_cfg)

    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    if any(str(s).lower() == "all" for s in symbols):
        symbols = _list_available_symbols(interval)

    all_symbol_metrics: List[Dict[str, Any]] = []
    all_trades_df_list: List[pd.DataFrame] = []

    if entry_trend_slope_k <= 0:
        entry_trend_slope_k = float(getattr(params, "slope_k", 0.0) or 0.0) * 0.5

    tasks = [
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
        )
        for symbol in symbols
    ]
    for symbol, metrics, trades_df in parallel_map(tasks, _run_symbol_task, n_jobs=n_jobs):
        if metrics is None:
            continue
        all_symbol_metrics.append(metrics)
        if not trades_df.empty:
            all_trades_df_list.append(trades_df)

    # Aggregated portfolio stats
    if all_trades_df_list:
        all_trades_df = pd.concat(all_trades_df_list, ignore_index=True)
        pnl_vals = all_trades_df["pnl"].tolist()
        base_all = f"{out_prefix}_ALL_{interval}_{tag}"
        stats_all_path = f"{base_all}_stats.json"
        trades_all_path = f"{base_all}_trades.csv"
        per_symbol_path = f"{base_all}_per_symbol_stats.csv"

        portfolio_stats = build_portfolio_stats(
            pnls=pnl_vals,
            equity0=equity0,
            symbols=[m["symbol"] for m in all_symbol_metrics],
        )

        with open(stats_all_path, "w", encoding="utf-8") as f:
            json.dump(portfolio_stats, f, ensure_ascii=False, indent=2)

        all_trades_df.to_csv(trades_all_path, index=False)

        per_symbol_df = pd.DataFrame(all_symbol_metrics)
        per_symbol_df.to_csv(per_symbol_path, index=False)
