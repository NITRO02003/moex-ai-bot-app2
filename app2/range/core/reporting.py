"""Reporting utilities for range-core backtester.

This module centralizes the conversion of trade records to DataFrames and the
writing of per-symbol and aggregated portfolio outputs to disk. By separating
reporting concerns from the backtest orchestration, it becomes easier to
maintain and test the output formats independently of the trading logic.
"""

from typing import Dict, List, Any

import json
import os

import pandas as pd

from .contracts import TradeRecord as Trade
from .portfolio import build_portfolio_stats


def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    """Convert a list of Trade records into a pandas DataFrame.

    The resulting DataFrame has stable column ordering matching the fields
    of the Trade dataclass. If the input list is empty, a DataFrame with
    the appropriate columns but no rows is returned.
    """
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


def save_symbol_outputs(
    out_prefix: str,
    symbol: str,
    interval: str,
    tag: str,
    metrics: Dict[str, Any],
    trades: List[Trade],
    debug_info: Dict[str, Any],
    sig_df: pd.DataFrame,
) -> None:
    """Persist per-symbol metrics, trades, snapshots and debug information to disk.

    The output files are named consistently with the existing backtest:
      - ``{out_prefix}_{symbol}_{interval}_{tag}_stats.json`` for metrics
      - ``{out_prefix}_{symbol}_{interval}_{tag}_trades.csv`` for trades
      - ``{out_prefix}_{symbol}_{interval}_{tag}_snapshots.csv`` for snapshots (signal DataFrame)
      - ``{out_prefix}_{symbol}_{interval}_{tag}_debug.json`` for debug information
    """
    base = f"{out_prefix}_{symbol}_{interval}_{tag}"
    # Ensure the output directory exists
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    # Write stats JSON
    stats_path = f"{base}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # Write trades CSV
    trades_df = trades_to_df(trades)
    trades_path = f"{base}_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    # Write snapshots (signal DataFrame) CSV
    snaps_path = f"{base}_snapshots.csv"
    sig_df.to_csv(snaps_path)
    # Write debug JSON
    debug_path = f"{base}_debug.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)


def save_portfolio_outputs(
    out_prefix: str,
    interval: str,
    tag: str,
    equity0: float,
    all_symbol_metrics: List[Dict[str, Any]],
    all_trades_df: pd.DataFrame,
) -> None:
    """Persist aggregated portfolio statistics, trades and per-symbol metrics to disk.

    The output files are named consistently with the existing backtest:
      - ``{out_prefix}_ALL_{interval}_{tag}_stats.json`` for aggregated stats
      - ``{out_prefix}_ALL_{interval}_{tag}_trades.csv`` for concatenated trades
      - ``{out_prefix}_ALL_{interval}_{tag}_per_symbol_stats.csv`` for per-symbol metrics
    """
    base_all = f"{out_prefix}_ALL_{interval}_{tag}"
    stats_all_path = f"{base_all}_stats.json"
    trades_all_path = f"{base_all}_trades.csv"
    per_symbol_path = f"{base_all}_per_symbol_stats.csv"
    # Compute aggregated portfolio stats using pnls from all trades
    pnl_vals = all_trades_df["pnl"].tolist() if not all_trades_df.empty else []
    portfolio_stats = build_portfolio_stats(
        pnls=pnl_vals,
        equity0=equity0,
        symbols=[m.get("symbol") for m in all_symbol_metrics],
    )
    # Ensure output directory exists
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    # Write aggregated stats
    with open(stats_all_path, "w", encoding="utf-8") as f:
        json.dump(portfolio_stats, f, ensure_ascii=False, indent=2)
    # Write concatenated trades
    all_trades_df.to_csv(trades_all_path, index=False)
    # Write per-symbol stats
    per_symbol_df = pd.DataFrame(all_symbol_metrics)
    per_symbol_df.to_csv(per_symbol_path, index=False)
