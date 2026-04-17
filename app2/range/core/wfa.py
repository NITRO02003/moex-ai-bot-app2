from __future__ import annotations

"""Walk‑forward analysis for trading results.

This module exposes a simple API for performing walk‑forward analysis on a set
of executed trades.  Given a CSV of trades produced by the core backtester
(`*_trades.csv`), it slices the trades into rolling windows and computes
portfolio‑level performance metrics for each window.  Metrics include total
profit and loss, total return, win rate, profit factor and maximum drawdown.

The primary entry point is :func:`run_walk_forward_analysis`.  There is
also a small CLI wrapper via ``python -m app2.range.core.wfa`` which is used
indirectly by the top‑level ``app2.cli`` through the ``range-core-wfa``
subcommand.

This tool is intended for research and robustness testing – it does not
attempt to re‑simulate trades, but rather consumes the output of an existing
backtest.  Windows are defined in days relative to the trade entry times.
"""

from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from .portfolio import build_portfolio_stats


def _infer_datetime_column(df: pd.DataFrame) -> str:
    """Heuristically determine the datetime column in a trades CSV.

    The core backtester writes trades with an ``entry_time`` column.  Fallbacks
    look for a ``begin`` or ``datetime`` column.  Raises if nothing is found.

    Parameters
    ----------
    df: pd.DataFrame
        Trades DataFrame loaded from CSV.

    Returns
    -------
    str
        Name of the datetime column.
    """
    for col in ("entry_time", "begin", "datetime"):
        if col in df.columns:
            return col
    raise ValueError("Cannot determine datetime column in trades file")


def _get_pnls(df: pd.DataFrame, equity0: float) -> List[float]:
    """Extract absolute PnL values from a trades DataFrame.

    If a ``pnl`` column is present it is assumed to contain absolute PnL per
    trade.  Otherwise, if a ``pnl_rel`` column is present it is multiplied by
    ``equity0`` to convert relative PnL to an absolute value.  Raises if
    neither column exists.

    Parameters
    ----------
    df: pd.DataFrame
        Trades DataFrame.
    equity0: float
        Initial equity used for scaling relative PnL into absolute PnL.

    Returns
    -------
    list of float
        List of absolute PnL values for each trade.
    """
    if "pnl" in df.columns:
        return df["pnl"].astype(float).tolist()
    if "pnl_rel" in df.columns:
        return (df["pnl_rel"].astype(float) * float(equity0)).tolist()
    raise ValueError("Trades file missing both 'pnl' and 'pnl_rel' columns")


def run_walk_forward_analysis(
    trades_path: str,
    equity0: float,
    window_days: int,
    step_days: int,
    out: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Run walk‑forward analysis on a set of trades.

    Parameters
    ----------
    trades_path: str
        Path to CSV file containing backtest trades.
    equity0: float
        Initial equity used to compute total returns.
    window_days: int
        Length of each rolling window in days.
    step_days: int
        Step between successive windows in days.
    out: Optional[str], default ``None``
        If provided, results are written to this CSV file (parent
        directories created as needed).

    Returns
    -------
    list of dict
        A list of dictionaries with metrics for each window.
    """
    df = pd.read_csv(trades_path)
    dt_col = _infer_datetime_column(df)
    # Parse and sort by datetime
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("Trades file is empty; cannot run WFA")

    start_date = df[dt_col].min().normalize()
    end_date_max = df[dt_col].max().normalize()

    results: List[Dict[str, float]] = []
    current_start = start_date
    window_delta = pd.Timedelta(days=window_days)
    step_delta = pd.Timedelta(days=step_days)

    while current_start <= end_date_max:
        window_end = current_start + window_delta
        # Select trades whose entry_time lies within the window
        mask = (df[dt_col] >= current_start) & (df[dt_col] < window_end)
        df_window = df.loc[mask]
        pnls = _get_pnls(df_window, equity0)
        symbols = df_window["symbol"].unique().tolist() if "symbol" in df_window.columns else []
        if pnls:
            stats = build_portfolio_stats(pnls, equity0, symbols)
            trades_count = len(pnls)
        else:
            stats = {
                "symbols": symbols,
                "total_pnl": 0.0,
                "total_return": 0.0,
                "win_rate": 0.0,
                "pf": 0.0,
                "max_drawdown": 0.0,
            }
            trades_count = 0
        results.append(
            {
                "window_start": current_start.isoformat(),
                "window_end": (window_end - pd.Timedelta(seconds=1)).isoformat(),
                "trades": trades_count,
                "total_pnl": float(stats["total_pnl"]),
                "total_return": float(stats["total_return"]),
                "win_rate": float(stats["win_rate"]),
                "pf": float(stats["pf"]),
                "max_drawdown": float(stats["max_drawdown"]),
            }
        )
        current_start += step_delta

    # Save if requested
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        return results
    return results


def main() -> None:
    """CLI entry point for the walk‑forward analysis tool."""
    import argparse

    parser = argparse.ArgumentParser(description="Walk‑forward analysis on backtest trades")
    parser.add_argument(
        "--trades-path",
        type=str,
        required=True,
        help="Path to CSV file with trades (from core backtest)",
    )
    parser.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity used for total return calculations",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        required=True,
        help="Length of rolling window in days",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        required=True,
        help="Step between successive windows in days",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV for WFA results (optional)",
    )
    args = parser.parse_args()
    run_walk_forward_analysis(
        trades_path=args.trades_path,
        equity0=args.equity0,
        window_days=args.window_days,
        step_days=args.step_days,
        out=args.out,
    )