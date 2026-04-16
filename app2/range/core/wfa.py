"""
Walk‑Forward Analysis (WFA) utilities for the range‑core backtester.

This module provides a function to compute rolling performance
statistics across a set of executed trades.  It is intended to help
assess the temporal stability of a trading strategy by splitting a
history of trades into a series of overlapping windows and
computing standard portfolio metrics for each window.  The output
can then be analysed to understand how returns, win rates and
drawdown evolve through time.

Typical usage from the command line::

    python -m app2.range.core.wfa \
        --trades-path out/range_v3_baseline_ALL_30min_baseline_trades.csv \
        --equity0 1000000 \
        --window-days 90 \
        --step-days 30 \
        --out out/range_v3_wfa.csv

This will read the aggregated trades CSV, split the timeline into
90‑day windows advanced by 30 days, compute portfolio statistics
for each window and save the results to ``out/range_v3_wfa.csv``.

The function ``run_walk_forward_analysis`` can also be imported and
called programmatically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .portfolio import build_portfolio_stats


@dataclass
class WFAWindowResult:
    """Holds the results for a single walk‑forward window."""

    window_start: datetime
    window_end: datetime
    num_trades: int
    total_pnl: float
    total_return: float
    win_rate: float
    pf: float
    max_drawdown: float

    def to_dict(self) -> dict:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "num_trades": self.num_trades,
            "total_pnl": self.total_pnl,
            "total_return": self.total_return,
            "win_rate": self.win_rate,
            "pf": self.pf,
            "max_drawdown": self.max_drawdown,
        }


def _determine_datetime_column(df: pd.DataFrame) -> str:
    """
    Attempt to guess which column of a trades DataFrame contains the exit
    timestamps.  Returns the column name.  Raises ValueError if none
    could be found.

    Preference order:
      1. ``exit_time``
      2. ``exit_timestamp``
      3. ``timestamp``
      4. ``datetime``
      5. ``ts``
      6. any column ending with ``_time`` or ``_timestamp``

    A ValueError is raised if no datetime‑like column is present.
    """
    candidates = [
        "exit_time",
        "exit_timestamp",
        "timestamp",
        "datetime",
        "ts",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any column ending with _time or _timestamp
    for col in df.columns:
        if str(col).lower().endswith("_time") or str(col).lower().endswith(
            "_timestamp"
        ):
            return col
    raise ValueError(
        "Could not determine a datetime column. The trades file must include a column"
        " representing exit timestamps, such as 'exit_time'."
    )


def run_walk_forward_analysis(
    trades_path: str,
    equity0: float,
    window_days: int,
    step_days: int,
    out_path: Optional[str] = None,
) -> List[WFAWindowResult]:
    """
    Perform a walk‑forward analysis on a trades CSV.

    Parameters
    ----------
    trades_path:
        Path to the trades CSV. The file must contain a column
        representing exit timestamps (see ``_determine_datetime_column``)
        and a ``pnl`` column representing per‑trade absolute PnL.

    equity0:
        Initial equity baseline used to compute total returns.  When
        constructing portfolio metrics for a window, the number of
        active symbols within that window will be taken into account.

    window_days:
        Length of each window in days.

    step_days:
        Step size between windows in days.  A new window starts
        ``step_days`` after the previous window start.

    out_path:
        Optional path to write the results CSV.  If ``None`` (default)
        then no file will be written.

    Returns
    -------
    List[WFAWindowResult]
        A list of window results.  Each result contains the window
        start/end and computed portfolio statistics.
    """
    trades_df = pd.read_csv(trades_path)
    if trades_df.empty:
        raise ValueError(f"Trades file {trades_path} is empty")
    time_col = _determine_datetime_column(trades_df)
    # Parse datetime; assume UTC if no timezone info
    trades_df[time_col] = pd.to_datetime(trades_df[time_col], utc=False)
    trades_df = trades_df.sort_values(time_col).reset_index(drop=True)
    # Ensure pnl exists
    if "pnl" not in trades_df.columns:
        raise ValueError(
            f"Trades file {trades_path} must include a 'pnl' column representing per‑trade PnL."
        )
    # Precompute symbol lists per row for portfolio stats
    # Each row corresponds to one trade on a symbol; we will compute metrics across
    # all trades in a window and treat all unique symbols as active
    results: List[WFAWindowResult] = []
    start_dt = trades_df[time_col].iloc[0]
    end_dt = trades_df[time_col].iloc[-1]
    delta_window = timedelta(days=window_days)
    delta_step = timedelta(days=step_days)
    window_start = start_dt
    while window_start <= end_dt:
        window_end = window_start + delta_window
        mask = (trades_df[time_col] >= window_start) & (trades_df[time_col] < window_end)
        trades_win = trades_df.loc[mask]
        if not trades_win.empty:
            pnls = trades_win["pnl"].astype(float).tolist()
            symbols = list(trades_win["symbol"].unique()) if "symbol" in trades_win.columns else []
            # Compute portfolio stats; use equity0 and number of symbols in window
            stats = build_portfolio_stats(pnls, equity0, symbols or ["_agg"])
            result = WFAWindowResult(
                window_start=window_start.to_pydatetime(),
                window_end=window_end.to_pydatetime(),
                num_trades=len(trades_win),
                total_pnl=float(stats["total_pnl"]),
                total_return=float(stats["total_return"]),
                win_rate=float(stats["win_rate"]),
                pf=float(stats["pf"]),
                max_drawdown=float(stats["max_drawdown"]),
            )
            results.append(result)
        window_start = window_start + delta_step
    # Optionally write to file
    if out_path:
        out_path = str(out_path)
        # Convert to DataFrame
        df_out = pd.DataFrame([r.to_dict() for r in results])
        df_out.to_csv(out_path, index=False)
    return results


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Perform a walk‑forward analysis on a set of trades."
    )
    parser.add_argument(
        "--trades-path",
        type=str,
        required=True,
        help="Path to the aggregated trades CSV (e.g. *_ALL_*_trades.csv).",
    )
    parser.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity used for calculating total returns. Defaults to 1,000,000.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Length of each window in days. Defaults to 90 days.",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=30,
        help="Step size between windows in days. Defaults to 30 days.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path for saving the walk‑forward results as CSV.",
    )
    args = parser.parse_args()
    run_walk_forward_analysis(
        trades_path=args.trades_path,
        equity0=args.equity0,
        window_days=args.window_days,
        step_days=args.step_days,
        out_path=args.out,
    )
    print(
        f"[wfa] Completed walk‑forward analysis; results saved to {args.out if args.out else 'no file'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())