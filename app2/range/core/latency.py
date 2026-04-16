"""Latency sensitivity analysis for trading strategies.

This module implements a simple analysis to assess how sensitive a trading
strategy is to execution latency.  Given a CSV of executed trades, it
sorts the trades chronologically and then progressively discards the
first *n* trades to simulate a latency of *n* bars.  For each delay
value the portfolio statistics (total PnL, total return, win rate,
profit factor, max drawdown) are computed using the remaining trades.

Example usage::

    python -m app2.range.core.latency \
        --trades-path out/range_v3_baseline_ALL_30min_baseline_trades.csv \
        --equity0 1000000 \
        --delays 1,2,5 \
        --out out/range_v3_latency.json

This will compute metrics for delays of 1, 2 and 5 bars and save a JSON
report mapping each delay to its metrics.

The main function ``run_latency_sensitivity`` can also be imported
programmatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .portfolio import build_portfolio_stats


def run_latency_sensitivity(
    trades_path: str,
    equity0: float,
    delays: Iterable[int],
    out_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute latency sensitivity by dropping the first *n* trades.

    Parameters
    ----------
    trades_path:
        Path to the aggregated trades CSV.  Must contain at least
        ``pnl`` and a datetime column (``entry_time`` is preferred).

    equity0:
        Initial equity baseline used in ``build_portfolio_stats``.

    delays:
        Iterable of integers representing the number of initial trades to
        discard when computing portfolio statistics.  For example,
        ``[0, 1, 2]`` will compute metrics for no delay, a delay of 1
        trade and a delay of 2 trades.

    out_path:
        Optional path to write the results JSON.  If provided, a JSON
        mapping each delay to its metrics will be written.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary mapping each delay (converted to string) to a
        dictionary of portfolio metrics:
        ``{'total_pnl', 'total_return', 'win_rate', 'pf', 'max_drawdown', 'coverage'}``.
        The ``coverage`` key indicates the fraction of trades retained.
    """
    df = pd.read_csv(trades_path)
    if df.empty:
        raise ValueError(f"Trades file {trades_path} is empty")
    # Determine time ordering
    # Prefer entry_time, but fallback to exit_time/timestamp/datetime
    time_cols = [c for c in df.columns if c in ('entry_time', 'exit_time', 'timestamp', 'datetime', 'ts')]
    if time_cols:
        time_col = time_cols[0]
    else:
        # fallback: any column ending with _time or _timestamp
        time_col = None
        for col in df.columns:
            lc = str(col).lower()
            if lc.endswith('_time') or lc.endswith('_timestamp'):
                time_col = col
                break
    if not time_col:
        raise ValueError("Could not determine a datetime column in the trades file; expected 'entry_time' or similar.")
    df[time_col] = pd.to_datetime(df[time_col], utc=False, errors='coerce')
    df = df.sort_values(time_col).reset_index(drop=True)
    if "pnl" not in df.columns:
        raise ValueError(f"Trades file {trades_path} must include a 'pnl' column representing per-trade PnL.")
    pnls = df["pnl"].astype(float).tolist()
    symbols = list(df["symbol"].unique()) if "symbol" in df.columns else []
    results: Dict[str, Dict[str, float]] = {}
    total_trades = len(pnls)
    for d in delays:
        if d < 0:
            raise ValueError("Delay values must be non-negative integers")
        if d >= total_trades:
            # No trades left, skip
            continue
        # Drop first d trades
        pnls_d = pnls[d:]
        # Compute stats
        stats = build_portfolio_stats(pnls_d, equity0, symbols or ["_agg"])
        coverage = len(pnls_d) / total_trades
        results[str(d)] = {
            "total_pnl": float(stats["total_pnl"]),
            "total_return": float(stats["total_return"]),
            "win_rate": float(stats["win_rate"]),
            "pf": float(stats["pf"]),
            "max_drawdown": float(stats["max_drawdown"]),
            "coverage": coverage,
        }
    if out_path:
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the sensitivity of a strategy to execution latency."
    )
    parser.add_argument(
        "--trades-path",
        type=str,
        required=True,
        help="Path to aggregated trades CSV (must contain a 'pnl' column).",
    )
    parser.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity baseline used for total return. Defaults to 1,000,000.",
    )
    parser.add_argument(
        "--delays",
        type=str,
        default="1,2",
        help="Comma-separated list of delay values (number of trades to drop). Include 0 for baseline.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON file for latency sensitivity results.",
    )
    args = parser.parse_args()
    # Parse delays
    delays: List[int] = []
    for part in args.delays.split(","):
        part = part.strip()
        if part:
            try:
                delays.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid delay value '{part}', must be an integer")
    # Ensure baseline included
    if 0 not in delays:
        delays.insert(0, 0)
    run_latency_sensitivity(
        trades_path=args.trades_path,
        equity0=args.equity0,
        delays=delays,
        out_path=args.out,
    )
    print(
        f"[latency] Completed latency sensitivity analysis on {args.trades_path}; "
        f"results saved to {args.out if args.out else 'not saved'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())