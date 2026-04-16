"""
Bootstrap (resampling) analysis for trading strategies.

This module implements a simple bootstrap Monte Carlo analysis for the
range‑core strategy.  Given a CSV file of executed trades, it
randomly resamples the per‑trade PnL values with replacement to
generate a distribution of performance metrics such as total PnL,
total return, win rate, profit factor and maximum drawdown.  The
bootstrap is useful to understand the distribution of outcomes one
could expect due to randomness in trade sequencing.

Example usage::

    python -m app2.range.core.bootstrap \
        --trades-path out/range_v3_baseline_ALL_30min_baseline_trades.csv \
        --equity0 1000000 \
        --samples 1000 \
        --out out/range_v3_bootstrap.json

This will perform 1 000 bootstrap resamples and write summary
percentiles (5th/50th/95th) of each metric into the specified JSON file.

The core function ``run_bootstrap_analysis`` can also be imported
programmatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .portfolio import build_portfolio_stats


def _get_pnls_from_trades(trades_path: str) -> List[float]:
    df = pd.read_csv(trades_path)
    if "pnl" not in df.columns:
        raise ValueError(f"Trades file {trades_path} must contain a 'pnl' column")
    return df["pnl"].astype(float).tolist()


def _bootstrap_metric_distribution(
    pnls: Iterable[float],
    equity0: float,
    symbols: List[str],
    n_samples: int,
    sample_size: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Run bootstrap resampling on a list of per‑trade PnL values.

    Parameters
    ----------
    pnls:
        Iterable of per‑trade PnL values.

    equity0:
        Initial equity baseline.  This is used to compute total return.

    symbols:
        List of symbols active in the original run.  The number of symbols
        influences the denominator for total return in
        ``build_portfolio_stats``.

    n_samples:
        Number of bootstrap resamples.

    sample_size:
        Size of each resample.  If None, uses ``len(pnls)``.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary mapping metric names to lists of values of length
        ``n_samples``.
    """
    pnls_list = list(pnls)
    if not pnls_list:
        raise ValueError("Cannot bootstrap empty pnl list")
    n = len(pnls_list)
    sample_size = sample_size or n
    results: Dict[str, List[float]] = {
        "total_pnl": [],
        "total_return": [],
        "win_rate": [],
        "pf": [],
        "max_drawdown": [],
    }
    for _ in range(n_samples):
        # sample with replacement
        sample = np.random.choice(pnls_list, size=sample_size, replace=True)
        stats = build_portfolio_stats(sample, equity0, symbols or ["_agg"])
        results["total_pnl"].append(float(stats["total_pnl"]))
        results["total_return"].append(float(stats["total_return"]))
        results["win_rate"].append(float(stats["win_rate"]))
        results["pf"].append(float(stats["pf"]))
        results["max_drawdown"].append(float(stats["max_drawdown"]))
    return results


def run_bootstrap_analysis(
    trades_path: str,
    equity0: float,
    n_samples: int = 1000,
    sample_size: Optional[int] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Perform bootstrap resampling on trades and compute percentile bands
    for portfolio metrics.

    Parameters
    ----------
    trades_path:
        Path to the trades CSV.  Must contain at least a ``pnl`` column
        and optionally a ``symbol`` column.

    equity0:
        Initial equity baseline used in ``build_portfolio_stats``.

    n_samples:
        Number of bootstrap resamples.  Defaults to 1000.

    sample_size:
        Size of each resample.  If None, uses the number of trades in
        the dataset.

    out_path:
        Optional path to write the summary JSON.  The summary contains
        percentile (5th, 50th, 95th) values for each metric.  If
        omitted, nothing will be written.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary mapping metric names to a dict of percentiles:
        ``{"5": ..., "50": ..., "95": ...}``.
    """
    df = pd.read_csv(trades_path)
    if "pnl" not in df.columns:
        raise ValueError(f"Trades file {trades_path} must include a 'pnl' column")
    pnls = df["pnl"].astype(float).tolist()
    symbols = list(df["symbol"].unique()) if "symbol" in df.columns else []
    dist = _bootstrap_metric_distribution(
        pnls=pnls,
        equity0=equity0,
        symbols=symbols,
        n_samples=n_samples,
        sample_size=sample_size,
    )
    # Compute percentiles
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in dist.items():
        arr = np.array(values, dtype=float)
        summary[key] = {
            "5": float(np.percentile(arr, 5)),
            "50": float(np.percentile(arr, 50)),
            "95": float(np.percentile(arr, 95)),
        }
    if out_path:
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Perform bootstrap analysis on trading strategy trades."
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
        "--samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Size of each resample. If omitted, uses the number of trades in the dataset."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON file for percentile summary.",
    )
    args = parser.parse_args()
    run_bootstrap_analysis(
        trades_path=args.trades_path,
        equity0=args.equity0,
        n_samples=args.samples,
        sample_size=args.sample_size,
        out_path=args.out,
    )
    print(
        f"[bootstrap] Completed bootstrap analysis on {args.trades_path}; summary saved to {args.out if args.out else 'not saved'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())