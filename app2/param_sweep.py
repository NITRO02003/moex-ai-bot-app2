"""Parameter sweep for rule‑based strategies.

This module implements a simple grid search over selected hyper‑parameters
for the trend, mean‑reversion and breakout strategies.  It runs the
rule‑based backtest for each combination of parameters on a set of
symbols and writes the resulting metrics to a CSV file.  The aim is
to provide a baseline for tuning without depending on ML models.

Usage (as a script)::

    python -m app2.param_sweep --strategy trend --symbols SBER GAZP \
        --out out/reports/trend_sweep.csv

The default parameter grids are defined in the module.  You can
customise them by editing the dictionaries ``TREND_GRID``,
``MEANREV_GRID`` and ``BREAKOUT_GRID``.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

import os
from .rule_strategies import (
    TrendParams,
    MeanRevParams,
    BreakoutParams,
    generate_trend_signals,
    generate_meanrev_signals,
    generate_breakout_signals,
)
from .rule_backtest import RuleBtParams, run_rule_symbol
from .parallel import parallel_map, default_n_jobs

# Default hyper‑parameter grids for each strategy.  Adjust these lists
# to explore different values.  Note that some parameters (such as
# EMA lengths) are kept constant here; they can be extended in future
# iterations.
TREND_GRID: Dict[str, List] = {
    "trend_thr": [2.0, 2.5, 3.0],
    "min_gap_bars": [20, 25, 30],
}
TREND_BT_GRID: Dict[str, List] = {
    "risk_per_trade": [0.001, 0.0015, 0.002],
    "sl_atr_mult": [1.5, 2.0, 2.5],
}

MEANREV_GRID: Dict[str, List] = {
    "rsi_low": [10.0, 15.0, 20.0],
    "rsi_high": [80.0, 85.0, 90.0],
    "min_gap_bars": [10, 15, 20],
}
MEANREV_BT_GRID: Dict[str, List] = {
    "risk_per_trade": [0.001, 0.0015, 0.002],
    "sl_atr_mult": [1.5, 2.0, 2.5],
}

BREAKOUT_GRID: Dict[str, List] = {
    "channel_len": [20, 30, 40],
    "confirm_bars": [1, 2, 3],
    "min_gap_bars": [10, 15, 20],
}
BREAKOUT_BT_GRID: Dict[str, List] = {
    "risk_per_trade": [0.001, 0.0015, 0.002],
    "sl_atr_mult": [1.5, 2.0, 2.5],
}

# Simple in-memory cache for price data to avoid repeated file loads within
# worker processes.  Each process will maintain its own cache; this reduces
# disk I/O significantly during large sweeps.
_PRICE_CACHE: Dict[str, pd.DataFrame] = {}


def _load_prices_cached(sym: str) -> pd.DataFrame | None:
    """Load price data for a symbol with caching.

    First checks the in-process cache; if the data is not already loaded,
    attempts to read from ``data/{sym}.csv`` or ``{sym}.csv`` in the current
    directory.  Returns the DataFrame or None if not found.
    """
    if sym in _PRICE_CACHE:
        return _PRICE_CACHE[sym]
    # Search for the CSV file
    for candidate in [os.path.join("data", f"{sym}.csv"), f"{sym}.csv"]:
        if os.path.exists(candidate):
            try:
                df = pd.read_csv(candidate)
            except Exception:
                return None
            _PRICE_CACHE[sym] = df
            return df
    return None


def _param_product(grid: Dict[str, List]) -> Iterable[Dict[str, any]]:
    """Yield dicts for each combination of parameters in a grid.

    Parameters
    ----------
    grid : dict
        Mapping from parameter names to lists of values.

    Yields
    ------
    dict
        Mapping of parameter names to a chosen value for that combination.
    """
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def _sweep_task_runner(task: Dict[str, any]) -> Dict[str, any] | None:
    """Run a single combination of parameters on a symbol.

    This helper function is executed in parallel across multiple processes.
    It loads the price data, generates signals using the appropriate rule
    strategy, runs the backtest with the provided risk settings and returns
    a row of results.  If data cannot be loaded or an exception occurs,
    None is returned and the row is skipped.
    """
    strat = task["strat"]
    param_vals = task["param_vals"]
    bt_vals = task["bt_vals"]
    sym = task["sym"]
    equity0 = task["equity0"]
    try:
        # Load prices with caching
        prices = _load_prices_cached(sym)
        if prices is None:
            return None
        # Instantiate strategy parameters and select signal generator
        if strat == "trend":
            s_params = TrendParams(**param_vals)
            signal_func = generate_trend_signals
        elif strat == "meanrev":
            s_params = MeanRevParams(**param_vals)
            signal_func = generate_meanrev_signals
        elif strat == "breakout":
            s_params = BreakoutParams(**param_vals)
            signal_func = generate_breakout_signals
        else:
            return None
        side = signal_func(prices, s_params)
        # Instantiate backtest parameters
        bt_params = RuleBtParams()
        for k, v in bt_vals.items():
            setattr(bt_params, k, v)
        res = run_rule_symbol(prices, side, bt_params, equity0)
        metrics = res.get("metrics", {}) or {}
        row = {
            "strategy": strat,
            "symbol": sym,
            **param_vals,
            **bt_vals,
            "total_return": metrics.get("total_return", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "avg_trade": metrics.get("avg_trade", 0.0),
            "trade_count": metrics.get("trade_count", 0),
        }
        return row
    except Exception:
        return None


def _run_rule_sweep(
    symbols: List[str],
    equity0: float,
    strat: str,
    param_grid: Dict[str, List],
    bt_grid: Dict[str, List],
    out_path: Path,
    n_jobs: int | None = None,
) -> int:
    """Run a parameter sweep for a single rule‑based strategy using parallel execution.

    This helper constructs all combinations of signal and backtest parameters
    for the given symbols, evaluates them in parallel using a process pool
    and writes the results to a CSV file.  The number of worker processes
    can be specified via ``n_jobs``; if ``None`` or <=0 it defaults to
    ``os.cpu_count() - 1``.

    Returns the number of result rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build list of tasks (one per combination and symbol)
    tasks: List[Dict[str, any]] = []
    for param_vals in _param_product(param_grid):
        for bt_vals in _param_product(bt_grid):
            for sym in symbols:
                tasks.append(
                    {
                        "strat": strat,
                        "param_vals": param_vals,
                        "bt_vals": bt_vals,
                        "sym": sym,
                        "equity0": equity0,
                    }
                )
    # Evaluate tasks in parallel
    rows = parallel_map(tasks, _sweep_task_runner, n_jobs=n_jobs)
    # Write non‑None rows to CSV
    rows_written = 0
    fieldnames = None
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for row in rows:
            if row is None:
                continue
            if writer is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            rows_written += 1
    return rows_written


def run_sweep(
    strategy: str,
    symbols: List[str],
    out_csv: Path,
    equity0: float = 1_000_000.0,
    n_jobs: int | None = None,
) -> Dict[str, any]:
    """Entry point to run parameter sweeps for rule strategies.

    Parameters
    ----------
    strategy : str
        One of ``trend``, ``meanrev``, ``breakout``.
    symbols : list of str
        Symbols (tickers) to run the sweep on.
    out_csv : pathlib.Path
        File path where the CSV results will be written.  Parent directories
        are created automatically.
    equity0 : float, default 1e6
        Starting equity for each backtest.

    Returns
    -------
    dict
        Summary with number of rows written.
    """
    if strategy == "trend":
        n_rows = _run_rule_sweep(
            symbols,
            equity0,
            "trend",
            TREND_GRID,
            TREND_BT_GRID,
            out_csv,
            n_jobs=n_jobs,
        )
    elif strategy == "meanrev":
        n_rows = _run_rule_sweep(
            symbols,
            equity0,
            "meanrev",
            MEANREV_GRID,
            MEANREV_BT_GRID,
            out_csv,
            n_jobs=n_jobs,
        )
    elif strategy == "breakout":
        n_rows = _run_rule_sweep(
            symbols,
            equity0,
            "breakout",
            BREAKOUT_GRID,
            BREAKOUT_BT_GRID,
            out_csv,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unsupported strategy for sweep: {strategy}")
    return {"rows_written": n_rows, "csv": str(out_csv)}


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parameter sweep for rule‑based strategies")
    ap.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "breakout"],
        required=True,
        help="Which rule strategy to sweep",
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of symbols (tickers) to include in the sweep",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="out/reports/param_sweep_results.csv",
        help="Path to the CSV file for results",
    )
    ap.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Starting equity for each backtest",
    )
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    out_path = Path(args.out)
    summary = run_sweep(args.strategy, args.symbols, out_path, equity0=args.equity0)
    print(summary)


if __name__ == "__main__":  # pragma: no cover
    main()