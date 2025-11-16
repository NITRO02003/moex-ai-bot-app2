"""Walk-forward (forward) testing utilities for rule and regime strategies.

This module implements a simple walk-forward testing framework: given a
complete time series of prices and signals (for rule strategies) or
mechanisms to generate signals on the fly (for regime strategies), it
splits the data into sequential non-overlapping windows and runs a
backtest on each test window.  Results can then be analysed for
stability across time.

Forward testing helps to detect overfitting by evaluating strategies on
out-of-sample periods that follow the training period.  While the
implementation here does not retrain or recalibrate rule parameters
between windows, it provides a baseline for observing performance drift.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

# Import centralised data utilities to avoid CLI circular dependencies
from .data_utils import load_prices, resample_prices
from .rule_strategies import (
    TrendParams,
    MeanRevParams,
    BreakoutParams,
    generate_trend_signals,
    generate_meanrev_signals,
    generate_breakout_signals,
)
from .rule_backtest import RuleBtParams, run_rule_symbol


def _forward_splits(n_bars: int, train_window: int, test_window: int, step: Optional[int] = None) -> List[int]:
    """Compute starting indices for forward-test windows.

    Parameters
    ----------
    n_bars : int
        Total number of bars in the dataset.
    train_window : int
        Length of the training portion (number of bars).  The first test
        window starts immediately after the training window.
    test_window : int
        Length of each test window (number of bars).
    step : int, optional
        Step size between windows.  If None, uses ``test_window`` (i.e. no
        overlap between test windows).

    Returns
    -------
    list of int
        Starting indices for each test window.
    """
    if step is None or step <= 0:
        step = test_window
    starts: List[int] = []
    i = train_window
    while i + test_window <= n_bars:
        starts.append(i)
        i += step
    return starts


def _metrics_from_backtest(res: Dict) -> Dict[str, float]:
    """Extract metrics dictionary from backtest result, ensuring defaults."""
    metrics = res.get("metrics", {}) or {}
    return {
        "total_return": metrics.get("total_return", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "avg_trade": metrics.get("avg_trade", 0.0),
        "trade_count": float(metrics.get("trade_count", 0)),
    }


def run_forward_test(
    strategy: str,
    symbols: Iterable[str],
    interval: str = "10min",
    train_window: int = 1000,
    test_window: int = 200,
    step: Optional[int] = None,
    equity0: float = 1_000_000.0,
) -> Dict[str, any]:
    """Perform walk-forward testing for the given symbols and strategy.

    Parameters
    ----------
    strategy : str
        One of ``"trend"``, ``"meanrev"``, ``"breakout"`` or ``"regime"``.
    symbols : iterable of str
        List of ticker symbols to test.
    interval : str, default "10min"
        Bar interval to which raw prices should be resampled.
    train_window : int, default 1000
        Number of bars to use as the initial training period.  Test
        windows start after this period.  Note that no actual training is
        performed for rule strategies; this parameter simply controls
        where the first test window begins.
    test_window : int, default 200
        Number of bars in each test window.
    step : int, optional
        Step size between the starts of successive test windows.  If
        omitted, uses ``test_window`` (non-overlapping windows).
    equity0 : float, default 1e6
        Starting equity for each backtest.

    Returns
    -------
    dict
        A dictionary containing the test specification and results.  The
        ``symbols`` key holds per-symbol lists of metrics per window and
        aggregate averages.
    """
    results: Dict[str, any] = {}
    results["strategy"] = strategy
    results["interval"] = interval
    results["train_window"] = train_window
    results["test_window"] = test_window
    results["step"] = step if step is not None else test_window
    sym_results: Dict[str, any] = {}
    for sym in symbols:
        # Load and resample price data
        try:
            raw = load_prices(sym)
            prices = resample_prices(raw, interval)
        except Exception:
            sym_results[sym] = {"error": f"failed to load/resample {sym}"}
            continue
        n_bars = len(prices)
        if n_bars < train_window + test_window:
            sym_results[sym] = {"error": "not enough data"}
            continue
        # Precompute signals if rule strategy
        side_series: Optional[pd.Series] = None
        if strategy == "trend":
            side_series = generate_trend_signals(prices, TrendParams())
        elif strategy == "meanrev":
            side_series = generate_meanrev_signals(prices, MeanRevParams())
        elif strategy == "breakout":
            side_series = generate_breakout_signals(prices, BreakoutParams())
        # For regime strategy, we will run each window on the fly
        starts = _forward_splits(n_bars, train_window, test_window, step)
        per_window: List[Dict[str, float]] = []
        for start in starts:
            # extract test window
            end = start + test_window
            df_sub = prices.iloc[start:end].reset_index(drop=True)
            if strategy == "regime":
                try:
                    from .regime_rule_backtest import RegimeRuleBtParams, run_symbol as run_regime_rule_symbol  # type: ignore
                except Exception as imp_err:
                    per_window.append({"error": f"failed import regime module: {imp_err}"})
                    continue
                params = RegimeRuleBtParams()
                res = run_regime_rule_symbol(df_sub, params, equity0=equity0)
                metrics = _metrics_from_backtest(res)
                per_window.append(metrics)
            else:
                # rule strategy
                assert side_series is not None
                side_sub = side_series.iloc[start:end].reset_index(drop=True)
                bt_params = RuleBtParams()
                res = run_rule_symbol(df_sub, side_sub, bt_params, equity0)
                metrics = _metrics_from_backtest(res)
                per_window.append(metrics)
        # Compute aggregate averages
        if per_window and isinstance(per_window[0], dict) and "error" not in per_window[0]:
            # Compute mean metrics across windows
            totals = {}
            count = 0
            for m in per_window:
                for k, v in m.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                count += 1
            avg_metrics = {k: (v / count if count else 0.0) for k, v in totals.items()}
            sym_results[sym] = {
                "windows": per_window,
                "average": avg_metrics,
                "num_windows": len(per_window),
            }
        else:
            sym_results[sym] = {"windows": per_window, "num_windows": len(per_window)}
    results["symbols"] = sym_results
    return results