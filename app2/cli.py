"""Command line interface for rule‑based and regime‑aware strategies.

This module exposes two primary commands:

``rule-backtest``
    Run a single rule‑based strategy (trend, meanrev or breakout) on one or
    more symbols.  It loads local CSV price files, generates signals
    using functions from :mod:`app2.rule_strategies` and then
    executes the backtest via :func:`app2.rule_backtest.run_rule_symbol`.

``regime-rule-backtest``
    Run a regime‑aware strategy: detect the market regime using the
    default settings in :mod:`app2.regime_detector` and then choose a
    rule strategy for each regime.  By default the trend strategy is
    used in trend regime, mean‑reversion in range regime and breakout
    in high volatility regime.  Results are summarised per symbol.

Example usage::

    python -m app2.cli rule-backtest --strategy trend --symbols SBER GAZP
    python -m app2.cli regime-rule-backtest --symbols SBER GAZP

The commands write JSON summary files into ``out/reports`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytz

from .rule_strategies import (
    TrendParams,
    MeanRevParams,
    BreakoutParams,
    generate_trend_signals,
    generate_meanrev_signals,
    generate_breakout_signals,
)
from .rule_backtest import RuleBtParams, run_rule_symbol
from .param_sweep import run_sweep as run_param_sweep
from .data_pipeline import process_all as process_data_all
from .forward_test import run_forward_test

# Import centralised data utilities so that our CLI wrappers delegate
from .data_utils import load_prices as _load_prices_from_utils
from .data_utils import resample_prices as _resample_prices_from_utils


def _load_prices(symbol: str) -> pd.DataFrame:
    """Load price data for a given symbol from a CSV file.

    This wrapper delegates to :func:`app2.data_utils.load_prices`, which
    searches for ``{symbol}.csv`` in a set of default directories (by
    default ``data`` and the current working directory) and returns a
    DataFrame.  It is kept for backward compatibility; new modules
    should import :func:`app2.data_utils.load_prices` directly.
    """
    # Delegate to the central utility.  The utils function already
    # raises FileNotFoundError if the file is missing.
    return _load_prices_from_utils(symbol)


def _resample_prices(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate a price dataframe to a different interval.

    This wrapper delegates to :func:`app2.data_utils.resample_prices`, which
    handles timezone conversion to Europe/Moscow and OHLCV aggregation.  It
    is retained for backward compatibility; new modules should import
    :func:`app2.data_utils.resample_prices` directly.

    Parameters
    ----------
    df : pandas.DataFrame
        Input OHLCV dataframe with columns ``open``, ``high``, ``low``, ``close`` and optionally ``volume``.
    interval : str
        Pandas offset alias such as ``'10min'``, ``'30min'`` or ``'1h'``.  The default ``'10min'`` returns
        the input unchanged aside from timezone conversion.

    Returns
    -------
    pandas.DataFrame
        Resampled dataframe with aggregated OHLCV columns and a ``begin`` column for the new interval.
    """
    return _resample_prices_from_utils(df, interval)


def _ensure_dirs(path: Path) -> None:
    """Create parent directories for a given output path."""
    outdir = path.parent
    outdir.mkdir(parents=True, exist_ok=True)


def cmd_rule_backtest(args: argparse.Namespace) -> None:
    """Run a simple rule‑based backtest for each symbol."""
    strategy = args.strategy
    equity0 = args.equity0
    results: dict[str, dict] = {}
    for sym in args.symbols:
        try:
            prices = _load_prices(sym)
            # Resample prices to desired interval and convert to Moscow timezone
            prices = _resample_prices(prices, args.interval)
        except Exception as e:
            print(f"[warn] failed to load data for {sym}: {e}", file=sys.stderr)
            continue
        # Generate side signals according to the chosen strategy
        if strategy == "trend":
            side = generate_trend_signals(prices, TrendParams())
        elif strategy == "meanrev":
            side = generate_meanrev_signals(prices, MeanRevParams())
        elif strategy == "breakout":
            side = generate_breakout_signals(prices, BreakoutParams())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        # Run backtest
        rp = RuleBtParams()
        res = run_rule_symbol(prices, side, rp, equity0)
        results[sym] = res.get("metrics", {})
    # Write summary to file
    out_path = Path(args.out) if args.out else Path("out/reports") / f"rule_{strategy}_backtest_summary.json"
    _ensure_dirs(out_path)
    payload = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "strategy": strategy,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload))


def cmd_regime_rule_backtest(args: argparse.Namespace) -> None:
    """Run a regime‑aware rule backtest for each symbol.

    The import of `RegimeRuleBtParams` and `run_symbol` from
    ``app2.regime_rule_backtest`` is performed inside the function to
    avoid import errors when only the simple rule backtests are
    executed.  This lazy import ensures that missing or broken regime
    modules do not prevent usage of the rule‑based commands.
    """
    # Perform the import lazily here to avoid import errors if
    # regime_rule_backtest is missing or malformed.  If it cannot be
    # imported, report the error and abort the regime run.
    try:
        from .regime_rule_backtest import RegimeRuleBtParams, run_symbol as run_regime_rule_symbol  # type: ignore
    except Exception as imp_err:
        print(
            f"[error] failed to import regime_rule_backtest: {imp_err}.\n"
            "Ensure that app2/regime_rule_backtest.py is present and up to date.",
            file=sys.stderr,
        )
        return
    equity0 = args.equity0
    use_breakout = not args.no_breakout
    results: dict[str, dict] = {}
    for sym in args.symbols:
        try:
            prices = _load_prices(sym)
            prices = _resample_prices(prices, args.interval)
        except Exception as e:
            print(f"[warn] failed to load data for {sym}: {e}", file=sys.stderr)
            continue
        # Assemble parameter bundle
        params = RegimeRuleBtParams()
        params.use_breakout_in_high_vol = use_breakout
        # Run regime‑aware backtest
        res = run_regime_rule_symbol(prices, params, equity0=equity0)
        results[sym] = res.get("metrics", {})
    out_path = Path(args.out) if args.out else Path("out/reports") / "regime_rule_backtest_summary.json"
    _ensure_dirs(out_path)
    payload = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload))


def cmd_param_sweep(args: argparse.Namespace) -> None:
    """Run a parameter sweep over rule strategies and write CSV results."""
    out_path = Path(args.out) if args.out else Path("out/reports") / f"param_sweep_{args.strategy}.csv"
    summary = run_param_sweep(
        args.strategy,
        args.symbols,
        out_path,
        equity0=args.equity0,
        n_jobs=args.n_jobs,
    )
    print(summary)


def cmd_process_data(args: argparse.Namespace) -> None:
    """Process raw CSV data into resampled, cleaned files.

    This command reads CSV files for each symbol from ``--input-dir``, resamples
    them to the specified intervals using the same logic as the backtester
    (conversion to Europe/Moscow and OHLCV aggregation) and writes the
    results into ``--output-dir``.  Existing files will be overwritten.
    """
    intervals = args.intervals or ["10min"]
    summary = process_data_all(
        args.symbols,
        intervals=intervals,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    # Write summary to JSON if requested
    if args.out:
        out_path = Path(args.out)
        _ensure_dirs(out_path)
        payload = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "processed": summary,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload))
    else:
        print(summary)


def cmd_forward_test(args: argparse.Namespace) -> None:
    """Run walk-forward tests for rule or regime strategies.

    Parameters are passed through to :func:`run_forward_test`.  Results
    are written to the specified JSON file or printed to stdout.
    """
    result = run_forward_test(
        strategy=args.strategy,
        symbols=args.symbols,
        interval=args.interval,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        equity0=args.equity0,
    )
    if args.out:
        out_path = Path(args.out)
        _ensure_dirs(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result))
    else:
        print(result)


def main(argv: List[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Rule and regime backtester")
    sub = ap.add_subparsers(dest="command")
    # Rule backtest command
    p_rb = sub.add_parser("rule-backtest", help="Run a simple rule‑based backtest")
    p_rb.add_argument("--strategy", choices=["trend", "meanrev", "breakout"], required=True)
    p_rb.add_argument("--symbols", nargs="+", required=True, help="List of symbols to backtest")
    p_rb.add_argument(
        "--interval",
        type=str,
        default="10min",
        help="Bar interval for aggregation (e.g. 10min, 30min, 1h).",
    )
    p_rb.add_argument("--equity0", type=float, default=1_000_000.0, help="Starting equity")
    p_rb.add_argument("--out", type=str, help="Output JSON summary path")
    p_rb.set_defaults(func=cmd_rule_backtest)
    # Regime rule backtest command
    p_reg = sub.add_parser(
        "regime-rule-backtest",
        help="Run a regime‑aware backtest combining rule strategies",
    )
    p_reg.add_argument("--symbols", nargs="+", required=True, help="List of symbols to backtest")
    p_reg.add_argument(
        "--interval",
        type=str,
        default="10min",
        help="Bar interval for aggregation (e.g. 10min, 30min, 1h).",
    )
    p_reg.add_argument("--equity0", type=float, default=1_000_000.0, help="Starting equity")
    p_reg.add_argument(
        "--no-breakout",
        action="store_true",
        help="Do not trade breakout strategy in high volatility regime",
    )
    p_reg.add_argument("--out", type=str, help="Output JSON summary path")
    p_reg.set_defaults(func=cmd_regime_rule_backtest)

    # Parameter sweep command
    p_sw = sub.add_parser(
        "param-sweep", help="Run a parameter sweep for a rule‑based strategy"
    )
    p_sw.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "breakout"],
        required=True,
        help="Which rule strategy to sweep",
    )
    p_sw.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of symbols to include in the sweep",
    )
    p_sw.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Starting equity for each backtest",
    )
    p_sw.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (defaults to out/reports/param_sweep_<strategy>.csv)",
    )
    p_sw.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel processes to use (defaults to CPU count minus one)",
    )
    p_sw.set_defaults(func=cmd_param_sweep)

    # Data processing command
    p_pd = sub.add_parser(
        "process-data", help="Process raw CSVs into resampled, cleaned files"
    )
    p_pd.add_argument(
        "--symbols", nargs="+", required=True, help="List of symbols to process"
    )
    p_pd.add_argument(
        "--intervals",
        nargs="+",
        default=["10min"],
        help="List of bar intervals to generate (e.g. 10min 30min 1h)",
    )
    p_pd.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Directory containing raw CSV files",
    )
    p_pd.add_argument(
        "--output-dir",
        type=str,
        default="processed",
        help="Directory where processed CSVs will be written",
    )
    p_pd.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON summary path for processed files",
    )
    p_pd.set_defaults(func=cmd_process_data)

    # Forward test command
    p_ft = sub.add_parser(
        "forward-test", help="Run walk-forward tests for a rule or regime strategy"
    )
    p_ft.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "breakout", "regime"],
        required=True,
        help="Which strategy to test",
    )
    p_ft.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of symbols to forward test",
    )
    p_ft.add_argument(
        "--interval",
        type=str,
        default="10min",
        help="Bar interval for resampling prices",
    )
    p_ft.add_argument(
        "--train-window",
        type=int,
        default=1000,
        help="Length of the initial training period (bars)",
    )
    p_ft.add_argument(
        "--test-window",
        type=int,
        default=200,
        help="Length of each test window (bars)",
    )
    p_ft.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step between windows (bars); defaults to test-window if omitted",
    )
    p_ft.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Starting equity for each backtest",
    )
    p_ft.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for JSON report",
    )
    p_ft.set_defaults(func=cmd_forward_test)
    # Parse arguments and dispatch
    args = ap.parse_args(argv)
    if not hasattr(args, "func"):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()