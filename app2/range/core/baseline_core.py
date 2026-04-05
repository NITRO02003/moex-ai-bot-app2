"""Baseline generation script for the range‑core backtester.

This module provides a simple command‑line interface to run the
core range backtester across a predefined set of symbols and
timeframes and persist the resulting portfolio statistics and
trades.  It is intended to produce an immutable baseline
reference against which future strategy changes can be compared.

Usage:
    python -m app2.range.core.baseline_core --symbols all --interval 30min \
        --equity0 1000000 --config-range app2/range/config.json \
        --out-prefix-root out/range_v3_baseline --tag baseline

The script delegates all of the heavy lifting to ``runner.run_range_backtest``.
By default no AI entry gating is applied; to test the core strategy
alone simply omit any entry‑model parameters.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import List

from .runner import run_range_backtest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a baseline run for the range‑core backtester."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["all"],
        help=(
            "List of tickers to run or 'all' to run on all available symbols. "
            "Defaults to 'all'."
        ),
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default="baseline_symbols.txt",
        help=(
            "Optional path to a file containing a whitespace- or comma-separated list of symbols."
            " If specified and --symbols is not provided or set to 'all', the baseline will run on"
            " symbols from this file. Defaults to 'baseline_symbols.txt' in the project root."
        ),
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe to run backtests on (e.g. 10min, 30min, 1h). Defaults to 30min.",
    )
    parser.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity baseline. Defaults to 1,000,000.",
    )
    parser.add_argument(
        "--config-range",
        type=str,
        default="app2/range/config.json",
        help="Path to the range strategy configuration JSON."
        " Defaults to 'app2/range/config.json'.",
    )
    parser.add_argument(
        "--out-prefix-root",
        type=str,
        default="out/range_v3_baseline",
        help=(
            "Root prefix for output files. The script will append symbol and "
            "interval information as needed. Defaults to 'out/range_v3_baseline'."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Tag to append to output files. Defaults to 'baseline'.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. 0 (or <=0) means use all cores."
            " Defaults to 0."
        ),
    )
    args = parser.parse_args()
    # Prepare namespace for runner.run_range_backtest.  We reuse the same
    # attributes as the CLI backtest uses; unused attributes (e.g. entry_model)
    # will be supplied with defaults inside run_range_backtest.
    # Determine symbols list. If symbols is default ['all'] and a symbols file is provided and exists,
    # load tickers from the file. Otherwise use args.symbols as is.
    symbols_list: List[str] = args.symbols
    try:
        # only override if user did not explicitly provide custom list
        if len(args.symbols) == 1 and args.symbols[0].lower() == "all":
            from pathlib import Path

            sym_file = Path(args.symbols_file)
            if sym_file.exists():
                text = sym_file.read_text(encoding="utf-8")
                # split on whitespace or commas
                raw = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
                if raw:
                    symbols_list = raw
    except Exception:
        # If any error occurs, fall back to provided symbols
        symbols_list = args.symbols

    run_args = SimpleNamespace(
        symbols=symbols_list,
        interval=args.interval,
        equity0=args.equity0,
        config_range=args.config_range,
        out_prefix=args.out_prefix_root,
        tag=args.tag,
        n_jobs=args.n_jobs,
    )
    run_range_backtest(run_args)
    print(
        f"[baseline-core] baseline run completed for {args.symbols} at {args.interval}. "
        f"Outputs saved under prefix {args.out_prefix_root}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())