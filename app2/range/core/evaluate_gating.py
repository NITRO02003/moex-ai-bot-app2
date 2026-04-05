"""Evaluate the impact of AI entry gating against a baseline.

This script compares a baseline run (without AI entry gating) to a
current run that may include gating. It computes the coverage of
trades (ratio of current trades to baseline trades) and deltas for
key portfolio metrics (profit factor, win rate, maximum drawdown).

The baseline and current runs should both have been generated via
``runner.run_range_backtest`` or the baseline helper script. The
aggregated stats JSON and concatenated trades CSV files are used
as inputs.

Usage:

    python -m app2.range.core.evaluate_gating \
        --baseline-stats out/range_v3_baseline_ALL_30min_baseline_stats.json \
        --current-stats out/range_v3_model_ALL_30min_model_stats.json \
        --baseline-trades out/range_v3_baseline_ALL_30min_baseline_trades.csv \
        --current-trades out/range_v3_model_ALL_30min_model_trades.csv \
        --out out/range_v3_gating_impact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a baseline backtest to a current run with potential AI gating."
    )
    parser.add_argument(
        "--baseline-stats",
        type=str,
        required=True,
        help="Path to the aggregated stats JSON file from the baseline run.",
    )
    parser.add_argument(
        "--current-stats",
        type=str,
        required=True,
        help="Path to the aggregated stats JSON file from the current run.",
    )
    parser.add_argument(
        "--baseline-trades",
        type=str,
        required=True,
        help="Path to the aggregated trades CSV from the baseline run.",
    )
    parser.add_argument(
        "--current-trades",
        type=str,
        required=True,
        help="Path to the aggregated trades CSV from the current run.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the JSON report with coverage and metric deltas.",
    )
    args = parser.parse_args()
    baseline_stats_path = Path(args.baseline_stats)
    current_stats_path = Path(args.current_stats)
    baseline_trades_path = Path(args.baseline_trades)
    current_trades_path = Path(args.current_trades)
    out_path = Path(args.out)

    # Load statistics
    try:
        with baseline_stats_path.open("r", encoding="utf-8") as f:
            baseline_stats = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Baseline stats file not found: {baseline_stats_path}")
    try:
        with current_stats_path.open("r", encoding="utf-8") as f:
            current_stats = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Current stats file not found: {current_stats_path}")
    # Load trades data
    try:
        baseline_trades_df = pd.read_csv(baseline_trades_path)
    except FileNotFoundError:
        raise SystemExit(f"Baseline trades file not found: {baseline_trades_path}")
    try:
        current_trades_df = pd.read_csv(current_trades_path)
    except FileNotFoundError:
        raise SystemExit(f"Current trades file not found: {current_trades_path}")

    # Compute coverage (fraction of baseline trades retained in current run)
    baseline_trade_count = int(len(baseline_trades_df))
    current_trade_count = int(len(current_trades_df))
    coverage = (
        float(current_trade_count) / baseline_trade_count
        if baseline_trade_count > 0
        else 0.0
    )

    # Extract key metrics from stats dicts
    def _get_metric(stats: dict, key: str) -> float:
        return float(stats.get(key, 0.0) or 0.0)

    baseline_pf = _get_metric(baseline_stats, "pf")
    current_pf = _get_metric(current_stats, "pf")
    baseline_win_rate = _get_metric(baseline_stats, "win_rate")
    current_win_rate = _get_metric(current_stats, "win_rate")
    baseline_max_dd = _get_metric(baseline_stats, "max_drawdown")
    current_max_dd = _get_metric(current_stats, "max_drawdown")

    result = {
        "baseline_trades": baseline_trade_count,
        "current_trades": current_trade_count,
        "coverage": coverage,
        "baseline_pf": baseline_pf,
        "current_pf": current_pf,
        "pf_delta": current_pf - baseline_pf,
        "baseline_win_rate": baseline_win_rate,
        "current_win_rate": current_win_rate,
        "win_rate_delta": current_win_rate - baseline_win_rate,
        "baseline_max_drawdown": baseline_max_dd,
        "current_max_drawdown": current_max_dd,
        "max_drawdown_delta": current_max_dd - baseline_max_dd,
    }

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[gating-eval] coverage={coverage:.3f}, pf_delta={result['pf_delta']:.3f}, "
        f"win_rate_delta={result['win_rate_delta']:.3f}, max_dd_delta={result['max_drawdown_delta']:.3f}"
    )
    print(f"[gating-eval] saved report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())