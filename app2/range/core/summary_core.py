"""Summary builder for core range backtests.

This module collects aggregated portfolio statistics from multiple core backtest runs
and produces a unified DataFrame for analysis. It is analogous to ``range.summary``
for the legacy range engine but operates on the new core statistics format (a flat
dictionary of metrics such as total_pnl, total_return, pf, win_rate, etc.).

Usage:

    python -m app2.range.core.summary_core \
        --stats-glob "out/range_v3_*_stats.json" \
        --out out/core_summary.csv

The script will write a CSV file with one row per stats JSON and columns for each
metric found.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Iterable, List, Dict, Any

import pandas as pd


def build_summary_from_paths(stats_paths: Iterable[str]) -> pd.DataFrame:
    """Build a DataFrame from a list of core stats JSON files.

    Each stats file is expected to contain a flat mapping from metric names to values,
    e.g. {"symbols": [...], "total_pnl": ..., "pf": ..., ...}.  The filename
    will be stored under the column ``source``.
    """
    rows: List[Dict[str, Any]] = []
    for path in stats_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            row: Dict[str, Any] = {"source": path}
            for key, value in data.items():
                row[key] = value
            rows.append(row)
        except Exception as exc:
            print(f"[core-summary] failed to load {path}: {exc!r}")
    return pd.DataFrame(rows)


def main(args) -> int:
    stats_glob: str = args.stats_glob
    out_path: str = args.out
    stats_paths = sorted(glob.glob(stats_glob))
    print(f"[core-summary] matched {len(stats_paths)} files for pattern {stats_glob}")
    df = build_summary_from_paths(stats_paths)
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[core-summary] written {len(df)} rows to {out_path}")
    else:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build summary CSV from core stats JSON files.")
    parser.add_argument(
        "--stats-glob",
        type=str,
        required=True,
        help="Glob pattern to match core stats JSON files, e.g. 'out/range_v3_*_stats.json'",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path (if empty, print summary to stdout)",
    )
    args = parser.parse_args()
    raise SystemExit(main(args))