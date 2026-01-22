from __future__ import annotations

import glob
import json
import os
from typing import Iterable, List, Dict, Any

import pandas as pd


def _extract_rows_from_data(data: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    """Normalize different stats.json layouts into a list of flat rows.

    Поддерживаем два формата:
    1) { "results": { "SBER": {..}, "GAZP": {..}, ... } }
    2) { "SBER": {..}, "GAZP": {..}, ... }
    """
    rows: List[Dict[str, Any]] = []

    if not isinstance(data, dict):
        return rows

    if isinstance(data.get("results"), dict):
        mapping = data["results"]
    else:
        mapping = data

    for symbol, stats in mapping.items():
        if not isinstance(stats, dict):
            continue
        row: Dict[str, Any] = {"symbol": symbol, "source": source}
        for key, value in stats.items():
            row[key] = value
        rows.append(row)

    return rows


def _load_stats_from_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _extract_rows_from_data(data, path)


def build_summary_from_paths(stats_paths: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    stats_paths = list(stats_paths)
    for path in stats_paths:
        try:
            rows.extend(_load_stats_from_file(path))
        except Exception as exc:
            print(f"[range-summary] failed to load {path}: {exc!r}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ставим самые полезные колонки вперед
    preferred = [
        "symbol",
        "source",
        "total_return",
        "pf",
        "win_rate",
        "trade_count",
        "max_drawdown",
        "volatility_ann",
        "sharpe_ann",
        "pnl_sum",
        "pnl_mean",
        "pnl_std",
        "bars_total",
        "bars_in_regime",
        "entries_raw",
        "entries_after_regime_filter",
        "regime_filter",
    ]
    cols_first = [c for c in preferred if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + other_cols]

    return df


def main(args):
    stats_glob: str = args.stats_glob
    out_path: str = args.out

    stats_paths = sorted(glob.glob(stats_glob))
    print(f"[range-summary] matched {len(stats_paths)} files for pattern {stats_glob}")

    df = build_summary_from_paths(stats_paths)

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[range-summary] written {len(df)} rows to {out_path}")
