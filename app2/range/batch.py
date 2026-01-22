from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List

from ..parallel import parallel_map
DEFAULT_RANGE_SYMBOLS: List[str] = [
    "GAZP",
    "ROSN",
    "SBER",
    "LKOH",
    "GMKN",
    "YNDX",
    "NVTK",
    "NLMK",
    "MTSS",
    "TATN",
    "CHMF",
    "SNGS",
    "PIKK",
    "PLZL",
    "MGNT",
    "VKCO",
    "OZON",
]


def _normalize_symbols(symbols: List[str]) -> List[str]:
    if len(symbols) == 1 and symbols[0].lower() == "all":
        return DEFAULT_RANGE_SYMBOLS
    return symbols


def _join_prefix(root: str, symbol: str, interval: str, tag: str) -> str:
    root_clean = root.rstrip("/\\")
    return f"{root_clean}/{symbol}_{interval}_{tag}"


def _run_batch_symbol_task(
    args: tuple[str, str, float, str | None, str | None, str, str, str, bool],
) -> tuple[str | None, str]:
    import importlib

    (
        sym,
        interval,
        equity0,
        regime_segments,
        regime_filter,
        config_range,
        out_root,
        tag,
        do_eda,
    ) = args

    range_backtest = importlib.import_module("app2.range.backtest")
    analysis_mod = importlib.import_module("app2.range.analysis") if do_eda else None

    out_prefix = _join_prefix(out_root, sym, interval, tag)
    bt_args = SimpleNamespace(
        symbols=[sym],
        interval=interval,
        equity0=equity0,
        regime_segments=regime_segments,
        regime_filter=regime_filter,
        config_range=config_range,
        out_prefix=out_prefix,
        n_jobs=1,
    )
    print(f"[range-batch] running range-backtest for {sym}, out_prefix={out_prefix}")
    range_backtest.main(bt_args)

    stats_path = f"{out_prefix}_stats.json"

    if do_eda:
        snapshots_path = f"{out_prefix}_snapshots.csv"
        trades_path = f"{out_prefix}_trades.csv"
        if os.path.exists(snapshots_path) and os.path.exists(trades_path):
            eda_out_prefix = f"{out_prefix}_eda"
            eda_args = SimpleNamespace(
                snapshots=snapshots_path,
                trades=trades_path,
                out_prefix=eda_out_prefix,
            )
            print(f"[range-batch] running range-analyze for {sym}")
            analysis_mod.main(eda_args)
        else:
            print(
                f"[range-batch] skip EDA for {sym}: "
                f"missing {snapshots_path} or {trades_path}"
            )

    return stats_path, sym


def main(args):
    import importlib

    symbols = _normalize_symbols(list(args.symbols))
    interval = args.interval
    equity0 = args.equity0
    regime_segments = args.regime_segments
    regime_filter = args.regime_filter
    config_range = args.config_range
    out_root = args.out_prefix_root
    tag = args.tag
    do_eda = not getattr(args, "no_eda", False)
    do_summary = not getattr(args, "no_summary", False)
    summary_out = getattr(args, "summary_out", None)
    n_jobs = getattr(args, "n_jobs", None)

    stats_paths: List[str] = []
    tasks = [
        (
            sym,
            interval,
            equity0,
            regime_segments,
            regime_filter,
            config_range,
            out_root,
            tag,
            do_eda,
        )
        for sym in symbols
    ]
    for stats_path, _sym in parallel_map(tasks, _run_batch_symbol_task, n_jobs=n_jobs):
        if stats_path:
            stats_paths.append(stats_path)

    if do_summary and stats_paths:
        summary_mod = importlib.import_module('app2.range.summary')
        if not summary_out:
            # default summary path under out_root
            summary_out = _join_prefix(out_root, 'ALL', interval, tag) + '_summary.csv'
        print(f"[range-batch] building summary -> {summary_out}")
        df_summary = summary_mod.build_summary_from_paths(stats_paths)
        out_dir = os.path.dirname(summary_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_summary.to_csv(summary_out, index=False)
        print(f"[range-batch] written {len(df_summary)} rows to {summary_out}")