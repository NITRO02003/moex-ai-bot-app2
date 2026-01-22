from __future__ import annotations

import argparse
import os
import json

import pandas as pd

from .parallel import parallel_map
from .utils import load_symbols
from .config import load_config
from .regime_detector import detect_regime, regime_distribution, build_regime_segments


# ---------- командные обёртки ----------


def cmd_rule_backtest(args):
    """Запуск простого rule-based бэктеста."""
    from . import rule_backtest

    # В модуле уже есть main(args), которая ожидает:
    #   args.strategy, args.symbols, args.interval, args.equity0, args.out
    return rule_backtest.main(args)


def cmd_regime_rule_backtest(args):
    """Запуск режимного бэктеста (trend/range/high_vol)."""
    from . import regime_rule_backtest

    # В модуле есть main(args), которая сама читает config.json
    return regime_rule_backtest.main(args)


def cmd_param_sweep(args):
    """Параметрический свип стратегий."""
    from . import param_sweep

    return param_sweep.run_sweep(
        strategy=args.strategy,
        config_path=args.config,
        csv_path=args.csv,
        symbols=args.symbols,
        equity0=args.equity0,
        use_breakout_in_high_vol=args.use_breakout_in_high_vol,
        n_jobs=args.n_jobs,
    )


def cmd_process_data(args):
    """Агрегация и проверка данных (data -> processed)."""
    from . import data_pipeline

    return data_pipeline.main(args)


def cmd_forward_test(args):
    """Walk-forward тестирование стратегии."""
    from . import forward_test

    # В модуле уже есть main(args), обёртка над run_forward_test(...)
    return forward_test.main(args)


def _detect_regime_task(
    args: tuple[str, str, dict],
) -> tuple[str, dict[str, float] | None, pd.DataFrame | None]:
    sym, interval, regime_cfg = args

    fname_proc = os.path.join("processed", f"{sym}_{interval}.csv")
    fname_raw = os.path.join("data", f"{sym}.csv")

    path = None
    if os.path.exists(fname_proc):
        path = fname_proc
    elif os.path.exists(fname_raw):
        path = fname_raw

    if path is None:
        print(f"[detect-regime] {sym}: no data file found")
        return sym, None, None

    df = pd.read_csv(path)

    dt_col = None
    if "datetime" in df.columns:
        dt_col = "datetime"
    elif "begin" in df.columns:
        dt_col = "begin"

    if dt_col is None:
        print(f"[detect-regime] {sym}: no datetime/begin column in {path}, skip")
        return sym, None, None

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    if "close" not in df.columns:
        print(f"[detect-regime] {sym}: no close column in {path}, skip")
        return sym, None, None

    df_reg = detect_regime(df, regime_cfg)
    dist = regime_distribution(df_reg["regime"])
    seg_df = build_regime_segments(df_reg, df_reg["regime"])
    if not seg_df.empty:
        seg_df.insert(0, "symbol", sym)
    return sym, dist, seg_df if not seg_df.empty else None


def cmd_detect_regime(args):
    """Диагностика долей режимов (trend/range/high_vol) по тикерам.

    Опционально сохраняет CSV с сегментами режимов по каждому тикеру.
    """
    symbols = load_symbols(args.symbols)
    cfg = load_config(args.config)
    regime_cfg = cfg.get("defaults", {}).get("RegimeParams", {})
    n_jobs = getattr(args, "n_jobs", None)

    results: dict[str, dict[str, float]] = {}
    segments_dfs = []

    tasks = [(sym, args.interval, regime_cfg) for sym in symbols]
    for sym, dist, seg_df in parallel_map(tasks, _detect_regime_task, n_jobs=n_jobs):
        if dist is None:
            continue
        results[sym] = dist
        if getattr(args, "segments_out", None) and seg_df is not None:
            segments_dfs.append(seg_df)

    out_obj = {
        "interval": args.interval,
        "config": args.config,
        "results": results,
    }

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[detect-regime] saved regime distribution to {args.out}")
    else:
        print(json.dumps(out_obj, ensure_ascii=False, indent=2))

    if getattr(args, "segments_out", None):
        seg_path = args.segments_out
        if segments_dfs:
            segments_all = pd.concat(segments_dfs, ignore_index=True)
        else:
            segments_all = pd.DataFrame(columns=["symbol", "regime", "start_dt", "end_dt", "bars"])

        seg_dir = os.path.dirname(seg_path)
        if seg_dir:
            os.makedirs(seg_dir, exist_ok=True)
        segments_all.to_csv(seg_path, index=False)
        print(f"[detect-regime] saved regime segments to {seg_path}")

    return out_obj


def cmd_analyze_trades(args):
    """Диагностика сделок и bar-логов с поддержкой профилей (conservative/aggressive)."""
    from . import analysis

    return analysis.run_analyze_trades(
        strategy=args.strategy,
        symbols=args.symbols,
        interval=args.interval,
        equity0=args.equity0,
        config_path=args.config,
        out_prefix=args.out_prefix,
        profile=args.profile,
        n_jobs=getattr(args, "n_jobs", None),
    )
def cmd_range_backtest(args):
    """Range regime backtest (long-only, R&D)."""
    import importlib
    range_backtest = importlib.import_module("app2.range.backtest")
    return range_backtest.main(args)


def cmd_range_analyze(args):
    """EDA по снапшотам range-стратегии (univariate/bivariate/time)."""
    import importlib
    analysis_mod = importlib.import_module("app2.range.analysis")
    return analysis_mod.main(args)


def cmd_range_batch(args):
    """Запуск range-бэктеста по набору тикеров + опциональный EDA и сводный отчет."""
    import importlib
    batch_mod = importlib.import_module("app2.range.batch")
    return batch_mod.main(args)


def cmd_range_summary(args):
    """Сводный отчет по *_stats.json для range-режима."""
    import importlib
    summary_mod = importlib.import_module("app2.range.summary")
    return summary_mod.main(args)
def cmd_range_feature_sweep(args):
    """Большой свип фич по снапшотам range-режима (offline-аналитика)."""
    import importlib
    sweep_mod = importlib.import_module("app2.range.feature_sweep")
    return sweep_mod.main(args)










def cmd_range_v3_backtest(args):
    """Range V3 strategy backtest.

    --engine legacy|core:
      - legacy: use frozen Range V3 implementation (range_v3_legacy + v3_backtest_legacy)
      - core:   future core_v4 engine (not implemented yet)
    """
    import importlib

    engine = getattr(args, "engine", "legacy")
    if engine == "legacy":
        mod_name = "app2.range.v3_backtest_legacy"
    elif engine == "core":
        # core_v4 engine: offline scaffold in app2.range.core.backtest
        mod_name = "app2.range.core.backtest"
    else:
        raise ValueError(f"Unknown range-v3 engine: {engine}")

    v3_mod = importlib.import_module(mod_name)
    return v3_mod.main(args)


def cmd_range_debug_segments(args):
    """Debug offline range segments (CSV + JSON)."""
    from .range import debug_segments

    return debug_segments.main(args)


def cmd_range_v3_make_datasets(args):
    """Generate entry/in-trade datasets from range-v3 artifacts."""
    from .range import make_datasets

    return make_datasets.main(args)


def cmd_leakage_check(args):
    """Run leakage validator for feature/pipe checks."""
    from .tools import leakage_validator

    return leakage_validator.main(args)


def cmd_range_core_sweep(args):
    """Grid sweep for core_v4 parameters."""
    from .range.core import sweep

    return sweep.main(args)

# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(prog="app2.cli")
    subparsers = parser.add_subparsers(dest="command")

    # rule-backtest
    p_rb = subparsers.add_parser(
        "rule-backtest",
        help="Simple rule-based backtest",
    )
    p_rb.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "meanrev_v2", "breakout"],
        required=True,
    )
    p_rb.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_rb.add_argument(
        "--interval",
        type=str,
        default="10min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_rb.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_rb.add_argument(
        "--out",
        type=str,
        help="Path to JSON report (optional)",
    )
    p_rb.add_argument(
        "--regime-segments",
        type=str,
        help="Optional CSV with regime segments to filter signals",
    )
    p_rb.add_argument(
        "--regime-filter",
        type=str,
        help="Regime name to keep (e.g. trend, range, high_vol)",
    )
    p_rb.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_rb.set_defaults(func=cmd_rule_backtest)

    # regime-rule-backtest
    p_reg = subparsers.add_parser(
        "regime-rule-backtest",
        help="Regime-based rule backtest (trend/range/high_vol)",
    )
    p_reg.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_reg.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_reg.add_argument(
        "--no-breakout",
        action="store_true",
        help="Disable breakout in high_vol regime",
    )
    p_reg.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity for regime-rule backtest",
    )
    p_reg.add_argument(
        "--out",
        type=str,
        help="Path to JSON report (optional)",
    )
    p_reg.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_reg.set_defaults(func=cmd_regime_rule_backtest)
    # range-backtest
    p_range = subparsers.add_parser(
        "range-backtest",
        help="Range regime backtest (long-only, R&D)",
    )
    p_range.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_range.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_range.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_range.add_argument(
        "--regime-segments",
        type=str,
        help="Optional CSV with regime segments to filter signals (precomputed)",
    )
    p_range.add_argument(
        "--regime-filter",
        type=str,
        default="range",
        help="Regime name to keep (default: range)",
    )
    p_range.add_argument(
        "--config-range",
        type=str,
        default="app2/range/config.json",
        help="Path to range-specific config JSON",
    )
    p_range.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Output prefix, e.g. out/range/SBER_30min_rangeV2",
    )
    p_range.add_argument(
        "--profile",
        type=str,
        help="Optional profile name from 'profiles' section in range config (if omitted, use defaults)",
    )
    p_range.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_range.set_defaults(func=cmd_range_backtest)
    # range-v3-backtest
    p_rv3 = subparsers.add_parser(
        "range-v3-backtest",
        help="Range V3 strategy backtest (experimental).",
    )
    p_rv3.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_rv3.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_rv3.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_rv3.add_argument(
        "--config-range",
        type=str,
        default="app2/range/config.json",
        help="Path to range-specific config JSON",
    )
    p_rv3.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Output prefix, e.g. out/range_v3/SBER_30min_v3",
    )
    p_rv3.add_argument(
        "--tag",
        type=str,
        default="rangeV3",
        help="Tag to include in out-prefix, e.g. rangeV3",
    )
    p_rv3.add_argument(
        "--engine",
        type=str,
        choices=["legacy", "core"],
        default="legacy",
        help="Range engine: 'legacy' (default) or 'core' (experimental, offline only)",
    )
    p_rv3.add_argument(
        "--entry-model-path",
        type=str,
        default="",
        help="Optional entry model path for AI gating (core only)",
    )
    p_rv3.add_argument(
        "--entry-model-mode",
        type=str,
        choices=["off", "threshold", "top_pct"],
        default="off",
        help="Entry AI gating mode (core only)",
    )
    p_rv3.add_argument(
        "--entry-model-threshold",
        type=float,
        default=0.5,
        help="Entry AI threshold (mode=threshold)",
    )
    p_rv3.add_argument(
        "--entry-model-top-pct",
        type=float,
        default=0.3,
        help="Entry AI top-pct (mode=top_pct)",
    )
    p_rv3.add_argument(
        "--entry-model-trend-path",
        type=str,
        default="",
        help="Optional trend entry model path for AI gating (core only)",
    )
    p_rv3.add_argument(
        "--entry-model-trend-mode",
        type=str,
        choices=["off", "threshold", "top_pct"],
        default="off",
        help="Trend entry AI gating mode (core only)",
    )
    p_rv3.add_argument(
        "--entry-model-trend-threshold",
        type=float,
        default=0.5,
        help="Trend entry AI threshold (mode=threshold)",
    )
    p_rv3.add_argument(
        "--entry-model-trend-top-pct",
        type=float,
        default=0.3,
        help="Trend entry AI top-pct (mode=top_pct)",
    )
    p_rv3.add_argument(
        "--entry-trend-slope-k",
        type=float,
        default=0.0,
        help="Slope threshold for high-confidence trend (0 = slope_k * 0.5)",
    )
    p_rv3.add_argument(
        "--entry-feature-include",
        type=str,
        default="",
        help="Comma-separated entry feature list for AI gating (core only)",
    )
    p_rv3.add_argument(
        "--no-hold-weekend",
        action="store_true",
        help="Close open positions before weekend (core only)",
    )
    p_rv3.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_rv3.set_defaults(func=cmd_range_v3_backtest)

    # range-debug-segments
    p_rdbg = subparsers.add_parser(
        "range-debug-segments",
        help="Debug range segments (CSV snapshots + debug JSON)",
    )
    p_rdbg.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Ticker, e.g. SBER",
    )
    p_rdbg.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_rdbg.add_argument(
        "--date",
        type=str,
        default="",
        help="Optional date filter YYYY-MM-DD",
    )
    p_rdbg.add_argument(
        "--config-range",
        type=str,
        default="app2/range/config.json",
        help="Path to range-specific config JSON",
    )
    p_rdbg.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Output prefix, e.g. out/range_v3/SEGMENTS_DEBUG_SBER_30m_2024-01-15",
    )
    p_rdbg.set_defaults(func=cmd_range_debug_segments)

    # range-v3-make-datasets
    p_rds = subparsers.add_parser(
        "range-v3-make-datasets",
        help="Build entry snapshots dataset from range-v3 backtest artifacts",
    )
    p_rds.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers (must match backtest artifacts)",
    )
    p_rds.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_rds.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Base prefix used by range-v3-backtest (out/range_v3/ALL_30m_BASE)",
    )
    p_rds.add_argument(
        "--config-range",
        type=str,
        default="app2/range/config.json",
        help="Path to range-specific config JSON (for entry candidates)",
    )
    p_rds.add_argument(
        "--tag",
        type=str,
        default="v3seg_base",
        help="Tag used in backtest outputs",
    )
    p_rds.add_argument(
        "--mode",
        type=str,
        choices=["entry", "intrade", "both"],
        default="both",
        help="Which datasets to build: entry, intrade, or both (default: both)",
    )
    p_rds.add_argument(
        "--entry-mode",
        type=str,
        choices=["trades", "candidates", "both"],
        default="trades",
        help="Entry dataset mode: trades (default) or candidates",
    )
    p_rds.add_argument(
        "--entry-label-mode",
        type=str,
        choices=["ret", "ret_mae", "mfe_mae", "quantile"],
        default="ret",
        help="Labeling mode for entry candidates",
    )
    p_rds.add_argument(
        "--entry-horizon-bars",
        type=int,
        default=6,
        help="Forward horizon (bars) for entry candidate labels",
    )
    p_rds.add_argument(
        "--entry-return-threshold",
        type=float,
        default=0.001,
        help="Return threshold for y_entry on candidates",
    )
    p_rds.add_argument(
        "--entry-mfe-threshold",
        type=float,
        default=0.0,
        help="MFE threshold for y_entry (mfe_mae mode)",
    )
    p_rds.add_argument(
        "--entry-mae-threshold",
        type=float,
        default=0.0,
        help="MAE threshold for y_entry (mfe_mae/ret_mae mode)",
    )
    p_rds.add_argument(
        "--entry-quantile",
        type=float,
        default=0.3,
        help="Quantile for y_entry in quantile mode",
    )
    p_rds.add_argument(
        "--entry-quantile-drop-middle",
        action="store_true",
        help="Drop middle quantiles for y_entry (quantile mode)",
    )
    p_rds.add_argument(
        "--exit-improve-threshold",
        type=float,
        default=0.0,
        help=(
            "Labeling threshold for y_exit: "
            "next-bar pnl_rel must exceed final trade pnl_rel by this value"
        ),
    )
    p_rds.add_argument(
        "--exit-min-bars",
        type=int,
        default=0,
        help="Minimum bars_held before y_exit can be set to 1 (default: 0)",
    )
    p_rds.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_rds.set_defaults(func=cmd_range_v3_make_datasets)

    # leakage-check
    p_lc = subparsers.add_parser(
        "leakage-check",
        help="Scan code/data for possible data leakage patterns",
    )
    p_lc.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Files or directories to scan",
    )
    p_lc.add_argument(
        "--extensions",
        type=str,
        default=".py,.csv",
        help="Comma-separated extensions to include (default: .py,.csv)",
    )
    p_lc.add_argument(
        "--allowlist",
        type=str,
        default="",
        help="Comma-separated regex patterns to include (if empty, include all)",
    )
    p_lc.add_argument(
        "--min-severity",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum severity to include in the report",
    )
    p_lc.add_argument(
        "--out",
        type=str,
        help="Optional JSON report path",
    )
    p_lc.set_defaults(func=cmd_leakage_check)

    # range-core-sweep
    p_cs = subparsers.add_parser(
        "range-core-sweep",
        help="Grid sweep for core_v4 parameters",
    )
    p_cs.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_cs.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_cs.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_cs.add_argument(
        "--config-range",
        type=str,
        default="app2/range/config.json",
        help="Path to range-specific config JSON",
    )
    p_cs.add_argument(
        "--grid",
        type=str,
        default="",
        help="Optional JSON file with grid overrides",
    )
    p_cs.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Optional cap on number of combinations",
    )
    p_cs.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path for sweep results",
    )
    p_cs.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_cs.set_defaults(func=cmd_range_core_sweep)


    # range-analyze
    p_ra = subparsers.add_parser(
        "range-analyze",
        help="EDA for range snapshots (univariate/bivariate/time reports)",
    )
    p_ra.add_argument(
        "--snapshots",
        type=str,
        required=True,
        help="Path to *_snapshots.csv",
    )
    p_ra.add_argument(
        "--trades",
        type=str,
        required=True,
        help="Path to *_trades.csv",
    )
    p_ra.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Output prefix under out/, e.g. out/range/SBER_30min_rangeV2_eda",
    )
    p_ra.set_defaults(func=cmd_range_analyze)

    # range-batch
    p_batch = subparsers.add_parser(
        "range-batch",
        help="Run range-backtest for multiple tickers + optional EDA and summary",
    )
    p_batch.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_batch.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_batch.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_batch.add_argument(
        "--regime-segments",
        type=str,
        help="Optional CSV with regime segments to filter signals (precomputed)",
    )
    p_batch.add_argument(
        "--regime-filter",
        type=str,
        default="range",
        help="Regime name to keep (default: range)",
    )
    p_batch.add_argument(
        "--config-range",
        type=str,
        default="app2/range/config.json",
        help="Path to range-specific config JSON",
    )
    p_batch.add_argument(
        "--out-prefix-root",
        type=str,
        default="out/range",
        help="Root directory for per-symbol outputs, e.g. out/range",
    )
    p_batch.add_argument(
        "--tag",
        type=str,
        default="rangeV2",
        help="Tag to include in out-prefix, e.g. rangeV2",
    )
    p_batch.add_argument(
        "--profiles",
        type=str,
        default="",
        help=(
            "Comma-separated list of profile names from range config. "
            "Use 'baseline' for defaults without profile and 'all' for baseline + all profiles."
        ),
    )
    p_batch.add_argument(
        "--no-eda",
        action="store_true",
        help="Disable EDA (range-analyze) after backtests",
    )
    p_batch.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summary CSV generation",
    )
    p_batch.add_argument(
        "--summary-out",
        type=str,
        help="Optional path for summary CSV; if not set, a default will be used under out-prefix-root",
    )
    p_batch.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_batch.set_defaults(func=cmd_range_batch)

    # range-summary
    p_sum = subparsers.add_parser(
        "range-summary",
        help="Build summary CSV from existing *_stats.json files for range",
    )
    p_sum.add_argument(
        "--stats-glob",
        type=str,
        required=True,
        help="Glob for stats JSON files, e.g. 'out/range/*_30min_rangeV2_stats.json'",
    )
    p_sum.add_argument(
        "--out",
        type=str,
        default="out/range/rangeV2_summary.csv",
        help="Path to summary CSV (default: out/range/rangeV2_summary.csv)",
    )
    p_sum.set_defaults(func=cmd_range_summary)

    # range-feature-sweep
    p_rfs = subparsers.add_parser(
        "range-feature-sweep",
        help="Univariate свип фич по снапшотам range-режима",
    )
    p_rfs.add_argument(
        "--snapshots-glob",
        type=str,
        required=True,
        help="Глоб-шаблон для *_snapshots.csv, напр. 'out/range/*_30min_rangeV2_snapshots.csv'",
    )
    p_rfs.add_argument(
        "--features",
        nargs="+",
        default=[
            "atr_14_pct",
            "dist_from_ma",
            "band_pos",
            "bar_range_pct",
            "bar_body_pct",
            "z_ma",
            "band_width_pct",
            "edge_proximity",
            "range_vs_atr",
            "body_vs_range",
        ],
        help="Список фич для свипа (как сырых, так и производных)",
    )
    p_rfs.add_argument(
        "--quantiles",
        type=str,
        default="0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Квантильные уровни через запятую для построения бинов",
    )
    p_rfs.add_argument(
        "--min-trades-bin",
        type=int,
        default=50,
        help="Минимальное число сделок в бине (иначе бин отбрасывается)",
    )
    p_rfs.add_argument(
        "--mode",
        type=str,
        choices=["pooled", "per-symbol"],
        default="pooled",
        help="pooled — все тикеры вместе; per-symbol — отдельный свип по каждому тикеру",
    )
    p_rfs.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Опциональный фильтр по тикерам (если не задано — все из снапшотов)",
    )
    p_rfs.add_argument(
        "--no-time-stats",
        action="store_true",
        help="Не считать отдельные отчёты по hour и day_of_week",
    )
    p_rfs.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Префикс выходных файлов в out/, напр. 'out/range/sweep_rangeV2'",
    )
    p_rfs.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_rfs.set_defaults(func=cmd_range_feature_sweep)




    # param-sweep
    p_sweep = subparsers.add_parser(
        "param-sweep",
        help="Parameter sweep for strategies",
    )
    p_sweep.add_argument(
        "--strategy",
        choices=["meanrev", "meanrev_v2"],
        required=True,
        help="Strategy to sweep ('meanrev' or 'meanrev_v2')",
    )
    p_sweep.add_argument(
        "--config",
        type=str,
        default="app2/config.json",
        help="Path to config.json",
    )
    p_sweep.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Output CSV path for sweep results",
    )
    p_sweep.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_sweep.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_sweep.add_argument(
        "--use-breakout-in-high-vol",
        action="store_true",
        help="Use breakout strategy in high_vol regime (if applicable)",
    )
    p_sweep.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel processes (-1 = all cores)",
    )
    p_sweep.set_defaults(func=cmd_param_sweep)

    # process-data
    p_pd = subparsers.add_parser(
        "process-data",
        help="Aggregate and validate raw data into processed/",
    )
    p_pd.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_pd.add_argument(
        "--intervals",
        nargs="+",
        required=True,
        help="List of intervals, e.g. 10min 30min 1h",
    )
    p_pd.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Input directory with raw CSV (default: data)",
    )
    p_pd.add_argument(
        "--output-dir",
        type=str,
        default="processed",
        help="Output directory for aggregated data (default: processed)",
    )
    p_pd.add_argument(
        "--out",
        type=str,
        help="Optional JSON summary path",
    )
    p_pd.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_pd.set_defaults(func=cmd_process_data)

    # forward-test
    p_ft = subparsers.add_parser(
        "forward-test",
        help="Walk-forward testing of a strategy",
    )
    p_ft.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "meanrev_v2", "breakout"],
        required=True,
    )
    p_ft.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_ft.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_ft.add_argument(
        "--train-window",
        type=int,
        required=True,
        help="Training window (bars)",
    )
    p_ft.add_argument(
        "--test-window",
        type=int,
        required=True,
        help="Test window (bars)",
    )
    p_ft.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step between windows (bars)",
    )
    p_ft.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_ft.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path for forward-test results",
    )
    p_ft.add_argument(
        "--use-breakout-in-high-vol",
        action="store_true",
        help="Use breakout strategy in high_vol regime (if applicable)",
    )
    p_ft.set_defaults(func=cmd_forward_test)

    # detect-regime
    p_dr = subparsers.add_parser(
        "detect-regime",
        help="Compute regime distribution (trend/range/high_vol) for symbols",
    )
    p_dr.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of tickers or 'all'",
    )
    p_dr.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Timeframe, e.g. 10min, 30min, 1h",
    )
    p_dr.add_argument(
        "--config",
        type=str,
        default="app2/config.json",
        help="Path to config.json",
    )
    p_dr.add_argument(
        "--out",
        type=str,
        help="Output JSON path for regime distribution",
    )
    p_dr.add_argument(
        "--segments-out",
        type=str,
        help="Optional CSV path to save regime segments",
    )
    p_dr.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_dr.set_defaults(func=cmd_detect_regime)

    # analyze-trades
    p_an = subparsers.add_parser(
        "analyze-trades",
        help="Generate bar- and trade-level logs for diagnostics",
    )
    p_an.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "meanrev_v2", "breakout"],
        required=True,
        help="Имя стратегии",
    )
    p_an.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Список тикеров или 'all'",
    )
    p_an.add_argument(
        "--interval",
        type=str,
        default="30min",
        help="Таймфрейм (например, 10min, 30min, 1h)",
    )
    p_an.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Начальный капитал",
    )
    p_an.add_argument(
        "--config",
        type=str,
        default="app2/config.json",
        help="Путь к config.json",
    )
    p_an.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Префикс для выходных файлов, например out/diag_meanrev",
    )
    p_an.add_argument(
        "--profile",
        type=str,
        choices=["conservative", "aggressive"],
        help="Имя профиля из секции 'profiles' в config.json",
    )
    p_an.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of processes (0/<=0 = auto)",
    )
    p_an.set_defaults(func=cmd_analyze_trades)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()