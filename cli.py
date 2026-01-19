from __future__ import annotations

import argparse
import os
import json

import pandas as pd

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




def cmd_range_backtest(args):
    """Range regime backtest (long-only, R&D)."""
    from .range import backtest as range_backtest

    return range_backtest.main(args)

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


def cmd_detect_regime(args):
    """Диагностика долей режимов (trend/range/high_vol) по тикерам.

    Опционально сохраняет CSV с сегментами режимов по каждому тикеру.
    """
    symbols = load_symbols(args.symbols)
    cfg = load_config(args.config)
    regime_cfg = cfg.get("defaults", {}).get("RegimeParams", {})

    results: dict[str, dict[str, float]] = {}
    segments_dfs = []

    for sym in symbols:
        # пытаемся сначала взять агрегированные данные, потом сырые
        fname_proc = os.path.join("processed", f"{sym}_{args.interval}.csv")
        fname_raw = os.path.join("data", f"{sym}.csv")

        path = None
        if os.path.exists(fname_proc):
            path = fname_proc
        elif os.path.exists(fname_raw):
            path = fname_raw

        if path is None:
            print(f"[detect-regime] {sym}: no data file found")
            continue

        df = pd.read_csv(path)

        dt_col = None
        if "datetime" in df.columns:
            dt_col = "datetime"
        elif "begin" in df.columns:
            dt_col = "begin"

        if dt_col is None:
            print(f"[detect-regime] {sym}: no datetime/begin column in {path}, skip")
            continue

        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(dt_col).reset_index(drop=True)

        # нужны хотя бы цены закрытия
        if "close" not in df.columns:
            print(f"[detect-regime] {sym}: no close column in {path}, skip")
            continue

        df_reg = detect_regime(df, regime_cfg)
        dist = regime_distribution(df_reg["regime"])

        results[sym] = dist

        if getattr(args, "segments_out", None):
            seg_df = build_regime_segments(df_reg, df_reg["regime"])
            if not seg_df.empty:
                seg_df.insert(0, "symbol", sym)
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
    )


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
    p_reg.set_defaults(func=cmd_regime_rule_backtest)

    # range-backtest
    p_range = subparsers.add_parser(
        "range-backtest",
        help="Range regime backtest (long-only, R&D)",
    )
    p_range.add_argument(
        "--symbols",
        nargs='+',
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
        help="Output prefix, e.g. out/range/SBER_30min_rangeV0",
    )
    p_range.set_defaults(func=cmd_range_backtest)

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
    p_an.set_defaults(func=cmd_analyze_trades)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
