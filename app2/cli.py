from __future__ import annotations

import argparse
import os
import json

import pandas as pd

from .utils import load_symbols
from .config import load_config
from .regime_detector import detect_regime, regime_distribution


# ---------- командные обёртки ----------


def cmd_rule_backtest(args):
    """Wrapper для простого rule-based бэктеста."""
    from . import rule_backtest
    return rule_backtest.main(args)


def cmd_regime_rule_backtest(args):
    """Wrapper для regime-aware бэктеста."""
    from . import regime_rule_backtest
    return regime_rule_backtest.main(args)


def cmd_param_sweep(args):
    """Wrapper для свипа параметров."""
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
    """Wrapper для агрегации сырых данных."""
    from . import data_pipeline

    if hasattr(data_pipeline, "main"):
        return data_pipeline.main(args)
    else:
        return data_pipeline.run_data_processing(
            symbols=args.symbols,
            intervals=args.intervals,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            out=args.out,
        )


def cmd_forward_test(args):
    """Wrapper для rolling forward-test."""
    from . import forward_test

    if hasattr(forward_test, "main"):
        return forward_test.main(args)
    else:
        return forward_test.run_forward_test(
            strategy=args.strategy,
            symbols=args.symbols,
            interval=args.interval,
            train_window=args.train_window,
            test_window=args.test_window,
            step=args.step,
            equity0=args.equity0,
            out_path=args.out,
            use_breakout_in_high_vol=args.use_breakout_in_high_vol,
            n_jobs=args.n_jobs,
        )


def cmd_detect_regime(args):
    """Подсчёт распределения режимов по тикерам и сохранение в JSON."""
    cfg = load_config(args.config)
    regime_params = cfg.get("defaults", {}).get("RegimeParams", {})

    results = {}
    symbols = load_symbols(args.symbols)

    for sym in symbols:
        path = os.path.join("processed", f"{sym}_{args.interval}.csv")
        if not os.path.exists(path):
            print(f"[detect-regime] file not found: {path}")
            continue

        df = pd.read_csv(path)
        if "begin" in df.columns and "datetime" not in df.columns:
            df["datetime"] = pd.to_datetime(df["begin"])
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        df = detect_regime(df, regime_params)
        dist = regime_distribution(df["regime"])
        results[sym] = dist
        print(
            f"[detect-regime] {sym}: "
            f"trend={dist.get('trend', 0):.2f}%, "
            f"range={dist.get('range', 0):.2f}%, "
            f"high_vol={dist.get('high_vol', 0):.2f}%"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[detect-regime] saved distribution to {args.out}")


def cmd_analyze_trades(args):
    """Wrapper для диагностики сделок и bar-логов."""
    from . import analysis

    return analysis.run_analyze_trades(
        strategy=args.strategy,
        symbols=args.symbols,
        interval=args.interval,
        equity0=args.equity0,
        config_path=args.config,
        out_prefix=args.out_prefix,
    )


# ---------- основной CLI ----------


def main():
    parser = argparse.ArgumentParser(prog="app2.cli")
    subparsers = parser.add_subparsers(dest="command")

    # rule-backtest
    p_rb = subparsers.add_parser("rule-backtest", help="Simple rule-based backtest")
    p_rb.add_argument("--strategy", choices=["trend", "meanrev", "breakout"], required=True)
    p_rb.add_argument("--symbols", nargs="+", required=True)
    p_rb.add_argument("--interval", type=str, default="10min")
    p_rb.add_argument("--equity0", type=float, default=1_000_000.0)
    p_rb.add_argument("--out", type=str, help="Path to JSON report")
    p_rb.set_defaults(func=cmd_rule_backtest)

    # regime-rule-backtest
    p_reg = subparsers.add_parser(
        "regime-rule-backtest",
        help="Backtest with regime-aware switching",
    )
    p_reg.add_argument("--symbols", nargs="+", required=True)
    p_reg.add_argument("--interval", type=str, default="10min")
    p_reg.add_argument("--equity0", type=float, default=1_000_000.0)
    p_reg.add_argument(
        "--no-breakout",
        action="store_true",
        help="Disable breakout in high_vol regime",
    )
    p_reg.add_argument("--out", type=str, help="Path to JSON report")
    p_reg.set_defaults(func=cmd_regime_rule_backtest)

    # param-sweep
    p_sweep = subparsers.add_parser(
        "param-sweep",
        help="Parameter sweep for strategies",
    )
    p_sweep.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "breakout", "regime"],
        required=True,
    )
    p_sweep.add_argument("--config", type=str, required=True)
    p_sweep.add_argument("--csv", type=str, required=True)
    p_sweep.add_argument("--symbols", nargs="+", required=True)
    p_sweep.add_argument("--equity0", type=float, default=1_000_000.0)
    p_sweep.add_argument(
        "--use-breakout-in-high-vol",
        action="store_true",
        help="Use breakout strategy in high_vol regime (if applicable)",
    )
    p_sweep.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of processes for sweep (-1 = all cores, 1 = no multiprocessing)",
    )
    p_sweep.set_defaults(func=cmd_param_sweep)

    # process-data
    p_proc = subparsers.add_parser(
        "process-data",
        help="Aggregate raw data into timeframes",
    )
    p_proc.add_argument("--symbols", nargs="+", required=True)
    p_proc.add_argument("--intervals", nargs="+", required=True)
    p_proc.add_argument("--input-dir", type=str, required=True)
    p_proc.add_argument("--output-dir", type=str, required=True)
    p_proc.add_argument("--out", type=str, help="Optional JSON summary")
    p_proc.set_defaults(func=cmd_process_data)

    # forward-test
    p_fw = subparsers.add_parser(
        "forward-test",
        help="Rolling forward test",
    )
    p_fw.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "breakout", "regime"],
        required=True,
    )
    p_fw.add_argument("--symbols", nargs="+", required=True)
    p_fw.add_argument("--interval", type=str, default="10min")
    p_fw.add_argument("--train-window", type=int, required=True)
    p_fw.add_argument("--test-window", type=int, required=True)
    p_fw.add_argument("--step", type=int, required=True)
    p_fw.add_argument("--equity0", type=float, default=1_000_000.0)
    p_fw.add_argument("--out", type=str, required=True)
    p_fw.add_argument(
        "--use-breakout-in-high-vol",
        action="store_true",
        help="Use breakout in high_vol regime (if applicable)",
    )
    p_fw.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of processes (-1 = all cores, 1 = single process)",
    )
    p_fw.set_defaults(func=cmd_forward_test)

    # detect-regime
    p_det = subparsers.add_parser(
        "detect-regime",
        help="Compute regime distribution for symbols",
    )
    p_det.add_argument("--symbols", nargs="+", required=True)
    p_det.add_argument("--interval", type=str, required=True)
    p_det.add_argument("--config", type=str, required=True)
    p_det.add_argument("--out", type=str, required=True)
    p_det.set_defaults(func=cmd_detect_regime)

    # analyze-trades
    p_an = subparsers.add_parser(
        "analyze-trades",
        help="Generate bar- and trade-level logs for diagnostics",
    )
    p_an.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "breakout"],
        required=True,
    )
    p_an.add_argument("--symbols", nargs="+", required=True)
    p_an.add_argument("--interval", type=str, default="30min")
    p_an.add_argument(
        "--equity0",
        type=float,
        default=1_000_000.0,
        help="Initial equity",
    )
    p_an.add_argument(
        "--config",
        type=str,
        default="app2/config.json",
        help="Path to config.json",
    )
    p_an.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for output files, e.g. out/diag_meanrev",
    )
    p_an.set_defaults(func=cmd_analyze_trades)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
