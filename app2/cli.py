
import argparse
import os
import json
import pandas as pd

# ---- command implementations ----

def cmd_rule_backtest(args):
    """Wrapper for simple rule-based backtest."""
    from . import rule_backtest
    # rule_backtest.main is expected to handle args.strategy, args.symbols, etc.
    return rule_backtest.main(args)

def cmd_regime_rule_backtest(args):
    """Wrapper for regime-based rule backtest."""
    from . import regime_rule_backtest
    return regime_rule_backtest.main(args)

def cmd_param_sweep(args):
    """Wrapper for parameter sweep."""
    from . import param_sweep
    # We call run_sweep explicitly, since not all versions have main()
    return param_sweep.run_sweep(
        strategy=args.strategy,
        config_path=args.config,
        csv_path=args.csv,
        symbols=args.symbols,
        equity0=args.equity0,
        use_breakout_in_high_vol=args.use_breakout_in_high_vol,
    )

def cmd_process_data(args):
    """Wrapper for data aggregation / processing."""
    from . import data_pipeline
    # Expect data_pipeline.main(args) or run_data_processing; try main first.
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
    """Wrapper for rolling forward-test."""
    from . import forward_test
    # forward_test.main should exist; if not, fall back to run_forward_test
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
        )

def cmd_detect_regime(args):
    """Compute regime distribution per symbol and save JSON."""
    from .regime_detector import detect_regime, regime_distribution
    # We read aggregated data from processed/{SYM}_{interval}.csv
    results = {}
    for sym in args.symbols:
        path = os.path.join("processed", f"{sym}_{args.interval}.csv")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["begin"])
        if "begin" in df.columns:
            df = df.rename(columns={"begin": "datetime"})
        df = detect_regime(df, args.regime_params)
        dist = regime_distribution(df["regime"])
        results[sym] = dist

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved regime distribution to {args.out}")

# ---- main CLI ----

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
    p_reg = subparsers.add_parser("regime-rule-backtest", help="Backtest with regime-aware switching")
    p_reg.add_argument("--symbols", nargs="+", required=True)
    p_reg.add_argument("--interval", type=str, default="10min")
    p_reg.add_argument("--equity0", type=float, default=1_000_000.0)
    p_reg.add_argument("--no-breakout", action="store_true", help="Disable breakout in high_vol regime")
    p_reg.add_argument("--out", type=str, help="Path to JSON report")
    p_reg.set_defaults(func=cmd_regime_rule_backtest)

    # param-sweep
    p_sweep = subparsers.add_parser("param-sweep", help="Parameter sweep for strategies")
    p_sweep.add_argument("--strategy", choices=["trend", "meanrev", "breakout", "regime"], required=True)
    p_sweep.add_argument("--config", type=str, required=True)
    p_sweep.add_argument("--csv", type=str, required=True)
    p_sweep.add_argument("--symbols", nargs="+", required=True)
    p_sweep.add_argument("--equity0", type=float, default=1_000_000.0)
    p_sweep.add_argument("--use-breakout-in-high-vol", action="store_true")
    p_sweep.set_defaults(func=cmd_param_sweep)

    # process-data
    p_proc = subparsers.add_parser("process-data", help="Aggregate raw data into timeframes")
    p_proc.add_argument("--symbols", nargs="+", required=True)
    p_proc.add_argument("--intervals", nargs="+", required=True)
    p_proc.add_argument("--input-dir", type=str, required=True)
    p_proc.add_argument("--output-dir", type=str, required=True)
    p_proc.add_argument("--out", type=str, help="Optional JSON summary")
    p_proc.set_defaults(func=cmd_process_data)

    # forward-test
    p_fw = subparsers.add_parser("forward-test", help="Rolling forward test")
    p_fw.add_argument("--strategy", choices=["trend", "meanrev", "breakout", "regime"], required=True)
    p_fw.add_argument("--symbols", nargs="+", required=True)
    p_fw.add_argument("--interval", type=str, default="10min")
    p_fw.add_argument("--train-window", type=int, required=True)
    p_fw.add_argument("--test-window", type=int, required=True)
    p_fw.add_argument("--step", type=int, required=True)
    p_fw.add_argument("--equity0", type=float, default=1_000_000.0)
    p_fw.add_argument("--out", type=str, required=True)
    p_fw.add_argument("--use-breakout-in-high-vol", action="store_true")
    p_fw.set_defaults(func=cmd_forward_test)

    # detect-regime
    p_det = subparsers.add_parser("detect-regime", help="Compute regime distribution")
    p_det.add_argument("--symbols", nargs="+", required=True)
    p_det.add_argument("--interval", type=str, required=True)
    p_det.add_argument("--config", type=str, required=True)
    p_det.add_argument("--out", type=str, required=True)

    # режимные параметры берём из конфигурационного файла внутри обработчика
    def _det_wrapper(args):
        # загрузим RegimeParams из config
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        args.regime_params = cfg.get("defaults", {}).get("RegimeParams", {})
        return cmd_detect_regime(args)

    p_det.set_defaults(func=_det_wrapper)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)

if __name__ == "__main__":
    main()
