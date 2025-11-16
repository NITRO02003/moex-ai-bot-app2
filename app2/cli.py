
import argparse
import json
import os
import pandas as pd
from pathlib import Path
from . import rule_backtest, regime_rule_backtest, param_sweep, data_pipeline, forward_test
from .regime_detector import detect_regime, regime_distribution

def _ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def run_detect_regime(symbols, interval, config_path, out_path):
    from .paths import DATA_DIR
    with open(config_path, "r") as f:
        config = json.load(f)
    params = config["defaults"].get("RegimeParams", {})
    result = {}
    for sym in symbols:
        file_path = os.path.join(DATA_DIR, f"{sym}_{interval}.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        df = detect_regime(df, params)
        dist = regime_distribution(df["regime"])
        result[sym] = dist
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved regime distribution to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_rb = subparsers.add_parser("rule-backtest")
    p_rb.add_argument("--strategy", choices=["trend", "meanrev", "breakout"], required=True)
    p_rb.add_argument("--symbols", nargs="+", required=True)
    p_rb.add_argument("--interval", type=str, default="10min")
    p_rb.add_argument("--equity0", type=float, default=1_000_000.0)
    p_rb.add_argument("--out", type=str)
    p_rb.set_defaults(func=rule_backtest.main)

    p_reg = subparsers.add_parser("regime-rule-backtest")
    p_reg.add_argument("--symbols", nargs="+", required=True)
    p_reg.add_argument("--interval", type=str, default="10min")
    p_reg.add_argument("--equity0", type=float, default=1_000_000.0)
    p_reg.add_argument("--no-breakout", action="store_true")
    p_reg.add_argument("--out", type=str)
    p_reg.set_defaults(func=regime_rule_backtest.main)

    p_sweep = subparsers.add_parser("param-sweep")
    p_sweep.add_argument("--strategy", choices=["trend", "meanrev", "breakout", "regime"], required=True)
    p_sweep.add_argument("--config", type=str, required=True)
    p_sweep.add_argument("--csv", type=str, required=True)
    p_sweep.add_argument("--symbols", nargs="+", required=True)
    p_sweep.add_argument("--use-breakout-in-high-vol", action="store_true")
    p_sweep.set_defaults(func=param_sweep.main)

    p_proc = subparsers.add_parser("process-data")
    p_proc.add_argument("--symbols", nargs="+", required=True)
    p_proc.add_argument("--intervals", nargs="+", required=True)
    p_proc.add_argument("--input-dir", type=str, required=True)
    p_proc.add_argument("--output-dir", type=str, required=True)
    p_proc.add_argument("--out", type=str)
    p_proc.set_defaults(func=data_pipeline.main)

    p_fw = subparsers.add_parser("forward-test")
    p_fw.add_argument("--strategy", choices=["trend", "meanrev", "breakout", "regime"], required=True)
    p_fw.add_argument("--symbols", nargs="+", required=True)
    p_fw.add_argument("--interval", type=str, default="10min")
    p_fw.add_argument("--train-window", type=int, required=True)
    p_fw.add_argument("--test-window", type=int, required=True)
    p_fw.add_argument("--step", type=int, required=True)
    p_fw.add_argument("--equity0", type=float, default=1_000_000.0)
    p_fw.add_argument("--out", type=str, required=True)
    p_fw.add_argument("--use-breakout-in-high-vol", action="store_true")
    p_fw.set_defaults(func=forward_test.main)

    p_detect = subparsers.add_parser("detect-regime")
    p_detect.add_argument("--symbols", nargs="+", required=True)
    p_detect.add_argument("--interval", type=str, required=True)
    p_detect.add_argument("--config", type=str, required=True)
    p_detect.add_argument("--out", type=str, required=True)
    p_detect.set_defaults(func=lambda args: run_detect_regime(
        args.symbols, args.interval, args.config, args.out
    ))

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
