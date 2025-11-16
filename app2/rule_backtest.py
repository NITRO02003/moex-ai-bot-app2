
import pandas as pd
from .rule_strategies import generate_trend_signals, generate_meanrev_signals, generate_breakout_signals
from .rule_core import run_rule_symbol, RuleBtParams
from .utils import load_symbols, save_json
from .config import load_config
import os
import json

def main(args):
    symbols = load_symbols(args.symbols)
    strategy = args.strategy
    interval = args.interval
    config = load_config()
    out_path = args.out

    results = {}
    for sym in symbols:
        path = os.path.join("processed", f"{sym}_{interval}.csv")
        df = pd.read_csv(path, parse_dates=["datetime"])

        if strategy == "trend":
            df["signal"] = generate_trend_signals(df, **config["defaults"]["TrendParams"])
        elif strategy == "meanrev":
            df["signal"] = generate_meanrev_signals(df, **config["defaults"]["MeanRevParams"])
        elif strategy == "breakout":
            df["signal"] = generate_breakout_signals(df, **config["defaults"]["BreakoutParams"])
        else:
            raise ValueError("Invalid strategy")

        params = RuleBtParams(**config["defaults"]["RuleBtParams"])
        result = run_rule_symbol(df, params, args.equity0)
        results[sym] = result["metrics"]

    if out_path:
        save_json(results, out_path)
    else:
        print(json.dumps(results, indent=2))
