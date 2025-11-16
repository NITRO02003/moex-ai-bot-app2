
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from .rule_strategies import generate_trend_signals, generate_meanrev_signals, generate_breakout_signals
from .rule_core import run_rule_symbol, RuleBtParams
from .regime_detector import detect_regime

@dataclass
class RegimeRuleBtParams(RuleBtParams):
    trend_params: dict = field(default_factory=dict)
    meanrev_params: dict = field(default_factory=dict)
    breakout_params: dict = field(default_factory=dict)
    regime_params: dict = field(default_factory=lambda: {
        "high_vol_quantile": 0.98,
        "trend_threshold": 2.5,
        "atr_len": 14,
        "ema_fast": 12,
        "ema_slow": 48
    })

def run_regime_rule_symbol(df: pd.DataFrame, params: RegimeRuleBtParams, equity0=1_000_000, use_breakout_in_high_vol=False):
    df = detect_regime(df, params.regime_params)
    df["signal_trend"] = generate_trend_signals(df, **params.trend_params)
    df["signal_meanrev"] = generate_meanrev_signals(df, **params.meanrev_params)
    df["signal_breakout"] = generate_breakout_signals(df, **params.breakout_params)

    signal = []
    for i in range(len(df)):
        regime = df.iloc[i]["regime"]
        if regime == "trend":
            signal.append(df.iloc[i]["signal_trend"])
        elif regime == "range":
            signal.append(df.iloc[i]["signal_meanrev"])
        elif regime == "high_vol" and use_breakout_in_high_vol:
            signal.append(df.iloc[i]["signal_breakout"])
        else:
            signal.append(0)

    df["signal"] = signal
    return run_rule_symbol(df, params, equity0)

def main(args):
    import os
    import json
    from .utils import load_symbols, save_json
    from .config import load_config

    symbols = load_symbols(args.symbols)
    results = {}
    config = load_config()
    params = RegimeRuleBtParams()

    for sym in symbols:
        path = os.path.join("processed", f"{sym}_{args.interval}.csv")
        df = pd.read_csv(path, parse_dates=["datetime"])
        res = run_regime_rule_symbol(df, params, equity0=args.equity0, use_breakout_in_high_vol=not args.no_breakout)
        results[sym] = res["metrics"]

    if args.out:
        save_json(results, args.out)
    else:
        print(json.dumps(results, indent=2))
