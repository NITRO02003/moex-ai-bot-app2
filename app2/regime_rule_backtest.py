
import pandas as pd

from .parallel import parallel_map
from dataclasses import dataclass, field
from typing import Optional
from .rule_strategies import generate_trend_signals, generate_meanrev_signals, generate_breakout_signals, TrendParams
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
    trend_params = TrendParams(**params.trend_params)
    trend_params.regime_aware = True
    trend_params.close_on_regime_change = True
    trend_df = generate_trend_signals(df, trend_params)
    df["signal_trend"] = trend_df["signal"]
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

def _run_regime_rule_task(
    args: tuple[str, str, float, RegimeRuleBtParams, bool],
) -> tuple[str, dict | None]:
    sym, interval, equity0, params, use_breakout = args
    import os

    path = os.path.join("processed", f"{sym}_{interval}.csv")
    if not os.path.exists(path):
        return sym, None
    df = pd.read_csv(path, parse_dates=["begin"])
    res = run_regime_rule_symbol(df, params, equity0=equity0, use_breakout_in_high_vol=use_breakout)
    return sym, res.get("metrics", {})


def main(args):
    import json
    from .utils import load_symbols, save_json
    from .config import load_config

    symbols = load_symbols(args.symbols)
    n_jobs = getattr(args, "n_jobs", None)
    config = load_config()
    _ = config  # keep config load for future extensions
    params = RegimeRuleBtParams()

    tasks = [
        (sym, args.interval, float(args.equity0), params, not args.no_breakout)
        for sym in symbols
    ]
    results = {}
    for sym, metrics in parallel_map(tasks, _run_regime_rule_task, n_jobs=n_jobs):
        if metrics is not None:
            results[sym] = metrics

    if args.out:
        save_json(results, args.out)
    else:
        print(json.dumps(results, indent=2))