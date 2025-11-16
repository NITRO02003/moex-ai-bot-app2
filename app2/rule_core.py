
from dataclasses import dataclass
import pandas as pd

@dataclass
class RuleBtParams:
    commission: float = 0.0005
    slippage_bps: float = 1.0
    risk_per_trade: float = 0.0015
    atr_len: int = 14
    sl_atr_mult: float = 1.5
    tp_mult: float = 3.0

def run_rule_symbol(df: pd.DataFrame, params: RuleBtParams, equity0=1_000_000):
    df = df.copy()
    atr = df["close"].rolling(params.atr_len).std()  # simplistic ATR substitute
    df["position"] = df["signal"].shift()
    df["returns"] = df["close"].pct_change().shift(-1)
    df["strategy"] = df["position"] * df["returns"]
    df["equity"] = equity0 * (1 + df["strategy"].fillna(0)).cumprod()
    return {
        "equity_curve": df["equity"].tolist(),
        "metrics": {
            "total_return": df["equity"].iloc[-1] / equity0 - 1,
            "max_drawdown": (df["equity"].cummax() - df["equity"]).max() / df["equity"].cummax().max()
        }
    }
