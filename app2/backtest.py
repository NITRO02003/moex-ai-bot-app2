from __future__ import annotations
import json, numpy as np, pandas as pd
from dataclasses import dataclass
from .paths import REPORTS_DIR
from . import strategy as S, risk as R, metrics as MX

@dataclass
class BtParams:
    commission: float = 0.0005
    slippage_bps: float = 1.0
    horizon: int = 1

def run_symbol(prices: pd.DataFrame, model_bundle, rp: R.RiskParams, bt: BtParams,
               equity0: float = 1_000_000.0, threshold: float = 0.55):
    sig = S.signal_and_size(prices, model_bundle, rp, equity0, threshold=threshold)
    close = prices['close'].astype(float)
    ret = close.pct_change().fillna(0.0)
    shares = (sig['size'] / close).fillna(0.0)

    gross = (shares.shift(1) * ret * close).fillna(0.0)

    d_sh = shares.diff().abs().fillna(0.0)
    turn = (d_sh * close)
    commission = turn * float(bt.commission)
    slippage = turn * float(bt.slippage_bps) / 10_000.0
    cost = commission + slippage

    pnl = gross - cost
    pnl = R.apply_daily_risk_cap(pnl, equity0, rp)

    equity = pnl.cumsum() + equity0
    return {
        'signals': sig,
        'equity': equity,
        'pnl': pnl,
        'metrics': MX.summarize(equity, pnl)
    }

def save_report(name: str, res: dict):
    path = REPORTS_DIR / f"{name}_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res['metrics'], f, ensure_ascii=False, indent=2)
    return str(path)
