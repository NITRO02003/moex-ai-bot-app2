
from __future__ import annotations
import numpy as np
import pandas as pd

def equity_stats(eq: pd.Series) -> dict:
    eq = eq.astype(float)
    if len(eq) < 2:
        return dict(final_equity=float(eq.iloc[-1]) if len(eq) else 0.0,
                    total_return=0.0, max_drawdown=0.0, calmar=0.0, bars=int(len(eq)))
    ret = float(eq.iloc[-1]/eq.iloc[0]-1.0)
    dd_series = (eq/eq.cummax()-1.0)
    dd = float(dd_series.min()) if len(dd_series) else 0.0
    calmar = (ret/abs(dd)) if dd < 0 else (float("inf") if ret > 0 else 0.0)
    return dict(final_equity=float(eq.iloc[-1]), total_return=ret, max_drawdown=dd, calmar=calmar, bars=int(len(eq)))

def summary_from_pnl(pnl: pd.Series, equity0: float = 1_000_000.0) -> dict:
    pnl = pnl.astype(float).fillna(0.0)
    eq = pnl.cumsum() + float(equity0)
    s = equity_stats(eq)
    wins = float(pnl[pnl>0].sum()); losses = float(pnl[pnl<0].sum())
    s["profit_factor"] = (wins/abs(losses)) if losses < 0 else (float("inf") if wins>0 else 0.0)
    nonzero = pnl[pnl!=0]
    s["win_rate"] = float((nonzero>0).sum())/int(len(nonzero)) if len(nonzero)>0 else 0.0
    s["avg_trade"] = float(nonzero.mean()) if len(nonzero)>0 else 0.0
    return s
