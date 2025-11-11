from __future__ import annotations
import numpy as np, pandas as pd

def summarize(equity: pd.Series, pnl: pd.Series):
    pnl = pnl.fillna(0.0)
    eq = equity.ffill().fillna(0.0)  # Исправлено: ffill() вместо fillna(method='ffill')
    ret = pnl / eq.shift(1).replace(0, np.nan).fillna(eq.iloc[0] if len(eq) else 1.0)

    wins = pnl[pnl>0].sum()
    losses = -pnl[pnl<0].sum()
    profit_factor = float(wins/losses) if losses>0 else float('inf')
    wr = float((pnl>0).mean()) if len(pnl) else 0.0
    avg_win = float(pnl[pnl>0].mean()) if (pnl>0).any() else 0.0
    avg_loss = float(-pnl[pnl<0].mean()) if (pnl<0).any() else 0.0
    expectancy = float(avg_win/(avg_loss+1e-9)) if avg_loss>0 else float('inf')
    sharpe = float((ret.mean()/ (ret.std(ddof=1)+1e-12))*np.sqrt(252*6.5*6)) if len(ret)>2 else 0.0

    dd = (eq/eq.cummax() - 1.0).min() if len(eq)>1 else 0.0
    calmar = float((eq.iloc[-1]/eq.iloc[0]-1.0)/abs(dd)) if dd<0 else float('inf')

    freq = float((pnl != 0).astype(int).rolling(144).sum().mean())

    return {
        'final_equity': float(eq.iloc[-1]) if len(eq) else 0.0,
        'total_return': float(eq.iloc[-1]/eq.iloc[0]-1.0) if len(eq)>1 else 0.0,
        'max_drawdown': float(dd),
        'win_rate': wr,
        'profit_factor': profit_factor,
        'expectancy_ratio': expectancy,
        'sharpe_like': sharpe,
        'calmar_like': calmar,
        'trade_freq_proxy': freq
    }