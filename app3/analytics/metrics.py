from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union

__all__ = ["summarize"]

EPS = 1e-12

def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    try:
        return float(a / b) if (b is not None and abs(float(b)) > EPS) else float(default)
    except Exception:
        return float(default)

def _finite_or_none(x: Union[float, int, np.number, None]) -> Optional[float]:
    """Return float(x) if finite; None for None/NaN/inf or non-castable types."""
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if np.isfinite(xf):
        return xf
    return None

def _compute_equity_metrics(eq: pd.Series, periods_per_year: float) -> Dict[str, Any]:
    if not isinstance(eq, pd.Series):
        eq = pd.Series(eq, dtype=float)
    eq = eq.astype(float).dropna()
    if len(eq) < 2:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": None,
            "volatility_ann": None,
            "sharpe_ann": None,
            "bars": int(len(eq)),
            "final_equity": float(eq.iloc[-1]) if len(eq) else None,
        }

    start = float(eq.iloc[0]); end = float(eq.iloc[-1])
    if abs(start) > EPS:
        total_return = (end / start) - 1.0
    else:
        denom = max(abs(start), abs(end), 1.0)
        total_return = (end - start) / denom

    roll_max = eq.cummax()
    dd_series = eq / roll_max - 1.0
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0

    if max_dd < -EPS:
        calmar = _safe_div(total_return, abs(max_dd), default=np.nan)
    elif total_return > 0:
        calmar = np.inf
    else:
        calmar = 0.0

    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stdev = float(returns.std(ddof=1))
    mean_ret = float(returns.mean())
    sharpe_daily = _safe_div(mean_ret, stdev, default=0.0) if stdev > 0 else 0.0
    sharpe_ann = sharpe_daily * np.sqrt(periods_per_year) if stdev > 0 else 0.0
    vol_ann = stdev * np.sqrt(periods_per_year)

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": _finite_or_none(calmar),
        "volatility_ann": _finite_or_none(vol_ann),
        "sharpe_ann": _finite_or_none(sharpe_ann),
        "bars": int(len(eq)),
        "final_equity": float(end),
    }

def summarize(eq: pd.Series,
              pnl: Optional[pd.Series] = None,
              periods_per_year: float = 252.0) -> Dict[str, Any]:
    metrics = _compute_equity_metrics(eq, periods_per_year=periods_per_year)

    wins = losses = ties = trades = 0
    win_rate: Optional[float] = None

    if pnl is not None:
        if not isinstance(pnl, pd.Series):
            pnl = pd.Series(pnl, dtype=float)
        pnl = pnl.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

        trades = int(len(pnl))
        if trades > 0:
            wins = int((pnl > 0).sum())
            losses = int((pnl < 0).sum())
            ties = trades - wins - losses
            try:
                win_rate = float(wins) / float(trades)
            except Exception:
                win_rate = None

        pnl_sum = float(pnl.sum()) if trades else 0.0
        pnl_mean = float(pnl.mean()) if trades else 0.0
        pnl_std = float(pnl.std(ddof=1)) if trades > 1 else 0.0

        metrics.update({
            "pnl_sum": pnl_sum,
            "pnl_mean": pnl_mean,
            "pnl_std": _finite_or_none(pnl_std),
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": _finite_or_none(win_rate),
        })
    else:
        metrics.update({
            "pnl_sum": 0.0,
            "pnl_mean": 0.0,
            "pnl_std": None,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "win_rate": None,
        })

    return metrics
