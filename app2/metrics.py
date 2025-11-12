# app2/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

__all__ = ["compute_equity_metrics", "summarize"]

EPS = 1e-12

def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    return float(a / b) if (b is not None and abs(b) > EPS) else float(default)

def _finite_or_none(x: float) -> Optional[float]:
    return float(x) if np.isfinite(x) else None

def compute_equity_metrics(eq: pd.Series, periods_per_year: Optional[float] = None) -> Dict[str, Any]:
    """
    Безопасные метрики по equity. Возвращает JSON-дружественные значения (без inf/nan).
    """
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

    start = float(eq.iloc[0])
    end = float(eq.iloc[-1])

    # Total return: защищено от start == 0
    if abs(start) > EPS:
        total_return = (end / start) - 1.0
    else:
        denom = max(abs(start), abs(end), 1.0)
        total_return = (end - start) / denom

    # Max drawdown
    roll_max = eq.cummax()
    dd_series = eq / roll_max - 1.0
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0  # <= 0

    # Calmar
    if max_dd < -EPS:
        calmar = _safe_div(total_return, abs(max_dd), default=np.nan)
    elif total_return > 0:
        calmar = np.inf  # сериализуем как None
    else:
        calmar = 0.0

    # Доходности по шагам
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Если не задано, берём дневную частоту по умолчанию
    if periods_per_year is None:
        periods_per_year = 252.0

    stdev = float(returns.std(ddof=1))
    mean_ret = float(returns.mean())
    sharpe_daily = _safe_div(mean_ret, stdev, default=0.0) if stdev > 0 else 0.0

    sharpe_ann = sharpe_daily * np.sqrt(periods_per_year) if stdev > 0 else 0.0
    vol_ann = stdev * np.sqrt(periods_per_year)

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": _finite_or_none(calmar),          # None вместо inf
        "volatility_ann": _finite_or_none(vol_ann),
        "sharpe_ann": _finite_or_none(sharpe_ann),
        "bars": int(len(eq)),
        "final_equity": float(end),
    }

def summarize(eq: pd.Series,
              pnl: Optional[pd.Series] = None,
              periods_per_year: Optional[float] = None) -> Dict[str, Any]:
    """
    Backward-compatible API: MX.summarize(equity, pnl)
    Возвращает словарь с ключевыми метриками equity + (опционально) агрегатами по pnl,
    включая win_rate.
    """
    metrics = compute_equity_metrics(eq, periods_per_year=periods_per_year)

    # Дополняем PnL-сводкой и winrate
    wins = losses = ties = trades = 0
    win_rate = None

    if pnl is not None:
        if not isinstance(pnl, pd.Series):
            pnl = pd.Series(pnl, dtype=float)
        pnl = pnl.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

        trades = int(len(pnl))
        if trades > 0:
            wins = int((pnl > 0).sum())
            losses = int((pnl < 0).sum())
            ties = trades - wins - losses
            win_rate = float(wins / trades)

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
            "win_rate": _finite_or_none(win_rate),  # 0..1, None если сделок нет
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
