from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class RiskParams:
    max_dd: float = 0.10
    per_trade_risk: float = 0.001
    daily_risk_cap: float = 0.01
    vol_lb: int = 48
    conf_floor_stable: float = 0.55
    conf_floor_volatile: float = 0.60
    max_gross: float = 2.0

def realized_vol(close: pd.Series, lb: int) -> pd.Series:
    r = close.astype(float).pct_change().fillna(0.0)
    return r.rolling(lb).std().replace(0, np.nan).fillna(r.std() if len(r) else 0.0)

def regime(close: pd.Series, lb: int) -> pd.Series:
    rv = realized_vol(close, lb)
    med = rv.rolling(lb).median()
    return (rv > med).rename("volatile")

def simple_conf_gate(p: pd.Series, close: pd.Series, rp: RiskParams | None = None) -> pd.Series:
    rp = rp or RiskParams()
    reg = regime(close, rp.vol_lb).reindex(p.index).fillna(False)
    floor = np.where(reg, rp.conf_floor_volatile, rp.conf_floor_stable)
    return p.where(p >= pd.Series(floor, index=p.index), 0.0)

def conf_gate(p: pd.Series, close: pd.Series, rp: RiskParams | None = None) -> pd.Series:
    try:
        return simple_conf_gate(p, close, rp)
    except Exception:
        return p

def position_size(close: pd.Series, p_up: pd.Series, equity: float, rp: RiskParams | None = None) -> pd.Series:
    rp = rp or RiskParams()
    vol = realized_vol(close, rp.vol_lb).replace(0, np.nan).ffill().bfill()
    risk_dollars = equity * float(rp.per_trade_risk)
    size = (risk_dollars / (vol * close.replace(0, np.nan))).clip(lower=0).fillna(0.0)
    conf = (p_up.clip(lower=0.5) - 0.5) / 0.5
    return (size * conf).rename("size")

def apply_daily_risk_cap(pnl: pd.Series, equity0: float, rp: RiskParams | None = None) -> pd.Series:
    rp = rp or RiskParams()
    if pnl.empty:
        return pnl
    eq = pnl.cumsum() + equity0
    by_day = eq.groupby(eq.index.date)
    day_pnl = by_day.apply(lambda s: s - s.iloc[0])
    cap = float(rp.daily_risk_cap)
    lim = day_pnl.groupby(level=0).transform(lambda s: s.clip(upper=equity0 * cap))
    firsts = lim.groupby(level=0).apply(lambda s: s.iloc[0])
    capped = lim.groupby(level=0).diff().fillna(firsts)
    capped.index = pnl.index
    return capped

def dd_halt(equity_curve: pd.Series, rp: RiskParams | None = None) -> pd.Series:
    rp = rp or RiskParams()
    peak = equity_curve.cummax()
    dd = (equity_curve / peak - 1.0).fillna(0.0)
    return (dd <= -rp.max_dd)
