from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from .config import config

# Не тянем config.* на уровне объявлений — ставим безопасные дефолты
@dataclass
class RiskParams:
    max_dd: float = 0.10                 # допустимая просадка по эквити (10%)
    per_trade_risk: float = 0.001        # риск на сделку (0.1%)
    daily_risk_cap: float = 0.01         # дневной лимит убытка (1%)
    vol_lb: int = 48                      # окно для волатильности (интравал=10м -> ~1 день)
    conf_floor_stable: float = 0.55       # порог уверенности в спокойном режиме
    conf_floor_volatile: float = 0.60     # порог уверенности в волатильном режиме
    max_gross: float = 2.0                # макс. брутто-экспозиция (на будущее)

def from_config() -> RiskParams:
    """Заполняем из config.risk_cfg, если поля там есть (обратная совместимость)."""
    rp = RiskParams()
    rc = getattr(config, "risk_cfg", None)
    if rc is None:
        return rp
    for f in ("max_dd","per_trade_risk","daily_risk_cap","vol_lb",
              "conf_floor_stable","conf_floor_volatile","max_gross"):
        if hasattr(rc, f):
            setattr(rp, f, getattr(rc, f))
    return rp

def realized_vol(close: pd.Series, lb: int) -> pd.Series:
    r = close.pct_change().fillna(0)
    return r.rolling(lb).std().replace(0, np.nan).fillna(r.std())

def regime(close: pd.Series, lb: int) -> pd.Series:
    rv = realized_vol(close, lb)
    med = rv.rolling(lb).median()
    return (rv > med).rename("volatile")

def conf_gate(p: pd.Series, close: pd.Series, rp: RiskParams) -> pd.Series:
    reg = regime(close, rp.vol_lb).reindex(p.index).fillna(False)
    floor = np.where(reg, rp.conf_floor_volatile, rp.conf_floor_stable)
    return p.where(p >= pd.Series(floor, index=p.index), 0.0)

def position_size(close: pd.Series, p_up: pd.Series, equity: float, rp: RiskParams) -> pd.Series:
    p_eff = conf_gate(p_up, close, rp)
    vol = realized_vol(close, rp.vol_lb)
    risk_dollars = equity * rp.per_trade_risk
    size = (risk_dollars / (vol * close.replace(0,np.nan))).clip(upper=equity).fillna(0.0)
    # линейно от 0.5 к 1.0 по вероятности
    return size * ((p_eff - 0.5).clip(lower=0) / 0.5)

def apply_daily_risk_cap(pnl: pd.Series, equity0: float, rp: RiskParams) -> pd.Series:
    if pnl.empty:
        return pnl
    ec = pnl.cumsum() + equity0
    g = ec.groupby(ec.index.date)
    ddaily = g.apply(lambda s: (s - s.iloc[0]))
    capped = ddaily.groupby(level=0).transform(lambda s: s.clip(upper=equity0*rp.daily_risk_cap))
    per_bar = capped.groupby(level=0).diff()
    per_bar = per_bar.fillna(capped.groupby(level=0).apply(lambda s: s.iloc[0]))
    per_bar.index = pnl.index
    return per_bar

def dd_halt(equity_curve: pd.Series, rp: RiskParams) -> pd.Series:
    peak = equity_curve.cummax()
    dd = (equity_curve/peak - 1.0).fillna(0.0)
    return dd <= -rp.max_dd
