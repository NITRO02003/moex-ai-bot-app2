from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd

@dataclass
class RiskParams:
    per_trade_risk: float = 0.001
    daily_risk_cap: float = 0.01
    vol_lb: int = 48
    conf_floor_stable: float = 0.55
    conf_floor_volatile: float = 0.60
    max_gross: float = 1.0  # максимально 1x по умолчанию

def realized_vol(close: pd.Series, lb: int = 48) -> pd.Series:
    close = close.astype(float)
    ret = close.pct_change().rolling(lb).std()
    return ret.replace(0.0, np.nan).ffill().bfill().fillna(0.0)

def conf_gate(p: pd.Series, close: pd.Series, rp: RiskParams) -> pd.Series:
    """Вернёт confidence-floor по каждому бару в зависимости от волатильности.

    В более волатильных режимах поднимаем минимально допустимую вероятность,
    чтобы реже входить в рынок.
    """
    close = close.astype(float)
    if isinstance(p, pd.Series):
        p = p.astype(float).reindex(close.index).ffill().bfill()
    else:
        p = pd.Series(float(p), index=close.index)

    vol = realized_vol(close, rp.vol_lb).reindex(close.index).ffill().bfill()
    q = vol.rolling(200, min_periods=20).quantile(0.5).reindex(close.index).ffill().bfill()
    is_volatile = vol >= q

    stable_floor = float(getattr(rp, "conf_floor_stable", 0.55))
    volatile_floor = float(getattr(rp, "conf_floor_volatile", stable_floor))

    floor_values = np.where(is_volatile, volatile_floor, stable_floor)
    floor = pd.Series(floor_values, index=close.index, name="conf_floor")
    return floor.clip(0.5, 0.99)

def position_size(close: pd.Series, p: Union[float, pd.Series], equity: float, rp: RiskParams) -> pd.Series:
    """Размер позиции в деньгах, ограниченный max_gross * equity.

    close  - цена
    p      - вероятность (float или Series)
    equity - стартовый капитал
    rp     - параметры риска
    """
    close = close.astype(float)
    vol = realized_vol(close, rp.vol_lb).reindex(close.index).replace(0.0, np.nan).ffill().bfill()

    if isinstance(p, pd.Series):
        p_series = p.astype(float).reindex(close.index).ffill().bfill()
    else:
        p_series = pd.Series(float(p), index=close.index)

    p_series = p_series.clip(lower=1e-6, upper=1.0)

    dollar_risk = max(float(rp.per_trade_risk) * float(equity), 1.0)
    denom = (vol * p_series).replace(0.0, np.nan).ffill().bfill()
    notional = dollar_risk / denom

    max_gross = float(getattr(rp, "max_gross", 1.0))
    max_notional = max_gross * float(equity)
    notional = notional.clip(lower=0.0, upper=max_notional)

    return notional.reindex(close.index).ffill().bfill()

def from_config() -> RiskParams:
    """Считать RiskParams из config.risk_cfg (если он есть)."""
    try:
        from .config import config
        rcfg = getattr(config, "risk_cfg", None)
        if rcfg is None:
            return RiskParams()
        return RiskParams(
            per_trade_risk=getattr(rcfg, "per_trade_risk", 0.001),
            daily_risk_cap=getattr(rcfg, "daily_risk_cap", 0.01),
            vol_lb=getattr(rcfg, "vol_lb", 48),
            conf_floor_stable=getattr(rcfg, "conf_floor_stable", 0.55),
            conf_floor_volatile=getattr(rcfg, "conf_floor_volatile", 0.60),
            max_gross=getattr(rcfg, "max_gross", 1.0),
        )
    except Exception:
        return RiskParams()
