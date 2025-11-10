from __future__ import annotations
from dataclasses import dataclass

try:
    from .config import config
except Exception:
    config = None  # работаем с дефолтами, если конфиг недоступен на этапе импорта


@dataclass
class RiskParams:
    # безопасные дефолты — без обращения к конфигу на этапе импорта
    max_dd: float = 0.10
    per_trade_risk: float = 0.001
    daily_risk_cap: float = 0.01
    vol_lb: int = 48
    conf_floor_stable: float = 0.55
    conf_floor_volatile: float = 0.60
    max_gross: float = 2.0


def from_config() -> RiskParams:
    """
    Аккуратно соберём RiskParams из config.risk_cfg, если он есть.
    Любые отсутствующие поля оставляем по умолчанию.
    """
    rp = RiskParams()
    rc = getattr(config, "risk_cfg", None)
    if rc is None:
        return rp
    for field in rp.__dataclass_fields__:
        if hasattr(rc, field):
            setattr(rp, field, getattr(rc, field))
    return rp
