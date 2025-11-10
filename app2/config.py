from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from app.config import config as _legacy
except Exception:
    _legacy = None

@dataclass
class BacktestCfg:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = '10min'
    commission: float = 0.0005
    slippage_bps: float = 1.0
    initial_equity: float = 1_000_000.0

@dataclass
class SymbolsCfg:
    symbols: List[str] = field(default_factory=lambda: ['SBER','GAZP','LKOH','GMKN','ROSN'])

@dataclass
class ModelCfg:
    path: str = 'out/models/ai_strategy.pkl'
    horizon: int = 1
    random_state: int = 42

@dataclass
class RiskCfg:
    max_dd: float = 0.10
    per_trade_risk: float = 0.001
    daily_risk_cap: float = 0.01
    vol_lb: int = 48
    conf_floor_stable: float = 0.55
    conf_floor_volatile: float = 0.60
    max_gross: float = 2.0

@dataclass
class AppConfig:
    bt_cfg: BacktestCfg = field(default_factory=BacktestCfg)
    symbols_cfg: SymbolsCfg = field(default_factory=SymbolsCfg)
    model_cfg: ModelCfg = field(default_factory=ModelCfg)
    risk_cfg: RiskCfg = field(default_factory=RiskCfg)

config = _legacy if _legacy is not None else AppConfig()
