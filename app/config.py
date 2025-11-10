# config.py
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional

@dataclass
class SymbolsConfig:
    symbols: Tuple[str, ...] = ("SBER", "GAZP", "LKOH", "GMKN", "ROSN")
    tick_size: Dict[str, float] = field(default_factory=lambda: {"SBER":0.01,"GAZP":0.01,"LKOH":0.1,"GMKN":0.1,"ROSN":0.01})
    typical_spread_ticks: Dict[str, int] = field(default_factory=lambda: {"SBER":2,"GAZP":2,"LKOH":3,"GMKN":3,"ROSN":2})

@dataclass
class CostsConfig:
    commission_per_side: float = 0.0005

@dataclass
class RiskConfig:
    max_total_drawdown: float = 0.3

@dataclass
class StrategyConfig:
    use_ai: bool = True
    ai_threshold: float = 0.52  # per aggressive spec

@dataclass
class BacktestConfig:
    start_date: Optional[str] = "2024-01-01"
    end_date: Optional[str] = "2025-10-31"
    initial_equity: float = 1_000_000.0
    data_dir: str = "data"  # ДОБАВЛЯЕМ data_dir
    interval: str = "10min"  # ИЗМЕНИЛИ на 10min

@dataclass
class Config:
    symbols_cfg: SymbolsConfig = field(default_factory=SymbolsConfig)
    costs_cfg: CostsConfig = field(default_factory=CostsConfig)
    risk_cfg: RiskConfig = field(default_factory=RiskConfig)
    strategy_cfg: StrategyConfig = field(default_factory=StrategyConfig)
    bt_cfg: BacktestConfig = field(default_factory=BacktestConfig)

config = Config()