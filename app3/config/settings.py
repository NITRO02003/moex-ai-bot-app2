from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Settings:
    periods_per_year: float = 252.0
    fees_bps: float = 1.0
    slippage_bps: float = 0.0
    cooldown_bars: int = 20
    min_hold_bars: int = 10
    min_conf: float | None = 0.65
    top_k_per_day: int | None = 3
    atr_threshold: float | None = None
    max_spread_bps: float | None = 5.0
    time_start: str | None = "11:10"
    time_end: str | None = "18:30"
    use_regime: bool = False
