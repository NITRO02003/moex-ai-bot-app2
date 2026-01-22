"""Core v4 engine for Range (Phase 0.2 scaffold).

На этом этапе:
- build_params(...) строит RangeV3Params из секции конфига.
- run_core_for_symbol(...) вызывает build_range_states(...) из state_machine.

Логика сигналов и диагностики при этом совпадает с существующей
реализацией Range V3 (run_range_v3_for_symbol) до тех пор, пока
state_machine.build_range_states не будет переписана.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from ..range_v3 import RangeV3Params
from .state_machine import build_range_states


def build_params(cfg_section: Dict[str, object]) -> RangeV3Params:
    """Построить объект параметров core_v4 на основе секции RangeV3.params."""
    return RangeV3Params(cfg_section)


def run_core_for_symbol(df: pd.DataFrame, params: RangeV3Params) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Запуск core_v4 логики для одного тикера (пока — делегирование в state_machine)."""
    return build_range_states(df, params)
