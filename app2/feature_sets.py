from __future__ import annotations
from typing import Dict, List

# Базовое ядро фич – будет присутствовать во всех наборах.
CORE_FEATURES: List[str] = [
    "ret_3",
    "ret_12",
    "ema_12",
    "ema_48",
    "atr_14",
    "vol_ema_24",
    "rsi_14",
]

# Короткий и средний моментум по цене
MOMENTUM_SHORT: List[str] = [
    "ret_3",
    "ret_6",
    "ret_12",
]

MOMENTUM_LONG: List[str] = [
    "ret_24",
    "ret_48",
    "ret_96",
]

# Трендовые признаки – быстрые и медленные EMA в относительном виде
TREND_FAST: List[str] = [
    "ema_3",
    "ema_6",
    "ema_12",
]

TREND_SLOW: List[str] = [
    "ema_24",
    "ema_48",
    "ema_96",
]

# Волатильность и отклонение от среднего
VOLATILITY: List[str] = [
    "atr_3",
    "atr_6",
    "atr_12",
    "atr_24",
    "atr_48",
    "atr_96",
    "zscore_24",
]

# Объём и его сглаженные варианты
VOLUME: List[str] = [
    "vol_ema_3",
    "vol_ema_6",
    "vol_ema_12",
    "vol_ema_24",
    "vol_ema_48",
    "vol_ema_96",
    "obv",
]

# Осцилляторы / перекупленность‑перепроданность и уровни
OSCILLATORS: List[str] = [
    "rsi_14",
    "macd",
    "stoch_k",
    "cci_20",
    "williams_r",
]

PRICE_LEVELS: List[str] = [
    "vwap",
]

def _dedup(seq):
    """Убирает дубли, сохраняя порядок."""
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

FEATURE_SETS: Dict[str, List[str]] = {
    # 1. Ядро + короткий моментум + осцилляторы
    "fs_core_momo_short_osc": _dedup(
        CORE_FEATURES
        + MOMENTUM_SHORT
        + OSCILLATORS
    ),

    # 2. Ядро + длинный моментум + осцилляторы
    "fs_core_momo_long_osc": _dedup(
        CORE_FEATURES
        + MOMENTUM_LONG
        + OSCILLATORS
    ),

    # 3. Ядро + быстрый тренд + волатильность
    "fs_core_trend_fast_vol": _dedup(
        CORE_FEATURES
        + TREND_FAST
        + VOLATILITY
    ),

    # 4. Ядро + медленный тренд + волатильность
    "fs_core_trend_slow_vol": _dedup(
        CORE_FEATURES
        + TREND_SLOW
        + VOLATILITY
    ),

    # 5. Ядро + волатильность + объём
    "fs_core_vol_volume": _dedup(
        CORE_FEATURES
        + VOLATILITY
        + VOLUME
    ),

    # 6. Ядро + моментум (короткий) + медленный тренд
    "fs_core_momo_trend_mix": _dedup(
        CORE_FEATURES
        + MOMENTUM_SHORT
        + TREND_SLOW
    ),

    # 7. Ядро + осцилляторы + уровни (vwap, zscore)
    "fs_core_price_osc": _dedup(
        CORE_FEATURES
        + OSCILLATORS
        + PRICE_LEVELS
        + ["zscore_24"]
    ),

    # 8. Лёгкий «всеядный» набор: понемногу всего
    "fs_core_all_light": _dedup(
        CORE_FEATURES
        + MOMENTUM_SHORT
        + TREND_SLOW[:2]
        + VOLATILITY[:3]
        + VOLUME[:3]
        + OSCILLATORS[:2]
        + PRICE_LEVELS
    ),
}
