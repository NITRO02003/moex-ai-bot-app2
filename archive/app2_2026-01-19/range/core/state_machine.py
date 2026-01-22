"""Vectorized State Machine v1 for Range core_v4.

На этом шаге мы перестаём делегировать торговую логику в legacy-функцию.
Вместо этого, в core/state_machine.py содержится перенос текущей векторизованной
rolling-логики Range V3 (из app2/range/range_v3.py) в виде приватной функции
_run_v3_vectorized(...).

Дополнительно:
- считаем геометрию диапазонов (geometry.compute_geometry) и пишем агрегаты в debug_info["geometry"].
- геометрия НЕ влияет на сигналы (чистая аналитика).

Цели:
- сохранить идентичное поведение engine=core относительно текущего Range V3 baseline,
  но при этом убрать зависимость core от вызова run_range_v3_for_symbol.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ..range_v3 import RangeV3Params
from .blocks import (
    calc_atr_pct,
    calc_breakout_mask,
    calc_confirm_mask,
    calc_deadzone_mask,
    calc_entry_signal,
    calc_entry_zone,
    calc_height_mask,
    calc_recent_breakout,
    calc_roll_levels,
    calc_slope_mask,
    calc_vol_mask,
)
from .geometry import compute_geometry

def _params_snapshot(params: RangeV3Params) -> Dict[str, Any]:
    """Снимок ключевых параметров для диагностики расхождений core vs legacy."""
    keys = [
        "range_window_bars",
        "min_range_height_pct",
        "max_range_height_pct",
        "entry_zone_alpha",
        "slope_k",
        "slope_window",
        "atr_window",
        "atr_pct_min",
        "atr_pct_max",
        "sl_pct",
        "tp_pct",
        "max_bars_in_trade",
        "risk_pct_per_trade",
    ]
    snap: Dict[str, Any] = {}
    for k in keys:
        if hasattr(params, k):
            snap[k] = getattr(params, k)
    return snap


def _run_v3_vectorized(df: pd.DataFrame, params: RangeV3Params) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Honest (rolling) Range V3 baseline для backtest.

    Шаг 1: строим каузальные боксы L/U/M по прошлым барам.
    Шаг 2: входим в лонг при касании зоны вокруг L.
    Фильтры по высоте/наклону на этом этапе используются только косвенно (через форму бокса).
    """
    if df.empty:
        out = df.copy()
        out["v3_signal"] = 0.0
        out["v3_L"] = np.nan
        out["v3_U"] = np.nan
        out["v3_M"] = np.nan
        out["v3_segment_quality"] = None
        out["v3_breakout"] = False
        debug_info: Dict[str, Any] = {
            "rolling_diag": {
                "roll_L_notna_frac": 0.0,
                "valid_box_frac": 0.0,
                "signals_count": 0,
            }
        }
        debug_info["params_snapshot"] = _params_snapshot(params)
        return out, debug_info

    high = df["high"]
    low = df["low"]
    close = df["close"]

    window = params.range_window_bars

    # Rolling-квантили по прошлым барам
    # Разреженные данные: используем смягчённый min_periods, чтобы не получить сплошной NaN.
    min_valid = max(int(window * 0.4), 10)
    roll_L, roll_U, roll_H, roll_M = calc_roll_levels(low, high, window, min_valid)

    # Высота диапазона в процентах к цене (каузально: делим на предыдущий close)
    height_pct, mask_height = calc_height_mask(
        roll_H, close, params.min_range_height_pct, params.max_range_height_pct
    )

    # Slope-фильтр: измеряем наклон MA в относительных единицах (каузально)
    slope_pct, mask_slope = calc_slope_mask(close, window, params.slope_k)

    # Робастный ATR для фильтра по волатильности
    atr_window = params.atr_window
    min_valid_atr = max(
        int(atr_window * params.atr_min_valid_frac),
        params.atr_min_valid_bars,
    )
    high = df["high"]
    low = df["low"]
    atr_pct = calc_atr_pct(high, low, close, atr_window, min_valid_atr)
    mask_vol = calc_vol_mask(atr_pct, params.atr_pct_min, params.atr_pct_max)
    mask_deadzone = calc_deadzone_mask(atr_pct, params.deadzone_min_atr_pct)

    # Валидный бокс: L/U, высота и волатильность в допустимых пределах
    # slope пока используем только в диагностике
    mask_range = roll_L.notna() & roll_U.notna() & mask_height & mask_vol & mask_deadzone

    out = df.copy()
    out["v3_signal"] = 0.0
    out["v3_L"] = np.nan
    out["v3_U"] = np.nan
    out["v3_M"] = np.nan
    out["v3_segment_quality"] = None
    out["v3_breakout"] = False

    out.loc[mask_range, "v3_L"] = roll_L[mask_range]
    out.loc[mask_range, "v3_U"] = roll_U[mask_range]
    out.loc[mask_range, "v3_M"] = roll_M[mask_range]
    out.loc[mask_range, "v3_segment_quality"] = "ROLLING"

    # --- Логика входа (long-only baseline) ---
    # Зона входа вокруг L: [L - shadow, L + alpha]
    long_zone_low, long_zone_high, shadow, _alpha = calc_entry_zone(
        roll_L, roll_H, params.shadow_pct, params.entry_zone_alpha
    )

    # Бар касается зоны, если его диапазон [low, high] пересекается с зоной
    is_in_zone = (low <= long_zone_high) & (high >= long_zone_low)

    # Простой breakout вниз: закрытие значительно ниже L
    _breakout_level, is_breakout = calc_breakout_mask(close, roll_L, shadow)
    out.loc[is_breakout, "v3_breakout"] = True

    confirm_mask = calc_confirm_mask(is_in_zone, int(getattr(params, "min_confirmations", 1)))
    recent_breakout = calc_recent_breakout(is_breakout, int(getattr(params, "lock_bars_after_breakout", 0)))

    # Кандидаты на вход: есть бокс, бар коснулся зоны, не breakout
    signal_long = calc_entry_signal(mask_range, confirm_mask, is_breakout) & (~recent_breakout)

    out.loc[signal_long, "v3_signal"] = 1.0

    # Диагностика
    rolling_diag: Dict[str, Any] = {
        "roll_L_notna_frac": float(roll_L.notna().mean()) if len(roll_L) > 0 else 0.0,
        "valid_box_frac": float(mask_range.mean()) if len(mask_range) > 0 else 0.0,
        "mask_height_frac": float(mask_height.mean()) if len(mask_height) > 0 else 0.0,
        "mask_slope_frac": float(mask_slope.mean()) if len(mask_slope) > 0 else 0.0,
        "mask_vol_frac": float(mask_vol.mean()) if len(mask_vol) > 0 else 0.0,
        "mask_deadzone_frac": float(mask_deadzone.mean()) if len(mask_deadzone) > 0 else 0.0,
        "atr_notna_frac": float(atr_pct.notna().mean()) if len(atr_pct) > 0 else 0.0,
        "confirm_mask_frac": float(confirm_mask.mean()) if len(confirm_mask) > 0 else 0.0,
        "recent_breakout_frac": float(recent_breakout.mean()) if len(recent_breakout) > 0 else 0.0,
        "signals_count": int(out["v3_signal"].sum()),
    }
    debug_info: Dict[str, Any] = {"rolling_diag": rolling_diag}

    debug_info["params_snapshot"] = _params_snapshot(params)

    return out, debug_info

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def build_range_states(df: pd.DataFrame, params: RangeV3Params) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Построить сигналы и диагностические данные для одного тикера.

    Возвращает:
    - sig_df: DataFrame с сигналами (как в Range V3 baseline),
    - debug_info: словарь с диагностикой + секция geometry.
    """
    sig_df, debug_info = _run_v3_vectorized(df, params)

    # Геометрия диапазонов (аналитика, не влияет на сигналы).
    try:
        df_geo = compute_geometry(df, params)
        geom_info: Dict[str, object] = {}

        if df_geo is not None and not df_geo.empty:
            valid_mask = df_geo.get("geo_valid_box")
            cls = df_geo.get("geo_class")
            h_pct = df_geo.get("geo_H_pct")

            if valid_mask is not None:
                geom_info["valid_frac"] = _safe_float(valid_mask.mean())
                geom_info["valid_bars"] = int(valid_mask.sum())
                geom_info["total_bars"] = int(len(valid_mask))

            if cls is not None:
                geom_info["class_counts"] = {str(k): int(v) for k, v in cls.value_counts(dropna=False).items()}

            if h_pct is not None:
                hp = h_pct.dropna()
                if len(hp) > 0:
                    geom_info["h_pct_mean"] = _safe_float(hp.mean())
                    geom_info["h_pct_p25"] = _safe_float(hp.quantile(0.25))
                    geom_info["h_pct_p50"] = _safe_float(hp.quantile(0.50))
                    geom_info["h_pct_p75"] = _safe_float(hp.quantile(0.75))

        if isinstance(debug_info, dict):
            debug_info = dict(debug_info)
            debug_info["geometry"] = geom_info

    except Exception as exc:
        if isinstance(debug_info, dict):
            debug_info = dict(debug_info)
            debug_info["geometry_error"] = repr(exc)

    return sig_df, debug_info