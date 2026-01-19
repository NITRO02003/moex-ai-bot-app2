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


def _calc_ma(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].rolling(window=window, min_periods=window).mean()

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
    roll_L = low.rolling(window=window, min_periods=min_valid).quantile(0.15).shift(1)
    roll_U = high.rolling(window=window, min_periods=min_valid).quantile(0.85).shift(1)
    roll_H = roll_U - roll_L
    roll_M = (roll_L + roll_U) / 2.0

    # Высота диапазона в процентах к цене (каузально: делим на предыдущий close)
    height_pct = roll_H / close.shift(1)
    mask_height = (
        (height_pct >= params.min_range_height_pct)
        & (height_pct <= params.max_range_height_pct)
    )

    # Slope-фильтр: измеряем наклон MA в относительных единицах (каузально)
    ma = _calc_ma(df, window)
    slope_raw = (ma - ma.shift(window)) / float(max(window, 1))
    slope_abs = slope_raw.abs().shift(1)
    slope_pct = slope_abs / close.shift(1)
    slope_k = params.slope_k
    mask_slope = slope_pct < slope_k

    # Робастный ATR для фильтра по волатильности
    atr_window = params.atr_window
    min_valid_atr = max(
        int(atr_window * params.atr_min_valid_frac),
        params.atr_min_valid_bars,
    )
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=min_valid_atr).mean()
    atr_pct = (atr / close).shift(1)

    mask_vol = (
        atr_pct.notna()
        & (atr_pct >= params.atr_pct_min)
        & (atr_pct <= params.atr_pct_max)
    )

    # Валидный бокс: L/U, высота и волатильность в допустимых пределах
    # slope пока используем только в диагностике
    mask_range = roll_L.notna() & roll_U.notna() & mask_height & mask_vol

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
    shadow = roll_H * params.shadow_pct
    alpha = roll_H * params.entry_zone_alpha

    long_zone_low = roll_L - shadow
    long_zone_high = roll_L + alpha

    # Бар касается зоны, если его диапазон [low, high] пересекается с зоной
    is_in_zone = (low <= long_zone_high) & (high >= long_zone_low)

    # Простой breakout вниз: закрытие значительно ниже L
    breakout_level = roll_L - (shadow * 2.0)
    is_breakout = close < breakout_level
    out.loc[is_breakout, "v3_breakout"] = True

    # Кандидаты на вход: есть бокс, бар коснулся зоны, не breakout
    signal_long = mask_range & is_in_zone & (~is_breakout)

    out.loc[signal_long, "v3_signal"] = 1.0

    # Диагностика
    rolling_diag: Dict[str, Any] = {
        "roll_L_notna_frac": float(roll_L.notna().mean()) if len(roll_L) > 0 else 0.0,
        "valid_box_frac": float(mask_range.mean()) if len(mask_range) > 0 else 0.0,
        "mask_height_frac": float(mask_height.mean()) if len(mask_height) > 0 else 0.0,
        "mask_slope_frac": float(mask_slope.mean()) if len(mask_slope) > 0 else 0.0,
        "mask_vol_frac": float(mask_vol.mean()) if len(mask_vol) > 0 else 0.0,
        "atr_notna_frac": float(atr_pct.notna().mean()) if len(atr_pct) > 0 else 0.0,
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