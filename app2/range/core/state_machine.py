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

    # Валидный бокс: L/U, высота, волатильность и slope в допустимых пределах
    mask_range = roll_L.notna() & roll_U.notna() & mask_height & mask_vol & mask_deadzone & mask_slope

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

    # --- Логика входа (long + short) ---
    # Зона входа вокруг L: [L - shadow, L + alpha]
    long_zone_low, long_zone_high, shadow, alpha = calc_entry_zone(
        roll_L, roll_H, params.shadow_pct, params.entry_zone_alpha
    )
    # Зона входа вокруг U: [U - alpha, U + shadow]
    short_zone_low = roll_U - alpha
    short_zone_high = roll_U + shadow

    # Бар касается зоны, если его диапазон [low, high] пересекается с зоной
    is_in_zone_long = (low <= long_zone_high) & (high >= long_zone_low)
    is_in_zone_short = (high >= short_zone_low) & (low <= short_zone_high)

    # Простой breakout вниз/вверх
    _breakout_level, is_breakout_long = calc_breakout_mask(close, roll_L, shadow)
    breakout_level_short = roll_U + (shadow * 2.0)
    is_breakout_short = close > breakout_level_short
    out.loc[is_breakout_long | is_breakout_short, "v3_breakout"] = True

    confirm_long = calc_confirm_mask(is_in_zone_long, int(getattr(params, "min_confirmations", 1)))
    confirm_short = calc_confirm_mask(is_in_zone_short, int(getattr(params, "min_confirmations", 1)))
    recent_breakout_long = calc_recent_breakout(is_breakout_long, int(getattr(params, "lock_bars_after_breakout", 0)))
    recent_breakout_short = calc_recent_breakout(is_breakout_short, int(getattr(params, "lock_bars_after_breakout", 0)))

    # Кандидаты на вход: есть бокс, бар коснулся зоны, не breakout
    signal_long = calc_entry_signal(mask_range, confirm_long, is_breakout_long) & (~recent_breakout_long)
    signal_short = calc_entry_signal(mask_range, confirm_short, is_breakout_short) & (~recent_breakout_short)
    signal_short = signal_short & (~signal_long)

    out.loc[signal_long, "v3_signal"] = 1.0
    out.loc[signal_short, "v3_signal"] = -1.0

    # Диагностика
    rolling_diag: Dict[str, Any] = {
        "roll_L_notna_frac": float(roll_L.notna().mean()) if len(roll_L) > 0 else 0.0,
        "valid_box_frac": float(mask_range.mean()) if len(mask_range) > 0 else 0.0,
        "mask_height_frac": float(mask_height.mean()) if len(mask_height) > 0 else 0.0,
        "mask_slope_frac": float(mask_slope.mean()) if len(mask_slope) > 0 else 0.0,
        "mask_vol_frac": float(mask_vol.mean()) if len(mask_vol) > 0 else 0.0,
        "mask_deadzone_frac": float(mask_deadzone.mean()) if len(mask_deadzone) > 0 else 0.0,
        "atr_notna_frac": float(atr_pct.notna().mean()) if len(atr_pct) > 0 else 0.0,
        "confirm_mask_frac": float(confirm_long.mean()) if len(confirm_long) > 0 else 0.0,
        "recent_breakout_frac": float(recent_breakout_long.mean()) if len(recent_breakout_long) > 0 else 0.0,
        "signals_count": int((out["v3_signal"] != 0).sum()),
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

def _series_bool(values: pd.Series, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(False, index=index, dtype=bool)
    out = values.reindex(index)
    return out.fillna(False).astype(bool)


def build_regime_gate(
    sig_df: pd.DataFrame,
    params: RangeV3Params,
    mode: str = "off",
    mask_confirm_bars: int = 3,
    mask_min_active_frac: float = 0.67,
    state_candidate_bars: int = 3,
    state_broken_bars: int = 2,
) -> Tuple[pd.Series | None, Dict[str, object], pd.DataFrame]:
    """Построить диагностический regime gate поверх baseline-сигналов.

    Правила этапа Phase 2:
    - gate управляет только разрешением новых входов;
    - gate не меняет exit/risk/features;
    - gate не переписывает baseline-сигналы, а только возвращает allow-mask.

    Возвращает:
    - allow_mask: bool Series либо None для режима off;
    - gate_info: словарь с диагностикой gating;
    - annotated_sig_df: копия sig_df с колонками диагностики gate.
    """
    annotated = sig_df.copy()
    index = annotated.index
    raw_signal = annotated.get("v3_signal", pd.Series(0.0, index=index)).fillna(0.0)
    raw_signal_mask = raw_signal != 0

    base_eligible = (
        annotated.get("v3_L", pd.Series(index=index, dtype=float)).notna()
        & annotated.get("v3_U", pd.Series(index=index, dtype=float)).notna()
        & (~_series_bool(annotated.get("v3_breakout"), index))
    )
    base_eligible = base_eligible.fillna(False).astype(bool)

    allow_mask: pd.Series | None = None
    state_series = pd.Series("off", index=index, dtype=object)

    mode_normalized = str(mode or "off").strip().lower()
    if mode_normalized == "off":
        allow_mask = None
    elif mode_normalized == "mask":
        confirm_bars = max(int(mask_confirm_bars), 1)
        min_frac = float(mask_min_active_frac)
        min_frac = min(max(min_frac, 0.0), 1.0)
        persist_frac = base_eligible.astype(float).rolling(window=confirm_bars, min_periods=confirm_bars).mean()
        allow_mask = (persist_frac >= min_frac).fillna(False)
        state_series.loc[allow_mask] = "active"
        state_series.loc[(~allow_mask) & base_eligible] = "candidate"
        annotated["regime_persist_frac"] = persist_frac
    elif mode_normalized == "state_v0":
        candidate_bars = max(int(state_candidate_bars), 1)
        broken_bars = max(int(state_broken_bars), 1)
        allow_values = []
        state_values = []
        state = "inactive"
        good_streak = 0
        bad_streak = 0
        for eligible in base_eligible.astype(bool).tolist():
            if state == "inactive":
                if eligible:
                    good_streak = 1
                    bad_streak = 0
                    state = "candidate"
                else:
                    good_streak = 0
                    bad_streak = 0
            elif state == "candidate":
                if eligible:
                    good_streak += 1
                    if good_streak >= candidate_bars:
                        state = "active"
                        bad_streak = 0
                else:
                    state = "inactive"
                    good_streak = 0
                    bad_streak = 0
            elif state == "active":
                if eligible:
                    bad_streak = 0
                else:
                    state = "broken"
                    bad_streak = 1
            elif state == "broken":
                if eligible:
                    state = "active"
                    good_streak = candidate_bars
                    bad_streak = 0
                else:
                    bad_streak += 1
                    if bad_streak >= broken_bars:
                        state = "inactive"
                        good_streak = 0
                        bad_streak = 0
            allow_values.append(state == "active")
            state_values.append(state)
        allow_mask = pd.Series(allow_values, index=index, dtype=bool)
        state_series = pd.Series(state_values, index=index, dtype=object)
    else:
        raise ValueError(f"Unsupported regime gate mode: {mode}")

    annotated["regime_base_eligible"] = base_eligible
    annotated["regime_state"] = state_series
    annotated["regime_allow"] = True if allow_mask is None else allow_mask.astype(bool)

    raw_signals = int(raw_signal_mask.sum())
    allowed_signals = raw_signals if allow_mask is None else int((raw_signal_mask & allow_mask).sum())
    blocked_signals = 0 if allow_mask is None else int((raw_signal_mask & (~allow_mask)).sum())
    signal_coverage = float(allowed_signals / raw_signals) if raw_signals > 0 else 0.0
    blocked_entries_share = float(blocked_signals / raw_signals) if raw_signals > 0 else 0.0

    state_counts = state_series.value_counts(dropna=False).to_dict()
    state_fracs = {f"{str(k)}_frac": float(v / len(state_series)) for k, v in state_counts.items()} if len(state_series) > 0 else {}
    gate_info: Dict[str, object] = {
        "mode": mode_normalized,
        "base_eligible_frac": float(base_eligible.mean()) if len(base_eligible) > 0 else 0.0,
        "raw_signals": raw_signals,
        "allowed_signals": allowed_signals,
        "blocked_signals": blocked_signals,
        "blocked_entries_share": blocked_entries_share,
        "signal_coverage": signal_coverage,
        "active_state_frac": float((state_series == "active").mean()) if len(state_series) > 0 else 0.0,
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "mask_confirm_bars": int(mask_confirm_bars),
        "mask_min_active_frac": float(mask_min_active_frac),
        "state_candidate_bars": int(state_candidate_bars),
        "state_broken_bars": int(state_broken_bars),
    }
    gate_info.update(state_fracs)
    return allow_mask, gate_info, annotated
