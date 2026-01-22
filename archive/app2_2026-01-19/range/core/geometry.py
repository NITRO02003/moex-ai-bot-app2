"""Geometry v1 for Range core_v4 (аналитический уровень).

На этом этапе:
- вычисляем простые уровни диапазона (geo_L, geo_U, geo_H, geo_M) по скользящему окну;
- помечаем валидные боксы (geo_valid_box);
- вводим примитивную классификацию geo_class: 'A' или 'INVALID';
- считаем относительную высоту geo_H_pct.

ВАЖНО:
- геометрия НЕ влияет на торговые сигналы;
- используется только для диагностики и анализа (debug_info).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..range_v3 import RangeV3Params


_WINDOW_ATTR_CANDIDATES = (
    "box_window",
    "range_window",
    "roll_window",
    "rolling_window",
    "lookback",
    "window",
)


def _pick_window(params: RangeV3Params, default: int = 20) -> int:
    """Подобрать размер rolling-окна из params (если поле существует)."""
    for name in _WINDOW_ATTR_CANDIDATES:
        v = getattr(params, name, None)
        if isinstance(v, int) and v >= 2:
            return v
    return default


def compute_geometry(df: pd.DataFrame, params: RangeV3Params) -> pd.DataFrame:
    """Добавить в df базовые геометрические признаки диапазона.

    Возвращает новый DataFrame с колонками:
    - geo_L, geo_U, geo_H, geo_M
    - geo_H_pct
    - geo_valid_box
    - geo_class
    """
    if df is None or df.empty:
        return df

    window = _pick_window(params, default=20)

    out = df.copy()

    # Rolling levels
    geo_L = out["low"].rolling(window=window, min_periods=1).min()
    geo_U = out["high"].rolling(window=window, min_periods=1).max()
    geo_H = geo_U - geo_L
    geo_M = (geo_U + geo_L) / 2.0

    # Relative height (protect against zero/NaN close)
    close_abs = out["close"].abs().replace(0, pd.NA)
    geo_H_pct = (geo_H / close_abs).astype("float64")

    # Validity: non-null and positive height
    geo_valid_box = geo_H.notna() & (geo_H > 0)

    # Primitive class (analytics-only)
    geo_class = pd.Series("INVALID", index=out.index, dtype="object")
    geo_class.loc[geo_valid_box] = "A"

    out["geo_L"] = geo_L
    out["geo_U"] = geo_U
    out["geo_H"] = geo_H
    out["geo_M"] = geo_M
    out["geo_H_pct"] = geo_H_pct
    out["geo_valid_box"] = geo_valid_box
    out["geo_class"] = geo_class

    return out
