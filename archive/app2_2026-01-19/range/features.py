from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _rolling_by_segments(
    series: pd.Series,
    window: int,
    func: str,
) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid_idx = series.index[series.notna()]
    if len(valid_idx) == 0:
        return out
    seg = pd.to_numeric(series.loc[valid_idx], errors="coerce")
    if func == "mean":
        rolled = seg.rolling(window, min_periods=window).mean()
    elif func == "std":
        rolled = seg.rolling(window, min_periods=window).std()
    elif func == "ema":
        rolled = seg.ewm(span=window, adjust=False, min_periods=window).mean()
    elif func == "ewm_std":
        rolled = seg.ewm(span=window, adjust=False, min_periods=window).std(bias=False)
    else:
        raise ValueError(f"Unsupported rolling func: {func}")
    out.loc[valid_idx] = rolled.to_numpy()
    return out


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует наличие колонки datetime и сортировку по времени."""
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "begin" in df.columns:
        dt = pd.to_datetime(df["begin"])
    else:
        # если нет явной временной метки — пробуем индекс
        dt = pd.to_datetime(df.index)
    out = df.copy()
    out["datetime"] = dt
    out = out.sort_values("datetime").reset_index(drop=True)
    return out


def _compute_atr(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_len: int,
) -> pd.Series:
    """Упрощённый ATR для feature-engineering."""
    close_shifted = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - close_shifted).abs()
    tr3 = (low - close_shifted).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_len, min_periods=atr_len).mean()
    return atr


def add_basic_range_features(
    df: pd.DataFrame,
    ma_len: int = 50,
    band_mult: float = 1.5,
    atr_len: int = 14,
    extra_atr_lens: Optional[Iterable[int]] = None,
    ma_mode: str = "sma",
) -> pd.DataFrame:
    """Добавляет базовые фичи для анализа range-режима.

    Фичи строятся на тех же принципах, что и Range V0 стратегия, но
    могут использоваться отдельно для анализа / AI.

    Добавляемые колонки (если есть нужные исходные данные):

    - datetime          : нормализованный timestamp
    - ma_close          : скользящая средняя close
    - std_close         : скользящее std close
    - band_lower / band_upper
    - dist_from_ma      : (close - ma) / ma
    - band_pos          : положение внутри полос (0=у нижней, 1=у верхней)
    - atr_<len>         : ATR заданной длины
    - atr_<len>_pct     : ATR / close
    - bar_range         : high - low
    - bar_range_pct     : (high - low) / close
    - bar_body          : |close - open|
    - bar_body_pct      : |close - open| / close
    """
    if df.empty:
        return df.copy()

    extra_atr_lens = list(extra_atr_lens or [])
    if atr_len not in extra_atr_lens:
        extra_atr_lens.append(atr_len)

    out = _ensure_datetime(df)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    # скользящая и полосы
    if ma_mode == "ema":
        ma = _rolling_by_segments(close, ma_len, "ema")
        std = _rolling_by_segments(close, ma_len, "ewm_std")
    elif ma_mode == "sma":
        ma = _rolling_by_segments(close, ma_len, "mean")
        std = _rolling_by_segments(close, ma_len, "std")
    else:
        raise ValueError(f"Unsupported ma_mode: {ma_mode}")
    lower_band = ma - band_mult * std
    upper_band = ma + band_mult * std

    out["ma_close"] = ma
    out["std_close"] = std
    out["band_lower"] = lower_band
    out["band_upper"] = upper_band

    with np.errstate(divide="ignore", invalid="ignore"):
        out["dist_from_ma"] = (close - ma) / ma.replace(0, np.nan)
        band_width = (upper_band - lower_band).replace(0, np.nan)
        out["band_pos"] = (close - lower_band) / band_width

    # returns + volatility/momentum
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ret_1"] = close.pct_change(1, fill_method=None)
        out["ret_3"] = close.pct_change(3, fill_method=None)
        out["ret_6"] = close.pct_change(6, fill_method=None)
    ret_1 = out["ret_1"]
    out["ret_mean_20"] = _rolling_by_segments(ret_1, ma_len, "mean")
    out["ret_vol_20"] = _rolling_by_segments(ret_1, ma_len, "std")

    # ATR-фичи
    for l in extra_atr_lens:
        atr = _compute_atr(close, high, low, int(l))
        col = f"atr_{l}"
        out[col] = atr
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{col}_pct"] = atr / close.replace(0, np.nan)

    # баровые фичи
    bar_range = (high - low).abs()
    out["bar_range"] = bar_range
    with np.errstate(divide="ignore", invalid="ignore"):
        out["bar_range_pct"] = bar_range / close.replace(0, np.nan)

    if "open" in out.columns:
        body = (close - out["open"].astype(float)).abs()
        out["bar_body"] = body
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bar_body_pct"] = body / close.replace(0, np.nan)

    return out
