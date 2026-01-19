from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RangeParams:
    """Параметры базовой range-стратегии (long-only, V0).

    Эти параметры соответствуют структуре defaults.RangeParams
    в файле app2/range/config.json.
    """

    ma_len: int = 50
    band_mult: float = 1.5
    atr_len: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 2.0
    max_hold_bars: int = 48
    min_hold_bars: int = 0
    cooldown_bars: int = 0
    use_time_filter: bool = False
    session_start: str = "10:00"
    session_end: str = "18:40"
    min_atr_pct: float = 0.0005
    max_atr_pct: float = 0.03
    min_volume: float = 0.0

def _compute_atr(close: pd.Series, high: pd.Series, low: pd.Series, atr_len: int) -> pd.Series:
    """Упрощённый ATR для range-фильтра.

    Не используется для риск-менеджмента, только как фильтр волатильности,
    поэтому достаточно простой реализации.
    """
    close_shifted = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - close_shifted).abs()
    tr3 = (low - close_shifted).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_len, min_periods=atr_len).mean()
    return atr

def _time_filter_index(dt_index: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    """Маска по времени суток для intraday-баров."""
    try:
        start_h, start_m = map(int, start.split(":"))
        end_h, end_m = map(int, end.split(":"))
    except Exception:
        # если формат некорректный, не фильтруем по времени
        return pd.Series(True, index=dt_index)

    t = dt_index.time
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    minutes = pd.Index([ti.hour * 60 + ti.minute for ti in t])
    return (minutes >= start_minutes) & (minutes <= end_minutes)

def generate_range_signals(
    df: pd.DataFrame,
    params: RangeParams,
    regime_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Генерация сигналов для range-стратегии (long-only).

    На выходе DataFrame с колонкой ``signal`` (0 или 1), где 1 означает
    целевую long-позицию. Шорт не используется.

    Ожидаемые колонки входного df:
    - datetime или begin (будет конвертировано в datetime);
    - open, high, low, close;
    - volume (для фильтра по ликвидности, опционально).

    Параметры params должны соответствовать RangeParams / config.json.
    """
    if df.empty:
        return df.assign(signal=pd.Series(dtype=float))

    df = df.copy()

    # временной индекс
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "begin" in df.columns:
        dt = pd.to_datetime(df["begin"])
    else:
        # без явной временной колонки просто используем Range V0 без time-фильтра
        dt = pd.to_datetime(df.index)

    df["datetime"] = dt
    df = df.sort_values("datetime").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # базовые фильтры
    base_mask = pd.Series(True, index=df.index)

    # режимный фильтр (range)
    if regime_mask is not None:
        # приводим к индексу df
        if not isinstance(regime_mask, pd.Series):
            regime_mask = pd.Series(regime_mask, index=df.index)
        else:
            # если маска задавалась по исходному индексу, переиндексируем
            regime_mask = regime_mask.reindex(df.index)

        regime_mask = regime_mask.fillna(False).astype(bool)
        base_mask &= regime_mask

    # фильтр по времени сессии
    if params.use_time_filter:
        time_mask = _time_filter_index(df["datetime"], params.session_start, params.session_end)
        base_mask &= time_mask

    # ATR и фильтр по волатильности
    atr = _compute_atr(close, high, low, params.atr_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = atr / close.replace(0, np.nan)

    vol_mask = pd.Series(True, index=df.index)
    if params.min_atr_pct is not None:
        vol_mask &= atr_pct >= float(params.min_atr_pct)
    if params.max_atr_pct is not None and params.max_atr_pct > 0:
        vol_mask &= atr_pct <= float(params.max_atr_pct)

    base_mask &= vol_mask.fillna(False)

    # фильтр по объёму
    if "volume" in df.columns and params.min_volume is not None:
        vol = df["volume"].fillna(0).astype(float)
        base_mask &= vol >= float(params.min_volume)

    # скользящая и полосы (аналог Bollinger)
    ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
    std = close.rolling(params.ma_len, min_periods=params.ma_len).std()
    lower_band = ma - params.band_mult * std
    upper_band = ma + params.band_mult * std

    signal = pd.Series(0, index=df.index, dtype=float)
    in_pos = False
    bars_in_pos = 0

    for i in range(len(df)):
        if not base_mask.iat[i]:
            # вне режима / фильтров — принудительно без позиции
            in_pos = False
            bars_in_pos = 0
            signal.iat[i] = 0.0
            continue

        price = close.iat[i]
        lb = lower_band.iat[i]
        ub = upper_band.iat[i]

        if not in_pos:
            # вход в long: цена существенно ниже средней (ниже нижней полосы)
            if not np.isnan(lb) and price <= lb:
                in_pos = True
                bars_in_pos = 1
                signal.iat[i] = 1.0
            else:
                signal.iat[i] = 0.0
        else:
            # уже в позиции: решаем, держать или выходить
            bars_in_pos += 1

            exit_flag = False

            # минимальное время в позиции
            if params.min_hold_bars and bars_in_pos < params.min_hold_bars:
                exit_flag = False
            else:
                # базовое правило выхода: возврат к средней / верхней полосе
                if not np.isnan(ma.iat[i]) and price >= ma.iat[i]:
                    exit_flag = True
                if not np.isnan(ub) and price >= ub:
                    exit_flag = True

                # ограничение по числу баров в позиции
                if params.max_hold_bars and bars_in_pos >= params.max_hold_bars:
                    exit_flag = True

            if exit_flag:
                in_pos = False
                bars_in_pos = 0
                signal.iat[i] = 0.0
            else:
                signal.iat[i] = 1.0

    df["signal"] = signal
    return df
