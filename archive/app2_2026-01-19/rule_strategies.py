from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .regime_detector import detect_regime_series


@dataclass
@dataclass
class TrendParams:
    ema_fast: int = 12
    ema_slow: int = 48
    atr_len: int = 14
    trend_thr: float = 2.5
    min_gap_bars: int = 25

    # Optional asymmetric thresholds for long/short entries.
    # If None, fall back to `trend_thr`.
    trend_thr_long: Optional[float] = None
    trend_thr_short: Optional[float] = None

    # Exit threshold in terms of |trend_score|.
    # If None, fall back to `trend_thr`.
    trend_exit_thr: Optional[float] = None
    exit_confirm_bars: int = 1

    # Optional ADX filters. If adx_len == 0, ADX is not used.
    adx_len: int = 0
    adx_min: float = 0.0
    adx_exit_min: float = 0.0

    # Optional slope filters for the slow EMA.
    # If slope_window == 0, slope is not used.
    slope_window: int = 0
    slope_min_up: float = 0.0
    slope_min_down: float = 0.0

    # Optional regime awareness.
    # If True and a "regime" column is present in df, new entries are only
    # opened in regime == "trend". If close_on_regime_change is True,
    # active positions are closed when regime switches away from "trend".
    regime_aware: bool = False
    close_on_regime_change: bool = False


@dataclass
class MeanRevParams:
    rsi_len: int = 14
    rsi_low: float = 25.0
    rsi_high: float = 75.0
    bb_len: int = 20      # boll_window
    bb_k: float = 2.0     # boll_mult
    min_gap_bars: int = 20



@dataclass
class MeanRevV2Params:
    ma_len: int = 48
    atr_len: int = 14
    z_entry: float = 2.0
    z_entry_long: Optional[float] = None
    z_entry_short: Optional[float] = None
    # какие режимы считаем допустимыми для mean-reversion v2
    regime_filter: tuple[str, ...] = ("range", "low_vol")


@dataclass
class BreakoutParams:
    channel_len: int = 20
    confirm_bars: int = 1
    min_gap_bars: int = 20


def _compute_atr(df: pd.DataFrame, atr_len: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_len, min_periods=1).mean()
    return atr


def _rsi(close: pd.Series, length: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)

    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _bollinger(close: pd.Series, length: int, k: float):
    ma = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std(ddof=0)
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower



def _zscore_meanrev_v2(df: pd.DataFrame, ma_len: int, atr_len: int) -> pd.Series:
    """Z-score для mean-reversion v2: (close - EMA(ma_len)) / ATR(atr_len)."""
    close = df["close"].astype(float)

    # EMA по цене
    ma = close.ewm(span=ma_len, adjust=False).mean()

    # ATR — как в трендовой стратегии, если есть OHLC
    if {"high", "low", "close"}.issubset(df.columns):
        atr = _compute_atr(df, atr_len)
    else:
        # fallback: усреднённый диапазон по close
        atr = close.diff().abs().rolling(atr_len, min_periods=1).mean()

    z = (close - ma) / atr.replace(0.0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    return z


def generate_trend_signals(df: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    """Generate trend-following signals based on EMA spread / ATR.

    This implementation:
    - Uses normalized EMA spread as the primary trend_score.
    - Supports asymmetric entry thresholds for long/short.
    - Supports a separate exit threshold with confirmation bars.
    - Optionally filters entries by ADX and slope of the slow EMA.
    - Optionally respects a precomputed `regime` column:
      - new entries only in regime == "trend" (if regime_aware=True),
      - optional forced exit on regime change (close_on_regime_change=True).
    - Interprets `min_gap_bars` as a constraint only on *direction changes*
      (+1 <-> -1), not on exits to flat (0).
    """
    if df.empty:
        df = df.copy()
        df["signal"] = 0
        return df

    df = df.copy()

    # Base EMAs
    ema_fast = df["close"].ewm(span=params.ema_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=params.ema_slow, adjust=False).mean()

    # ATR for normalization
    atr = _compute_atr(df, params.atr_len)
    atr_safe = atr.replace(0.0, np.nan)

    trend_score = (ema_fast - ema_slow) / atr_safe
    trend_score = trend_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["trend_score"] = trend_score

    # Effective thresholds
    thr_long = params.trend_thr_long if params.trend_thr_long is not None else params.trend_thr
    thr_short = params.trend_thr_short if params.trend_thr_short is not None else params.trend_thr
    exit_thr = params.trend_exit_thr if params.trend_exit_thr is not None else params.trend_thr

    # Optional ADX
    adx = None
    if params.adx_len and params.adx_len > 0 and {"high", "low", "close"}.issubset(df.columns):
        adx = _calculate_adx(df["high"], df["low"], df["close"], params.adx_len)

    # Optional slope of the slow EMA
    slope = None
    if params.slope_window and params.slope_window > 0:
        slope = ema_slow.diff(params.slope_window)

    # Optional regime-awareness: we rely on a precomputed "regime" column
    # (e.g. prepared by regime_rule_backtest). If it is absent, we act
    # as if all bars are in a trend regime.
    regime_values = None
    if params.regime_aware and "regime" in df.columns:
        regime_values = df["regime"].to_numpy()

    n = len(df)
    signal = np.zeros(n, dtype=int)

    current = 0  # current position: -1, 0, +1
    last_dir = 0  # last non-zero direction (for min_gap_bars)
    bars_since_dir_change = 10**9
    exit_counter = 0

    for i in range(n):
        score = float(trend_score.iat[i])

        # Regime filter
        is_trend_regime = True
        if regime_values is not None:
            is_trend_regime = regime_values[i] == "trend"

        # ADX filter
        adx_ok = True
        adx_weak = False
        adx_val = None
        if adx is not None:
            adx_val = float(adx.iat[i])
            if params.adx_min > 0.0:
                adx_ok = adx_val >= params.adx_min
            if params.adx_exit_min > 0.0 and current != 0:
                adx_weak = adx_val < params.adx_exit_min

        # Slope filters
        slope_ok_long = True
        slope_ok_short = True
        slope_val = None
        if slope is not None:
            slope_val = float(slope.iat[i])
            if params.slope_min_up > 0.0:
                slope_ok_long = slope_val >= params.slope_min_up
            if params.slope_min_down > 0.0:
                slope_ok_short = slope_val <= -params.slope_min_down

        # Exit logic if we are in a position
        if current != 0:
            weak_trend = abs(score) <= exit_thr
            regime_exit = (
                regime_values is not None
                and params.regime_aware
                and params.close_on_regime_change
                and not is_trend_regime
            )

            if weak_trend or adx_weak or regime_exit:
                exit_counter += 1
            else:
                exit_counter = 0

            if exit_counter >= params.exit_confirm_bars:
                current = 0
                exit_counter = 0

        # Entry logic if we are flat
        if current == 0:
            if not params.regime_aware or is_trend_regime:
                new_dir = 0
                if score > thr_long and adx_ok and slope_ok_long:
                    new_dir = 1
                elif score < -thr_short and adx_ok and slope_ok_short:
                    new_dir = -1

                if new_dir != 0:
                    # min_gap_bars protects only against rapid direction changes
                    if last_dir != 0 and new_dir != last_dir and bars_since_dir_change < params.min_gap_bars:
                        new_dir = 0
                    else:
                        current = new_dir
                        if new_dir != last_dir:
                            last_dir = new_dir
                            bars_since_dir_change = 0

        signal[i] = current
        bars_since_dir_change += 1

    df["signal"] = signal
    return df


def generate_meanrev_signals(
    df: pd.DataFrame,
    params: Optional[MeanRevParams] = None,
    **kwargs,
) -> pd.Series:
    """Mean-reversion стратегия на RSI + Bollinger.

    long, если RSI < rsi_low и close < нижней полосы
    short, если RSI > rsi_high и close > верхней полосы
    Между сменой направления — min_gap_bars.
    """
    if params is None:
        kw = dict(kwargs)
        # совместимость с конфигом: boll_window/boll_mult -> bb_len/bb_k
        if "boll_window" in kw and "bb_len" not in kw:
            kw["bb_len"] = kw.pop("boll_window")
        if "boll_mult" in kw and "bb_k" not in kw:
            kw["bb_k"] = kw.pop("boll_mult")
        params = MeanRevParams(**kw)

    close = df["close"].astype(float)
    rsi = _rsi(close, params.rsi_len)
    _, upper, lower = _bollinger(close, params.bb_len, params.bb_k)

    side = pd.Series(0, index=df.index, dtype=float)
    current = 0
    bars_since_change = 10 ** 9

    for i in range(len(df)):
        c = close.iloc[i]
        r = rsi.iloc[i]
        u = upper.iloc[i]
        l = lower.iloc[i]

        desired = current

        if np.isfinite(r) and np.isfinite(c) and np.isfinite(u) and np.isfinite(l):
            # кандидаты
            if r < params.rsi_low and c < l:
                desired = 1
            elif r > params.rsi_high and c > u:
                desired = -1
            else:
                desired = 0

        if desired != current and bars_since_change < params.min_gap_bars:
            # выдерживаем паузу
            side.iloc[i] = float(current)
            bars_since_change += 1
            continue

        if desired != current:
            current = desired
            bars_since_change = 0
        else:
            bars_since_change += 1

        side.iloc[i] = float(current)

    return side





def generate_meanrev_v2_signals(
    df: pd.DataFrame,
    params: Optional[MeanRevV2Params] = None,
    regime_params: Optional[dict] = None,
    **kwargs,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Mean-reversion v2 на z-score + фильтр по режиму.

    Возвращает:
      side   : Series float в {-1, 0, +1}
      z_score: Series float
      regime : Series str ('trend' / 'range' / 'high_vol')
    """
    if params is None:
        params = MeanRevV2Params(**kwargs)

    z = _zscore_meanrev_v2(df, params.ma_len, params.atr_len)
    regime = detect_regime_series(df, regime_params or {})

    side = pd.Series(0.0, index=df.index, dtype=float)

    thr = float(params.z_entry)
    filt = set(params.regime_filter)

    for i in range(len(df)):
        z_i = z.iloc[i]
        reg_i = regime.iloc[i]

        if not np.isfinite(z_i):
            continue
        if reg_i not in filt:
            continue

        if z_i <= -thr:
            side.iloc[i] = 1.0
        elif z_i >= thr:
            side.iloc[i] = -1.0
        # иначе остаётся 0.0

    return side, z, regime


def generate_breakout_signals(
    df: pd.DataFrame,
    params: Optional[BreakoutParams] = None,
    **kwargs,
) -> pd.Series:
    """Breakout-стратегия по каналам max/min за channel_len баров."""
    if params is None:
        params = BreakoutParams(**kwargs)

    high = df["high"].astype(float)
    low = df["low"].astype(float)

    hh = high.rolling(params.channel_len, min_periods=params.channel_len).max()
    ll = low.rolling(params.channel_len, min_periods=params.channel_len).min()

    side = pd.Series(0, index=df.index, dtype=float)
    current = 0
    bars_since_change = 10 ** 9
    confirm_counter = 0
    pending: Optional[int] = None  # 1 = long, -1 = short

    for i in range(len(df)):
        h = high.iloc[i]
        l = low.iloc[i]
        ch = hh.iloc[i]
        cl = ll.iloc[i]

        desired = current

        if np.isfinite(ch) and h > ch:
            # возможный пробой вверх
            if pending == 1:
                confirm_counter += 1
            else:
                pending = 1
                confirm_counter = 1
        elif np.isfinite(cl) and l < cl:
            # возможный пробой вниз
            if pending == -1:
                confirm_counter += 1
            else:
                pending = -1
                confirm_counter = 1
        else:
            pending = None
            confirm_counter = 0

        if pending is not None and confirm_counter >= params.confirm_bars:
            desired = pending

        if desired != current and bars_since_change < params.min_gap_bars:
            side.iloc[i] = float(current)
            bars_since_change += 1
            continue

        if desired != current:
            current = desired
            bars_since_change = 0
        else:
            bars_since_change += 1

        side.iloc[i] = float(current)

    return side