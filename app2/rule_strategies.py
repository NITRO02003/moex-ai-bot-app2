from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd

from .features import _rsi  # используем твой RSI

Side = Literal[-1, 0, 1]


@dataclass
class TrendParams:
    ema_fast: int = 12
    ema_slow: int = 48
    atr_len: int = 14
    trend_thr: float = 0.5
    min_gap_bars: int = 5


@dataclass
class MeanRevParams:
    rsi_len: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    bb_len: int = 20
    bb_k: float = 2.0
    min_gap_bars: int = 5


@dataclass
class BreakoutParams:
    channel_len: int = 20
    confirm_bars: int = 1
    min_gap_bars: int = 5


def _ensure_ohlcv(px: pd.DataFrame) -> pd.DataFrame:
    required = {"close", "high", "low"}
    missing = required - set(px.columns)
    if missing:
        raise ValueError(f"prices dataframe missing columns: {missing}")
    return px


def _atr(px: pd.DataFrame, n: int) -> pd.Series:
    c = px["close"].astype(float)
    h = px["high"].astype(float)
    l = px["low"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            (h - l),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    return atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bollinger(close: pd.Series, n: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=1)
    up = ma + k * std
    dn = ma - k * std
    return ma, up, dn


def generate_trend_signals(prices: pd.DataFrame, p: TrendParams) -> pd.Series:
    px = _ensure_ohlcv(prices)
    c = px["close"].astype(float)

    ema_fast = c.ewm(span=p.ema_fast, adjust=False).mean()
    ema_slow = c.ewm(span=p.ema_slow, adjust=False).mean()
    atr = _atr(px, p.atr_len)

    spread = ema_fast - ema_slow
    trend_strength = spread / (atr + 1e-9)

    side = pd.Series(0, index=c.index, dtype="int8")

    raw_long = (trend_strength > p.trend_thr) & (c > ema_slow)
    raw_short = (trend_strength < -p.trend_thr) & (c < ema_slow)

    side[raw_long] = 1
    side[raw_short] = -1

    last_side: Side = 0  # type: ignore[assignment]
    last_change_i = -10**9
    idx = list(side.index)
    values = side.to_numpy()

    for i in range(len(values)):
        s = int(values[i])
        if s == last_side:
            continue
        if s != 0 and (i - last_change_i) < p.min_gap_bars:
            values[i] = last_side
            continue
        last_change_i = i
        last_side = s

    return pd.Series(values, index=idx, name="side_trend")


def generate_meanrev_signals(prices: pd.DataFrame, p: MeanRevParams) -> pd.Series:
    px = _ensure_ohlcv(prices)
    c = px["close"].astype(float)

    rsi = _rsi(c, p.rsi_len)
    bb_mid, bb_up, bb_dn = _bollinger(c, p.bb_len, p.bb_k)

    side = pd.Series(0, index=c.index, dtype="int8")

    raw_long = (rsi < p.rsi_low) & (c <= bb_dn)
    raw_short = (rsi > p.rsi_high) & (c >= bb_up)

    side[raw_long] = 1
    side[raw_short] = -1

    last_side: Side = 0  # type: ignore[assignment]
    last_change_i = -10**9
    idx = list(side.index)
    values = side.to_numpy()

    for i in range(len(values)):
        s = int(values[i])
        if s == last_side:
            continue
        if s != 0 and (i - last_change_i) < p.min_gap_bars:
            values[i] = last_side
            continue
        last_change_i = i
        last_side = s

    return pd.Series(values, index=idx, name="side_meanrev")


def generate_breakout_signals(prices: pd.DataFrame, p: BreakoutParams) -> pd.Series:
    px = _ensure_ohlcv(prices)
    c = px["close"].astype(float)
    h = px["high"].astype(float)
    l = px["low"].astype(float)

    hh = h.rolling(p.channel_len).max()
    ll = l.rolling(p.channel_len).min()

    side = pd.Series(0, index=c.index, dtype="int8")

    breakout_up = (c > hh.shift(1))
    breakout_dn = (c < ll.shift(1))

    side[breakout_up] = 1
    side[breakout_dn] = -1

    last_side: Side = 0  # type: ignore[assignment]
    last_change_i = -10**9
    idx = list(side.index)
    values = side.to_numpy()

    for i in range(len(values)):
        s = int(values[i])
        if s == last_side:
            continue
        if s != 0 and (i - last_change_i) < p.min_gap_bars:
            values[i] = last_side
            continue
        last_change_i = i
        last_side = s

    return pd.Series(values, index=idx, name="side_breakout")
