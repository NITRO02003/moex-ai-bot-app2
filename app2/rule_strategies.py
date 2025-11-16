"""Rule‑based trading strategies.

This module implements a few simple deterministic signal generators that
operate on OHLC dataframes.  Each strategy consumes a ``pandas.DataFrame``
containing at least the columns ``close``, ``high`` and ``low`` and
produces a ``pandas.Series`` of trade directions: ``+1`` for long,
``-1`` for short and ``0`` for flat.  These raw side series can then
be passed into the backtest engine in :mod:`app2.rule_backtest`.

The strategies provided here are intentionally straightforward and
parameterised via dataclasses.  They serve as a baseline for more
complex, regime‑aware systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_ohlc(px: pd.DataFrame) -> None:
    """Validate that required OHLC columns are present.

    Raises
    ------
    KeyError
        If any of ``close``, ``high`` or ``low`` are missing from the
        input dataframe.
    """
    required = {"close", "high", "low"}
    missing = required - set(px.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")


@dataclass
class TrendParams:
    """Parameter bundle for the trend following strategy.

    The defaults here have been tuned conservatively to reduce noise.
    A higher ``trend_thr`` requires a larger spread between the fast
    and slow EMAs (relative to ATR) before a trend is declared. A
    larger ``min_gap_bars`` prevents frequent flipping of the
    position. Adjust these parameters to control the aggressiveness
    of the trend strategy.
    """

    # Length of the fast EMA used to gauge short‑term trend.
    ema_fast: int = 12
    # Length of the slow EMA used to gauge long‑term trend.
    ema_slow: int = 48
    # Lookback for ATR calculation in bars.
    atr_len: int = 14
    # Normalised trend threshold (spread/ATR) to trigger a position.
    # Increased to 1.5 by default to require even stronger
    # momentum before entering a trade.  This helps to filter out
    # weaker moves and reduce the number of trades.
    trend_thr: float = 2.5
    # Minimum number of bars between signal flips.  Set to 15 by
    # default to further reduce churn and limit over‑trading.
    min_gap_bars: int = 25


@dataclass
class MeanRevParams:
    """Parameter bundle for the mean reversion strategy.

    The oversold/overbought thresholds are deliberately wide to
    capture only the most extreme deviations. The ``min_gap_bars``
    parameter has been increased to cut down on high‑frequency flips.
    """

    rsi_len: int = 14
    # RSI threshold below which a long entry may be considered.
    rsi_low: float = 15.0
    # RSI threshold above which a short entry may be considered.
    rsi_high: float = 85.0
    # Lookback for the Bollinger bands moving average.
    bb_len: int = 20
    # Width of the Bollinger bands in standard deviations.
    bb_k: float = 2.0
    # Minimum number of bars between signal flips.  Increased to 15
    # to further reduce the number of mean‑reversion trades and cut
    # down on commissions.
    min_gap_bars: int = 15


@dataclass
class BreakoutParams:
    """Parameter bundle for the breakout strategy.

    Breakout strategies can generate a large number of trades if not
    properly throttled.  The ``min_gap_bars`` default has been
    increased to 10 to avoid whipsawing around the channel extremes.
    """

    # Number of bars to compute the rolling high/low channel.
    channel_len: int = 30
    # Bars required to confirm a breakout.  A value of 1 means no
    # confirmation – trades are taken on the first break.
    confirm_bars: int = 2
    # Minimum number of bars between flips.  Increased to 15 by default
    # to curb frequent breakout attempts and lower trading frequency.
    min_gap_bars: int = 15


def generate_trend_signals(px: pd.DataFrame, params: Optional[TrendParams] = None) -> pd.Series:
    """Generate trend following signals.

    This strategy looks for sustained moves where a fast EMA diverges from
    a slow EMA by more than a multiple of the Average True Range (ATR).  It
    returns ``+1`` when the trend is deemed bullish, ``-1`` when it is
    bearish and ``0`` otherwise.  Rapid flips are suppressed by enforcing
    a minimum number of bars between signal changes.

    Parameters
    ----------
    px : pandas.DataFrame
        Price dataframe with columns ``close``, ``high`` and ``low``.
    params : TrendParams, optional
        Parameter dataclass controlling the lookback lengths and
        thresholds.  Default values will be used if omitted.

    Returns
    -------
    pandas.Series
        A series of ``int8`` values aligned to the index of ``px``.
    """
    if params is None:
        params = TrendParams()
    _ensure_ohlc(px)
    c = px["close"].astype(float)
    h = px["high"].astype(float)
    l = px["low"].astype(float)
    # Compute EMAs
    ema_fast = c.ewm(span=params.ema_fast, adjust=False).mean()
    ema_slow = c.ewm(span=params.ema_slow, adjust=False).mean()
    # True range for ATR
    prev_c = c.shift(1).bfill()
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=params.atr_len, adjust=False).mean()
    # Trend strength normalised by ATR
    spread = ema_fast - ema_slow
    trend_strength = spread / (atr + 1e-9)
    side = pd.Series(0, index=px.index, dtype="int8")
    last_dir = 0
    bars_since_flip = 0
    for i in range(len(side)):
        if bars_since_flip > 0:
            bars_since_flip -= 1
        ts = trend_strength.iat[i]
        if ts > params.trend_thr and c.iat[i] > ema_slow.iat[i]:
            new_dir = 1
        elif ts < -params.trend_thr and c.iat[i] < ema_slow.iat[i]:
            new_dir = -1
        else:
            new_dir = 0
        if new_dir != last_dir and bars_since_flip > 0:
            new_dir = last_dir
        if new_dir != last_dir:
            bars_since_flip = params.min_gap_bars
            last_dir = new_dir
        side.iat[i] = last_dir
    return side


def generate_meanrev_signals(px: pd.DataFrame, params: Optional[MeanRevParams] = None) -> pd.Series:
    """Generate mean reversion signals based on RSI and Bollinger bands.

    A long signal is issued when the price closes below the lower Bollinger
    band and RSI is in oversold territory.  A short signal is issued
    when the price closes above the upper band and RSI is overbought.
    Like the trend strategy, rapid flips are suppressed.

    Parameters
    ----------
    px : pandas.DataFrame
        Price dataframe with columns ``close``, ``high`` and ``low``.
    params : MeanRevParams, optional
        Parameter dataclass controlling RSI and Bollinger parameters.

    Returns
    -------
    pandas.Series
        A series of ``int8`` values representing trade directions.
    """
    if params is None:
        params = MeanRevParams()
    _ensure_ohlc(px)
    c = px["close"].astype(float)
    # RSI
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / params.rsi_len, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / params.rsi_len, adjust=False).mean()
    rs = up / (dn + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    # Bollinger bands
    ma = c.rolling(params.bb_len).mean()
    sd = c.rolling(params.bb_len).std(ddof=0)
    bb_up = ma + params.bb_k * sd
    bb_dn = ma - params.bb_k * sd
    side = pd.Series(0, index=px.index, dtype="int8")
    last_dir = 0
    bars_since_flip = 0
    for i in range(len(side)):
        if bars_since_flip > 0:
            bars_since_flip -= 1
        new_dir = 0
        # Long criteria
        if rsi.iat[i] < params.rsi_low and c.iat[i] <= bb_dn.iat[i]:
            new_dir = 1
        # Short criteria
        elif rsi.iat[i] > params.rsi_high and c.iat[i] >= bb_up.iat[i]:
            new_dir = -1
        if new_dir != last_dir and bars_since_flip > 0:
            new_dir = last_dir
        if new_dir != last_dir:
            bars_since_flip = params.min_gap_bars
            last_dir = new_dir
        side.iat[i] = last_dir
    return side


def generate_breakout_signals(px: pd.DataFrame, params: Optional[BreakoutParams] = None) -> pd.Series:
    """Generate breakout signals based on price channel breakout.

    A breakout occurs when the close crosses above the prior bar's
    upper channel (rolling high) or below the prior bar's lower channel
    (rolling low).  To reduce false signals, a minimum gap of bars is
    enforced between reversals.

    Parameters
    ----------
    px : pandas.DataFrame
        Price dataframe with columns ``close``, ``high`` and ``low``.
    params : BreakoutParams, optional
        Parameter dataclass controlling channel length and min gap.

    Returns
    -------
    pandas.Series
        A series of trade directions (int8).
    """
    if params is None:
        params = BreakoutParams()
    _ensure_ohlc(px)
    c = px["close"].astype(float)
    h = px["high"].astype(float)
    l = px["low"].astype(float)
    hh = h.rolling(params.channel_len).max()
    ll = l.rolling(params.channel_len).min()
    side = pd.Series(0, index=px.index, dtype="int8")
    last_dir = 0
    bars_since_flip = 0
    for i in range(len(side)):
        if bars_since_flip > 0:
            bars_since_flip -= 1
        new_dir = 0
        # Breakout up
        if c.iat[i] > hh.shift(1).iat[i]:
            new_dir = 1
        # Breakout down
        elif c.iat[i] < ll.shift(1).iat[i]:
            new_dir = -1
        if new_dir != last_dir and bars_since_flip > 0:
            new_dir = last_dir
        if new_dir != last_dir:
            bars_since_flip = params.min_gap_bars
            last_dir = new_dir
        side.iat[i] = last_dir
    return side
