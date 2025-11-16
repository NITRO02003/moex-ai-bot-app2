"""Market regime detection utilities.

This module provides a simple rule‑based market regime classifier.  A
regime encapsulates broad conditions under which different trading
strategies may perform better: trending markets, ranging markets and
high volatility markets.  The detector looks at average true range
relative to price and the differential between fast and slow moving
averages to classify each bar.

Users can adjust thresholds via the :class:`RegimeParams` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd


@dataclass
class RegimeParams:
    """Parameters controlling market regime detection."""

    atr_len: int = 14
    ema_fast: int = 12
    ema_slow: int = 48
    # Require a higher percentile of volatility before classifying as high volatility.
    # Raising this quantile further to 0.95 reduces the frequency of high‑volatility
    # regimes so that trades occur only in the most turbulent conditions.
    high_vol_quantile: float = 0.95
    # Require a stronger divergence between fast and slow EMAs for a trend regime.
    # Increasing this threshold to 1.5 reduces whipsaws and only labels
    # truly trending conditions.  This is in units of ATR.
    trend_threshold: float = 1.5


Regime = Literal["trend", "range", "high_vol"]


def detect_regime(px: pd.DataFrame, params: RegimeParams | None = None) -> pd.Series:
    """Classify each bar into a market regime.

    The rules are:

    * **High volatility** if the ATR (relative to price) is in the top
      ``high_vol_quantile`` of its rolling distribution.
    * Else, **Trend** if the normalised EMA spread exceeds
      ``trend_threshold``.
    * Otherwise **Range**.

    Parameters
    ----------
    px : pandas.DataFrame
        Price dataframe with at least columns ``close``, ``high`` and ``low``.
    params : RegimeParams, optional
        Parameter dataclass controlling lookbacks and thresholds.

    Returns
    -------
    pandas.Series
        Series of regime labels with the same index as ``px``.
    """
    if params is None:
        params = RegimeParams()
    # Required columns
    c = px["close"].astype(float)
    h = px["high"].astype(float)
    l = px["low"].astype(float)
    # Compute ATR
    prev_c = c.shift(1).bfill()
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=params.atr_len, adjust=False).mean()
    # ATR relative to price
    atr_rel = atr / (c.replace(0.0, np.nan))
    atr_rel = atr_rel.fillna(0.0)
    # Rolling quantile threshold for high volatility
    # To avoid lookahead, use expanding quantile or fixed quantile on training sample
    q = atr_rel.rolling(window=200, min_periods=20).quantile(params.high_vol_quantile)
    high_vol = atr_rel >= q
    # Compute EMA spread normalised by ATR
    ema_fast = c.ewm(span=params.ema_fast, adjust=False).mean()
    ema_slow = c.ewm(span=params.ema_slow, adjust=False).mean()
    spread = ema_fast - ema_slow
    trend_strength = spread / (atr + 1e-9)
    trend = trend_strength.abs() >= params.trend_threshold
    # Determine regime
    regime = pd.Series("range", index=px.index, dtype=object)
    regime.loc[trend] = "trend"
    regime.loc[high_vol] = "high_vol"
    # In case both high_vol and trend, high_vol takes precedence
    both = high_vol & trend
    regime.loc[both] = "high_vol"
    return regime