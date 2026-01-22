from __future__ import annotations

from typing import Tuple

import pandas as pd


def calc_roll_levels(
    low: pd.Series, high: pd.Series, window: int, min_valid: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    roll_L = low.rolling(window=window, min_periods=min_valid).quantile(0.15).shift(1)
    roll_U = high.rolling(window=window, min_periods=min_valid).quantile(0.85).shift(1)
    roll_H = roll_U - roll_L
    roll_M = (roll_L + roll_U) / 2.0
    return roll_L, roll_U, roll_H, roll_M


def calc_height_mask(
    roll_H: pd.Series, close: pd.Series, min_range_height_pct: float, max_range_height_pct: float
) -> Tuple[pd.Series, pd.Series]:
    height_pct = roll_H / close.shift(1)
    mask_height = (height_pct >= min_range_height_pct) & (height_pct <= max_range_height_pct)
    return height_pct, mask_height


def calc_slope_mask(
    close: pd.Series, window: int, slope_k: float
) -> Tuple[pd.Series, pd.Series]:
    min_valid = max(int(window * 0.4), 10)
    ma = close.rolling(window=window, min_periods=min_valid).mean()
    slope_raw = (ma - ma.shift(window)) / float(max(window, 1))
    slope_abs = slope_raw.abs().shift(1)
    slope_pct = slope_abs / close.shift(1)
    mask_slope = (slope_pct < slope_k).fillna(True)
    return slope_pct, mask_slope


def calc_atr_pct(
    high: pd.Series, low: pd.Series, close: pd.Series, atr_window: int, min_valid_atr: int
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=min_valid_atr).mean()
    atr_pct = (atr / close).shift(1)
    return atr_pct


def calc_vol_mask(atr_pct: pd.Series, atr_pct_min: float, atr_pct_max: float) -> pd.Series:
    return atr_pct.notna() & (atr_pct >= atr_pct_min) & (atr_pct <= atr_pct_max)


def calc_entry_zone(
    roll_L: pd.Series, roll_H: pd.Series, shadow_pct: float, entry_zone_alpha: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    shadow = roll_H * shadow_pct
    alpha = roll_H * entry_zone_alpha
    long_zone_low = roll_L - shadow
    long_zone_high = roll_L + alpha
    return long_zone_low, long_zone_high, shadow, alpha


def calc_breakout_mask(
    close: pd.Series, roll_L: pd.Series, shadow: pd.Series
) -> tuple[pd.Series, pd.Series]:
    breakout_level = roll_L - (shadow * 2.0)
    is_breakout = close < breakout_level
    return breakout_level, is_breakout


def calc_entry_signal(mask_range: pd.Series, is_in_zone: pd.Series, is_breakout: pd.Series) -> pd.Series:
    return mask_range & is_in_zone & (~is_breakout)


def calc_confirm_mask(is_in_zone: pd.Series, min_confirmations: int) -> pd.Series:
    if min_confirmations <= 1:
        return is_in_zone
    window = int(min_confirmations)
    return is_in_zone.rolling(window=window, min_periods=window).sum() >= window


def calc_recent_breakout(is_breakout: pd.Series, lock_bars: int) -> pd.Series:
    if lock_bars <= 0:
        return pd.Series(False, index=is_breakout.index)
    return is_breakout.rolling(window=lock_bars, min_periods=1).max().shift(1).fillna(False).astype(bool)


def calc_deadzone_mask(atr_pct: pd.Series, min_atr_pct: float) -> pd.Series:
    if min_atr_pct <= 0:
        return pd.Series(True, index=atr_pct.index)
    return atr_pct >= min_atr_pct
