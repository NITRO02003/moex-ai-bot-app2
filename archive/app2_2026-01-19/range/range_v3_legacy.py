import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# === Parameters ===


@dataclass
class RangeV3Params:
    """
    Lightweight wrapper over dict with typed accessors.

    Expects config like:
    {
        "RangeV3": {
            "params": {
                "range_window_bars": ...,
                ...
            }
        }
    }
    """

    cfg: Dict[str, Any]

    # ---- segmentation / geometry ----

    @property
    def range_window_bars(self) -> int:
        return int(self.cfg.get("range_window_bars", 50))

    @property
    def min_tests_AAA(self) -> int:
        return int(self.cfg.get("min_tests_AAA", 3))

    @property
    def min_tests_AA(self) -> int:
        return int(self.cfg.get("min_tests_AA", 2))

    @property
    def atr_low_q(self) -> float:
        return float(self.cfg.get("atr_low_q", 0.2))

    @property
    def atr_high_q(self) -> float:
        return float(self.cfg.get("atr_high_q", 0.8))

    @property
    def min_range_height_pct(self) -> float:
        return float(self.cfg.get("min_range_height_pct", 0.002))

    @property
    def max_range_height_pct(self) -> float:
        return float(self.cfg.get("max_range_height_pct", 0.02))

    @property
    def shadow_pct(self) -> float:
        return float(self.cfg.get("shadow_pct", 0.005))

    @property
    def entry_zone_alpha(self) -> float:
        return float(self.cfg.get("entry_zone_alpha", 0.2))

    # ---- trade / risk parameters ----

    @property
    def max_bars_in_trade(self) -> int:
        return int(self.cfg.get("max_bars_in_trade", 48))

    @property
    def risk_pct_per_trade(self) -> float:
        return float(self.cfg.get("risk_pct_per_trade", 0.007))

    @property
    def max_consecutive_losses(self) -> int:
        return int(self.cfg.get("max_consecutive_losses", 3))

    @property
    def daily_dd_limit_pct(self) -> float:
        return float(self.cfg.get("daily_dd_limit_pct", 0.02))

    # ---- breakout handling ----

    @property
    def breakout_bars(self) -> int:
        return int(self.cfg.get("breakout_bars", 2))

    @property
    def breakout_atr_q(self) -> float:
        return float(self.cfg.get("breakout_atr_q", 0.9))

    @property
    def lock_bars_after_breakout(self) -> int:
        return int(self.cfg.get("lock_bars_after_breakout", 20))

    # ---- deadzones ----

    @property
    def deadzone_min_atr_pct(self) -> float:
        return float(self.cfg.get("deadzone_min_atr_pct", 0.003))

    @property
    def deadzone_time_cfg(self) -> Dict[str, Any]:
        # example:
        # {
        #   "block_first_minutes": 30,
        #   "block_last_minutes": 30,
        #   "block_friday_after_hour": 15
        # }
        return self.cfg.get(
            "deadzone_time",
            {"block_first_minutes": 30, "block_last_minutes": 30, "block_friday_after_hour": 15},
        )


# === Segment structure ===


@dataclass
class RangeSegmentV3:
    start_idx: int
    end_idx: int
    quality: str  # "AAA" / "AA" / "A"


# === Internal helpers ===


def _calc_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def _calc_ma(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].rolling(window=window, min_periods=window).mean()


def _calc_ma_slope(ma: pd.Series, window: int) -> pd.Series:
    # simple difference over window
    return (ma - ma.shift(window)) / float(window)


def _in_deadzone_time(ts: pd.Timestamp, cfg: Dict[str, Any]) -> bool:
    """Simple intraday time-based deadzones."""
    first_min = int(cfg.get("block_first_minutes", 30))
    last_min = int(cfg.get("block_last_minutes", 30))
    fri_after = int(cfg.get("block_friday_after_hour", 15))

    minute_of_day = ts.hour * 60 + ts.minute
    if minute_of_day < first_min:
        return True
    if minute_of_day >= (24 * 60 - last_min):
        return True

    # Friday filter (weekday: Monday=0 .. Sunday=6)
    if ts.weekday() == 4 and ts.hour >= fri_after:
        return True

    return False


# === Segmentation ===


def detect_range_segments_v3(df: pd.DataFrame, params: RangeV3Params) -> List[RangeSegmentV3]:
    """
    Detects candidate range segments via ATR quantiles and MA slope.
    """
    if df.empty:
        return []

    window = params.range_window_bars
    if len(df) < window * 2:
        # not enough history
        return []

    atr = _calc_atr(df, window)
    ma = _calc_ma(df, window)
    slope = _calc_ma_slope(ma, window).abs()

    atr_q = atr.rank(pct=True)
    atr_ok = (atr_q >= params.atr_low_q) & (atr_q <= params.atr_high_q)

    close = df["close"]
    slope_threshold = close * 0.0005  # relative slope threshold, can be tuned later
    slope_ok = slope < slope_threshold

    base_mask = (atr_ok & slope_ok).fillna(False)

    segments_idx: List[Tuple[int, int]] = []
    in_seg = False
    seg_start: Optional[int] = None

    for i, flag in enumerate(base_mask):
        if flag and not in_seg:
            in_seg = True
            seg_start = i
        elif not flag and in_seg:
            if seg_start is not None and i - seg_start >= window:
                segments_idx.append((seg_start, i - 1))
            in_seg = False
            seg_start = None

    if in_seg and seg_start is not None and len(df) - seg_start >= window:
        segments_idx.append((seg_start, len(df) - 1))

    lows = df["low"]
    highs = df["high"]
    closes = df["close"]

    result: List[RangeSegmentV3] = []

    for start_idx, end_idx in segments_idx:
        seg_low = lows.iloc[start_idx : end_idx + 1]
        seg_high = highs.iloc[start_idx : end_idx + 1]
        seg_close = closes.iloc[start_idx : end_idx + 1]

        if seg_low.empty or seg_high.empty:
            continue

        L_raw = float(seg_low.quantile(0.15))
        U_raw = float(seg_high.quantile(0.85))
        H_raw = U_raw - L_raw
        if not math.isfinite(H_raw) or H_raw <= 0:
            continue

        mid_price = float(seg_close.mean())
        if not math.isfinite(mid_price) or mid_price <= 0:
            continue

        height_pct = H_raw / mid_price
        if height_pct < params.min_range_height_pct or height_pct > params.max_range_height_pct:
            continue

        lower_tests = (seg_low <= L_raw + 0.25 * H_raw).sum()
        upper_tests = (seg_high >= U_raw - 0.25 * H_raw).sum()
        tests = int(min(lower_tests, upper_tests))

        if tests >= params.min_tests_AAA:
            quality = "AAA"
        elif tests >= params.min_tests_AA:
            quality = "AA"
        else:
            quality = "A"

        result.append(RangeSegmentV3(start_idx=start_idx, end_idx=end_idx, quality=quality))

    return result


# === Box construction ===


def build_range_box_v3(
    df: pd.DataFrame, segment: RangeSegmentV3, params: RangeV3Params
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Compute L/U/H/M/shadow for given segment.
    """
    if df.empty:
        return None

    seg = df.iloc[segment.start_idx : segment.end_idx + 1]
    lows = seg["low"]
    highs = seg["high"]
    closes = seg["close"]

    if lows.empty or highs.empty:
        return None

    L = float(lows.quantile(0.15))
    U = float(highs.quantile(0.85))
    H = U - L
    if not math.isfinite(H) or H <= 0:
        return None

    mid_price = float(closes.mean())
    if not math.isfinite(mid_price) or mid_price <= 0:
        return None

    height_pct = H / mid_price
    if height_pct < params.min_range_height_pct or height_pct > params.max_range_height_pct:
        return None

    M = (L + U) / 2.0
    shadow = max(params.shadow_pct * mid_price, 0.0001)

    return L, U, H, M, shadow


# === Signal generation inside one segment ===


def generate_signals_v3_for_segment(
    df: pd.DataFrame,
    segment: RangeSegmentV3,
    box: Tuple[float, float, float, float, float],
    params: RangeV3Params,
) -> pd.DataFrame:
    """
    Generate v3 signals only inside given segment.
    """
    L, U, H, M, shadow = box
    entry_alpha = params.entry_zone_alpha

    out = df.copy()
    out["v3_signal"] = 0
    out["v3_segment_quality"] = None
    out["v3_L"] = np.nan
    out["v3_U"] = np.nan
    out["v3_M"] = np.nan
    out["v3_breakout"] = False

    atr = _calc_atr(df, params.range_window_bars)
    atr_pct = atr / df["close"]

    lower_zone_low = L - shadow
    lower_zone_high = L + entry_alpha * H
    upper_zone_low = U - entry_alpha * H
    upper_zone_high = U + shadow

    deadzone_atr_pct = params.deadzone_min_atr_pct
    deadzone_time_cfg = params.deadzone_time_cfg

    lower_touched = False
    upper_touched = False

    for i in range(segment.start_idx, segment.end_idx + 1):
        ts = df.index[i]
        price = float(df["close"].iloc[i])
        if not math.isfinite(price) or price <= 0:
            continue

        # ATR deadzone
        ap = float(atr_pct.iloc[i]) if i < len(atr_pct) and not math.isnan(atr_pct.iloc[i]) else None
        if ap is not None and ap < deadzone_atr_pct:
            continue

        # time deadzone
        if _in_deadzone_time(ts, deadzone_time_cfg):
            continue

        out.at[ts, "v3_segment_quality"] = segment.quality
        out.at[ts, "v3_L"] = L
        out.at[ts, "v3_U"] = U
        out.at[ts, "v3_M"] = M

        # breakout
        if price < lower_zone_low or price > upper_zone_high:
            out.at[ts, "v3_breakout"] = True
            continue

        # lower zone
        if lower_zone_low <= price <= lower_zone_high:
            if not lower_touched:
                lower_touched = True
                continue
            out.at[ts, "v3_signal"] = 1
            continue

        # upper zone
        if upper_zone_low <= price <= upper_zone_high:
            if not upper_touched:
                upper_touched = True
                continue
            out.at[ts, "v3_signal"] = -1
            continue

    return out


def apply_breakout_logic_v3(df: pd.DataFrame, params: RangeV3Params) -> pd.DataFrame:
    """
    Simple breakout filter: if we see a streak of v3_breakout bars,
    we suppress signals in that zone.
    """
    if "v3_breakout" not in df.columns:
        df["v3_breakout"] = False

    flag = df["v3_breakout"].astype(bool)
    if flag.empty:
        return df

    groups = (flag != flag.shift()).cumsum()
    streak = flag.groupby(groups).cumsum()
    mask = streak >= params.breakout_bars

    df.loc[mask, "v3_signal"] = 0
    return df


# === Top-level driver ===


def run_range_v3_offline_for_symbol(df: pd.DataFrame, params: RangeV3Params) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Offline/Labeler-реализация Range V3 для одного тикера.

    WARNING:
      Использует оффлайн-сегментацию и боксы по всему сегменту.
      Применять только для диагностики и разметки датасетов.

    Steps:
      1. Segment detection
      2. AAA/AA filtering
      3. Box construction for each segment
      4. Signal generation inside segments
      5. Breakout filtering
    """
    if df.empty:
        out = df.copy()
        out["v3_signal"] = 0
        out["v3_L"] = np.nan
        out["v3_U"] = np.nan
        out["v3_M"] = np.nan
        out["v3_segment_quality"] = None
        out["v3_breakout"] = False
        debug_info = {
            "segments_total": 0,
            "segments_tradable": 0,
            "segments_used": [],
        }
        return out, debug_info

    segments = detect_range_segments_v3(df, params)
    tradable_segments = [s for s in segments if s.quality in ("AAA", "AA")]

    combined = df.copy()
    combined["v3_signal"] = 0
    combined["v3_L"] = np.nan
    combined["v3_U"] = np.nan
    combined["v3_M"] = np.nan
    combined["v3_segment_quality"] = None
    combined["v3_breakout"] = False

    used_segments: List[Dict[str, Any]] = []

    for seg in tradable_segments:
        box = build_range_box_v3(df, seg, params)
        if box is None:
            continue

        local = generate_signals_v3_for_segment(df, seg, box, params)
        idx_slice = df.index[seg.start_idx : seg.end_idx + 1]

        for col in ["v3_signal", "v3_L", "v3_U", "v3_M", "v3_segment_quality", "v3_breakout"]:
            combined.loc[idx_slice, col] = local.loc[idx_slice, col]

        used_segments.append(
            {
                "start": int(seg.start_idx),
                "end": int(seg.end_idx),
                "quality": seg.quality,
                "L": float(box[0]),
                "U": float(box[1]),
                "H": float(box[2]),
                "M": float(box[3]),
            }
        )

    # if no usable segments – do not trade at all for this symbol
    if not used_segments:
        debug_info = {
            "segments_total": len(segments),
            "segments_tradable": len(tradable_segments),
            "segments_used": [],
        }
        return combined, debug_info

    combined = apply_breakout_logic_v3(combined, params)

    debug_info = {
        "segments_total": len(segments),
        "segments_tradable": len(tradable_segments),
        "segments_used": used_segments,
    }
    return combined, debug_info





def run_range_v3_for_symbol(df: pd.DataFrame, params: RangeV3Params) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

    # Валидный бокс: есть обе границы и адекватная высота
    mask_range = roll_L.notna() & roll_U.notna() & mask_height

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
        "signals_count": int(out["v3_signal"].sum()),
    }
    debug_info: Dict[str, Any] = {"rolling_diag": rolling_diag}

    return out, debug_info
