"""Entry gating module for range-core backtester.

This module encapsulates feature preparation and AI-based gating logic
for trade entries. It separates the entry-model computations from the
backtester orchestrator, making the code easier to test and evolve.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..range_v3 import RangeV3Params
from ..features import add_basic_range_features
from ..feature_sweep import add_derived_features
from ..baseline_ml import COMPACT_FEATURES
from .blocks import calc_slope_mask


def _select_entry_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Select and normalize entry features from the input frame.

    Missing columns are filled with NaNs. Infinities are replaced
    with NaNs to avoid propagating them into the ML model.
    """
    work = df.replace([np.inf, -np.inf], np.nan).copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    return work[feature_cols]


def _build_entry_features(df_sig: pd.DataFrame) -> pd.DataFrame:
    """Build the full set of entry features.

    This applies basic range features and then derived feature
    transformations. Feature building is separated here so it can
    be reused and tested in isolation.
    """
    feats = add_basic_range_features(df_sig, ma_len=20, ma_mode="ema")
    feats = add_derived_features(feats)
    return feats


def _build_trend_mask(
    df_sig: pd.DataFrame,
    params: RangeV3Params,
    trend_slope_k: float,
) -> pd.Series:
    """Compute a boolean mask indicating when the trend-model gating applies.

    A non-positive slope_k disables the trend mask completely. Otherwise
    we call the slope-mask helper from blocks.py and compare the slope
    against the provided threshold.
    """
    if trend_slope_k <= 0:
        return pd.Series(False, index=df_sig.index)
    window = int(getattr(params, "range_window_bars", 20) or 20)
    slope_pct, _mask = calc_slope_mask(df_sig["close"], window, trend_slope_k)
    return (slope_pct >= trend_slope_k).fillna(False)


def _compute_threshold(
    prob_series: pd.Series,
    mask: pd.Series,
    mode: str,
    threshold: float,
    top_pct: float,
) -> float:
    """Compute a probability threshold based on mode and top_pct.

    In "top_pct" mode we compute a quantile of the probabilities on
    the masked bars. Otherwise we return the provided threshold.
    """
    if mode == "top_pct":
        pct = min(max(float(top_pct), 0.0), 1.0)
        if mask.any():
            return float(prob_series[mask].quantile(1.0 - pct))
        # if no masked bars, default to high threshold to block entries
        return 1.0
    return float(threshold)


def apply_entry_ai_filter(
    df_sig: pd.DataFrame,
    params: RangeV3Params,
    entry_model_path: str,
    entry_model_mode: str,
    entry_model_threshold: float,
    entry_model_top_pct: float,
    trend_model_path: str,
    trend_model_mode: str,
    trend_model_threshold: float,
    trend_model_top_pct: float,
    trend_slope_k: float,
    entry_feature_cols: Optional[List[str]] = None,
) -> Tuple[Optional[pd.Series], Dict[str, object]]:
    """Apply entry AI and trend-model gating to the signal frame.

    Parameters controlling gating mirror those in the CLI. A return
    value of ``None`` for the mask indicates gating is disabled.
    The stats dict contains diagnostic information about thresholds
    and counts of allowed/blocked bars.
    """
    # If both models are off or paths are missing, disable gating.
    if (
        (not entry_model_path or entry_model_mode == "off")
        and (not trend_model_path or trend_model_mode == "off")
    ):
        return None, {}
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except Exception as exc:
        raise SystemExit("CatBoost is required for entry-model gating") from exc

    feature_cols = entry_feature_cols or COMPACT_FEATURES
    feats = _build_entry_features(df_sig)
    x_all = _select_entry_features(feats, feature_cols)
    signal_mask = df_sig["v3_signal"].fillna(0).astype(int) != 0

    # Trend gating mask: only apply trend model on bars that satisfy slope condition
    trend_mask = (
        _build_trend_mask(df_sig, params, trend_slope_k)
        if trend_model_path and trend_model_mode != "off"
        else pd.Series(False, index=df_sig.index)
    )
    base_mask = ~trend_mask

    # Base (non-trend) gating
    allow_base = pd.Series(True, index=df_sig.index)
    base_threshold = None
    if entry_model_path and entry_model_mode != "off":
        model = CatBoostClassifier()
        model.load_model(entry_model_path)
        prob_base = pd.Series(model.predict_proba(x_all)[:, 1], index=df_sig.index)
        base_threshold = _compute_threshold(
            prob_base, signal_mask & base_mask, entry_model_mode, entry_model_threshold, entry_model_top_pct
        )
        allow_base = prob_base >= base_threshold

    # Trend gating
    allow_trend = pd.Series(True, index=df_sig.index)
    trend_threshold = None
    if trend_model_path and trend_model_mode != "off":
        model = CatBoostClassifier()
        model.load_model(trend_model_path)
        prob_trend = pd.Series(model.predict_proba(x_all)[:, 1], index=df_sig.index)
        trend_threshold = _compute_threshold(
            prob_trend, signal_mask & trend_mask, trend_model_mode, trend_model_threshold, trend_model_top_pct
        )
        allow_trend = prob_trend >= trend_threshold

    allow_mask = (trend_mask & allow_trend) | (base_mask & allow_base)
    stats = {
        "entry_ai_mode": entry_model_mode,
        "entry_ai_threshold": base_threshold,
        "entry_ai_top_pct": float(entry_model_top_pct),
        "entry_ai_trend_mode": trend_model_mode,
        "entry_ai_trend_threshold": trend_threshold,
        "entry_ai_trend_top_pct": float(trend_model_top_pct),
        "entry_ai_trend_slope_k": float(trend_slope_k),
        "entry_ai_signal_bars": int(signal_mask.sum()),
        "entry_ai_trend_bars": int(trend_mask.sum()),
        "entry_ai_allowed": int((allow_mask & signal_mask).sum()),
        "entry_ai_blocked": int((~allow_mask & signal_mask).sum()),
    }
    return allow_mask, stats