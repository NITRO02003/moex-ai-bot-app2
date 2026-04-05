"""Snapshot analytics helpers for Range core.

This module computes causal per-bar analytics used at trade entry time.
These features do not affect trading logic directly; they are used for
diagnostics, dataset construction and optional ML gating.

The goal is to centralize calculations of geometry, volatility, slope and
relative distances to range boundaries, decoupling them from the
execution loop in the backtester.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..range_v3 import RangeV3Params
from .geometry import compute_geometry
from .blocks import calc_atr_pct, calc_slope_mask


def build_entry_snapshot_frame(df_sig: pd.DataFrame, params: RangeV3Params) -> pd.DataFrame:
    """Compute per-bar snapshot analytics for entry decisions.

    Parameters
    ----------
    df_sig : pd.DataFrame
        The signal DataFrame returned by ``run_core_for_symbol``.  It should
        contain at least columns ``open``, ``high``, ``low``, ``close`` and
        optionally ``v3_L``/``v3_U`` with rolling range levels.
    params : RangeV3Params
        The range configuration object providing ATR window and slope
        parameters.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed the same as ``df_sig`` with the following
        columns:

        - ``geo_class``: simple label of the current box ("V3" if both L and U
          are defined and the height is positive, otherwise "INVALID").
        - ``geo_valid_box``: boolean flag indicating a valid box.
        - ``geo_h_pct``: relative height of the box (range height divided by
          absolute close price).
        - ``atr_pct``: causal ATR percentage computed on ``high``, ``low`` and
          ``close`` columns.
        - ``slope_pct_per_bar``: absolute slope of the close price MA per bar
          normalised by price.
        - ``dist_L_pct``: normalised distance of the close price to the lower
          range boundary: ``(close - L) / (U - L)``.
        - ``dist_U_pct``: normalised distance of the close price to the upper
          range boundary: ``(U - close) / (U - L)``.

    Notes
    -----
    - All computations are causal: only past and present data are used.
    - If necessary columns are missing, the corresponding output will
      contain NaN.
    - This function does not modify ``df_sig``; it returns a separate
      DataFrame with analytics.
    """
    idx = df_sig.index
    # Determine range levels (L and U).  Prefer the v3 columns if present;
    # otherwise fall back to computing geometry on the input OHLCV data.
    if "v3_L" in df_sig.columns and "v3_U" in df_sig.columns:
        L = df_sig["v3_L"].astype(float)
        U = df_sig["v3_U"].astype(float)
    else:
        # compute_geometry returns geo_L and geo_U columns
        geo_df = compute_geometry(df_sig, params)
        L = geo_df.get("geo_L", pd.Series(np.nan, index=idx)).astype(float)
        U = geo_df.get("geo_U", pd.Series(np.nan, index=idx)).astype(float)

    H = U - L
    valid_mask = H.notna() & (H > 0)
    # Simple class: mark valid boxes as "V3", invalid as "INVALID"
    geo_class = pd.Series("INVALID", index=idx, dtype="object")
    geo_class.loc[valid_mask] = "V3"
    geo_valid_box = valid_mask.fillna(False)
    # Relative height: protect against division by zero by replacing zero
    # absolute close price with NaN
    close_abs = df_sig.get("close", pd.Series(np.nan, index=idx)).abs().replace(0, np.nan)
    geo_h_pct = H / close_abs

    # ATR percentage: robust rolling ATR over high/low/close
    atr_window = int(getattr(params, "atr_window", 14) or 14)
    # Determine minimum valid bars for ATR.  Use params attributes if present,
    # otherwise default to half the window.
    min_valid_frac = float(getattr(params, "atr_min_valid_frac", 0.5) or 0.5)
    min_valid_bars = int(getattr(params, "atr_min_valid_bars", max(int(atr_window * min_valid_frac), 1)) or 0)
    min_valid_atr = max(int(atr_window * min_valid_frac), min_valid_bars)
    if all(col in df_sig.columns for col in ["high", "low", "close"]):
        atr_pct = calc_atr_pct(
            df_sig["high"].astype(float),
            df_sig["low"].astype(float),
            df_sig["close"].astype(float),
            atr_window,
            min_valid_atr,
        )
    else:
        atr_pct = pd.Series(np.nan, index=idx)

    # Slope of price MA per bar normalised by price.
    window = int(getattr(params, "range_window_bars", 20) or 20)
    slope_k = float(getattr(params, "slope_k", 0.0) or 0.0)
    # Use close series if present; otherwise slope remains NaN.
    if "close" in df_sig.columns:
        slope_pct, _ = calc_slope_mask(df_sig["close"].astype(float), window, slope_k)
    else:
        slope_pct = pd.Series(np.nan, index=idx)
    slope_pct_per_bar = slope_pct

    # Distances from the price to range boundaries.
    dist_L_pct = pd.Series(np.nan, index=idx, dtype="float64")
    dist_U_pct = pd.Series(np.nan, index=idx, dtype="float64")
    # Only compute when both L and U are defined and H > 0
    valid_idx = valid_mask[valid_mask].index
    if not valid_idx.empty and "close" in df_sig.columns:
        close_vals = df_sig["close"].astype(float)
        # Compute vectorised distances
        h_vals = H[valid_idx].astype(float)
        l_vals = L[valid_idx].astype(float)
        u_vals = U[valid_idx].astype(float)
        c_vals = close_vals[valid_idx]
        dist_L_pct.loc[valid_idx] = (c_vals - l_vals) / h_vals
        dist_U_pct.loc[valid_idx] = (u_vals - c_vals) / h_vals

    return pd.DataFrame(
        {
            "geo_class": geo_class,
            "geo_valid_box": geo_valid_box,
            "geo_h_pct": geo_h_pct.astype(float),
            "atr_pct": atr_pct.astype(float),
            "slope_pct_per_bar": slope_pct_per_bar.astype(float),
            "dist_L_pct": dist_L_pct.astype(float),
            "dist_U_pct": dist_U_pct.astype(float),
        },
        index=idx,
    )
