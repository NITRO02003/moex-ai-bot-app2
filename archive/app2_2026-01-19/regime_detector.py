
import numpy as np
import pandas as pd
from .rule_core import _compute_atr

def detect_regime_series(df: pd.DataFrame, params: dict) -> pd.Series:
    """Detect market regime based on ATR and EMA spread.

    Returns a Series with values: 'high_vol', 'trend', 'range'.
    """
    high_vol_q = float(params.get("high_vol_quantile", 0.98))
    trend_thr = float(params.get("trend_threshold", 2.5))
    atr_len = int(params.get("atr_len", 14))
    ema_fast = int(params.get("ema_fast", 12))
    ema_slow = int(params.get("ema_slow", 48))

    if df.empty:
        return pd.Series([], index=df.index, dtype=object)

    df = df.copy()

    close = df["close"].astype(float)

    # ATR and normalized volatility
    try:
        atr = _compute_atr(df, atr_len)
    except Exception:
        # fallback: no high/low columns, treat as zero ATR
        atr = pd.Series(0.0, index=df.index)

    atr = atr.astype(float).replace([np.inf, -np.inf], np.nan)
    atr_pct = (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    atr_pct = atr_pct.fillna(0.0)

    # high_vol by global quantile of atr_pct
    if atr_pct.notna().any():
        try:
            vol_threshold = float(atr_pct.quantile(high_vol_q))
        except Exception:
            vol_threshold = float(atr_pct.quantile(0.98))
    else:
        vol_threshold = float("inf")

    is_high_vol = atr_pct >= vol_threshold

    # EMA-based trend score (spread normalized by ATR)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    atr_safe = atr.copy()
    atr_safe.replace(0.0, np.nan, inplace=True)
    trend_score = (ema_f - ema_s) / atr_safe
    trend_score = trend_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    is_trend = trend_score.abs() >= trend_thr

    regime = np.full(len(df), "range", dtype=object)

    hv_mask = is_high_vol.to_numpy()
    regime[hv_mask] = "high_vol"

    tr_mask = (~hv_mask) & is_trend.to_numpy()
    regime[tr_mask] = "trend"

    return pd.Series(regime, index=df.index)


def build_regime_segments(df: pd.DataFrame, regime_series: pd.Series) -> pd.DataFrame:
    """Build contiguous segments of the same regime.

    Returns DataFrame with columns: regime, start_dt, end_dt, bars.
    """
    if df.empty or regime_series.empty:
        return pd.DataFrame(columns=["regime", "start_dt", "end_dt", "bars"])

    if "datetime" in df.columns:
        t = pd.to_datetime(df["datetime"])
    elif "begin" in df.columns:
        t = pd.to_datetime(df["begin"])
    else:
        t = pd.to_datetime(df.index)

    regimes = regime_series.to_numpy()
    n = len(regimes)
    segments = []

    current_regime = regimes[0]
    start_idx = 0

    for i in range(1, n):
        if regimes[i] != current_regime:
            start_dt = t.iloc[start_idx]
            end_dt = t.iloc[i - 1]
            segments.append(
                {
                    "regime": current_regime,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "bars": int(i - start_idx),
                }
            )
            current_regime = regimes[i]
            start_idx = i

    start_dt = t.iloc[start_idx]
    end_dt = t.iloc[n - 1]
    segments.append(
        {
            "regime": current_regime,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "bars": int(n - start_idx),
        }
    )

    return pd.DataFrame(segments)

def regime_distribution(regime_series: pd.Series) -> dict:
    total = len(regime_series)
    dist = regime_series.value_counts(normalize=True)
    return {
        "trend": round(100 * dist.get("trend", 0.0), 2),
        "range": round(100 * dist.get("range", 0.0), 2),
        "high_vol": round(100 * dist.get("high_vol", 0.0), 2)
    }

def detect_regime(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    regime_series = detect_regime_series(df, params)
    df = df.copy()
    df["regime"] = regime_series
    return df