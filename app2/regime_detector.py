
import numpy as np
import pandas as pd

def detect_regime_series(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Detect market regime based on EMA, ATR, and quantile thresholds.
    Returns a Series with values: 'high_vol', 'trend', 'range'
    """
    high_vol_q = params.get("high_vol_quantile", 0.98)
    trend_thr = params.get("trend_threshold", 2.5)
    atr_len = params.get("atr_len", 14)
    ema_fast = params.get("ema_fast", 12)
    ema_slow = params.get("ema_slow", 48)

    close = df["close"]
    atr = df["high"].rolling(atr_len).max() - df["low"].rolling(atr_len).min()
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    norm_ema_diff = (ema_f - ema_s).abs() / (atr + 1e-8)
    atr_pct = atr / (close + 1e-8)

    vol_threshold = atr_pct.rolling(100).quantile(high_vol_q).shift(1)
    is_high_vol = atr_pct > vol_threshold
    is_trend = norm_ema_diff > trend_thr

    regime = np.where(is_high_vol, "high_vol",
              np.where(is_trend, "trend", "range"))
    return pd.Series(regime, index=df.index)

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
