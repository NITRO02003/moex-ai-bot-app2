from __future__ import annotations
import numpy as np, pandas as pd
from . import features as F, models as M, risk as R

def _trend_filter(close: pd.Series, fast: int = 20, slow: int = 60):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    slope = ema_fast.diff()
    up_trend = (ema_fast > ema_slow) & (slope > 0)
    down_trend = (ema_fast < ema_slow) & (slope < 0)
    return up_trend, down_trend

def _volume_filter(vol: pd.Series, q: float = 0.2):
    thr = vol.rolling(100).quantile(q)
    return (vol >= thr).fillna(False)

def signal_and_size(prices: pd.DataFrame, model_bundle, rp: R.RiskParams, equity: float,
                    threshold: float = 0.55, use_filters: bool = True) -> pd.DataFrame:
    X = F.build(prices)
    X = X[F.final_columns(X.columns)]
    close = prices['close'].astype(float)
    vol = prices['volume'].fillna(0)
    p = M.predict_proba(model_bundle, X, close)

    floor = R.conf_gate(p, close, rp)

    rv = close.pct_change().rolling(getattr(rp, 'vol_lb', 48)).std()
    reg_hi = rv > rv.rolling(getattr(rp, 'vol_lb', 48)).median()
    long_th = np.where(reg_hi, max(threshold, 0.60), max(threshold, 0.55))
    short_th = np.where(reg_hi, min(1 - threshold, 0.40), min(1 - threshold, 0.45))

    long_ok = (p >= long_th) & (p >= floor)
    short_ok = (p <= short_th) & ((1 - p) >= floor)

    if use_filters:
        up_tr, dn_tr = _trend_filter(close)
        vol_ok = _volume_filter(vol, q=0.2)
        long_ok = long_ok & up_tr & vol_ok
        short_ok = short_ok & dn_tr & vol_ok

    side = pd.Series(0, index=p.index, dtype='int8')
    side = side.where(~long_ok, 1)
    side = side.where(~short_ok, -1)

    size = R.position_size(close, p, equity, rp)
    out = pd.DataFrame({'p': p, 'side': side, 'size': size}).reindex(prices.index).fillna(0.0)
    return out
