
from __future__ import annotations
import numpy as np, pandas as pd
try:
    from app.features import build_features_15m as _user_build
    from app.features import FINAL_FEATURE_SET as _USER_SET
except Exception:
    _user_build = None
    _USER_SET = None

def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up/(dn+1e-9)
    return 100 - 100/(1+rs)

def _basic(px: pd.DataFrame) -> pd.DataFrame:
    c = px['close'].astype(float); h = px['high'].astype(float); l = px['low'].astype(float)
    v = px.get('volume', pd.Series(index=c.index, dtype=float)).fillna(0.0)
    f = pd.DataFrame(index=c.index)
    for w in (3,6,12,24,48,96):
        f[f'ret_{w}'] = c.pct_change(w)
        f[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()/c - 1.0
        f[f'atr_{w}'] = (h - l).ewm(span=w, adjust=False).mean() / c.replace(0,np.nan)
        f[f'vol_ema_{w}'] = v.ewm(span=w, adjust=False).mean()
    f['rsi_14'] = _rsi(c,14)
    f['zscore_24'] = (c - c.rolling(24).mean())/(c.rolling(24).std(ddof=1)+1e-9)
    return f.replace([np.inf,-np.inf], np.nan).fillna(0.0).astype('float32')

def build(px: pd.DataFrame) -> pd.DataFrame:
    base = _basic(px)
    if _user_build is not None:
        try:
            usr = _user_build(px)
            base = pd.concat([base, usr], axis=1)
        except Exception:
            pass
    base = base.loc[~base.index.duplicated(keep='last')].sort_index()
    return base.replace([np.inf,-np.inf], np.nan).fillna(0.0).astype('float32')

def final_columns(cols):
    if _USER_SET is not None and len(_USER_SET)>0:
        rest = [c for c in cols if c not in _USER_SET]
        return list(_USER_SET) + rest
    return list(cols)
