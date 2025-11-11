# features.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
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
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn + 1e-9)
    return 100 - 100 / (1 + rs)


def _basic(px: pd.DataFrame) -> pd.DataFrame:
    c = px['close'].astype(float);
    h = px['high'].astype(float);
    l = px['low'].astype(float)
    v = px.get('volume', pd.Series(index=c.index, dtype=float)).fillna(0.0)
    f = pd.DataFrame(index=c.index)
    for w in (3, 6, 12, 24, 48, 96):
        f[f'ret_{w}'] = c.pct_change(w)
        f[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean() / c - 1.0
        f[f'atr_{w}'] = (h - l).ewm(span=w, adjust=False).mean() / c.replace(0, np.nan)
        f[f'vol_ema_{w}'] = v.ewm(span=w, adjust=False).mean()
    f['rsi_14'] = _rsi(c, 14)
    f['zscore_24'] = (c - c.rolling(24).mean()) / (c.rolling(24).std(ddof=1) + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype('float32')


# РЕАЛИЗОВАННЫЕ ФУНКЦИИ ИНДИКАТОРОВ
def _calculate_adx(high, low, close, period):
    """Average Directional Index - РЕАЛИЗОВАНО"""
    try:
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low

        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)

        # Smoothed values
        tr_smooth = tr.rolling(period).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=high.index).rolling(period).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=high.index).rolling(period).mean()

        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(period).mean()

        return adx.fillna(0)
    except Exception:
        return pd.Series(0, index=high.index)


def _calculate_macd(close):
    """MACD - РЕАЛИЗОВАНО"""
    try:
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        return (ema12 - ema26).rename('macd')
    except Exception:
        return pd.Series(0, index=close.index)


def _calculate_stochastic(close, high, low, period):
    """Stochastic %K - РЕАЛИЗОВАНО"""
    try:
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    except Exception:
        return pd.Series(0, index=close.index)


def _calculate_cci(close, high, low, period):
    """Commodity Channel Index - РЕАЛИЗОВАНО"""
    try:
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=False
        )
        cci = (typical_price - sma) / (0.015 * mad + 1e-9)
        return cci.fillna(0)
    except Exception:
        return pd.Series(0, index=close.index)


def _calculate_williams_r(close, high, low, period):
    """Williams %R - РЕАЛИЗОВАНО"""
    try:
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
        return williams_r.fillna(0)
    except Exception:
        return pd.Series(0, index=close.index)


def _calculate_obv(close, volume):
    """On Balance Volume - РЕАЛИЗОВАНО"""
    try:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    except Exception:
        return pd.Series(0, index=close.index)


def _calculate_vwap(close, volume):
    """Volume Weighted Average Price - РЕАЛИЗОВАНО"""
    try:
        typical_price = (close + close + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap.fillna(close)
    except Exception:
        return close


def _momentum_features(px: pd.DataFrame) -> pd.DataFrame:
    """Дополнительные фичи момента - ОБНОВЛЕНО С РЕАЛИЗАЦИЕЙ"""
    c = px['close'].astype(float)
    h = px['high'].astype(float)
    l = px['low'].astype(float)
    v = px.get('volume', pd.Series(index=c.index, dtype=float)).fillna(0.0)

    feat = pd.DataFrame(index=c.index)

    # МОМЕНТУМ ИНДИКАТОРЫ
    feat['adx_14'] = _calculate_adx(h, l, c, 14)
    feat['macd'] = _calculate_macd(c)
    feat['stoch_k'] = _calculate_stochastic(c, h, l, 14)

    # ОСЦИЛЛЯТОРЫ
    feat['cci_20'] = _calculate_cci(c, h, l, 20)
    feat['williams_r'] = _calculate_williams_r(c, h, l, 14)

    # ОБЪЕМНЫЕ ИНДИКАТОРЫ
    feat['obv'] = _calculate_obv(c, v)
    feat['vwap'] = _calculate_vwap(c, v)

    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype('float32')


def build(px: pd.DataFrame) -> pd.DataFrame:
    base = _basic(px)
    momentum = _momentum_features(px)  # ТЕПЕРЬ РАБОТАЕТ
    base = pd.concat([base, momentum], axis=1)

    if _user_build is not None:
        try:
            usr = _user_build(px)
            base = pd.concat([base, usr], axis=1)
        except Exception:
            pass

    base = base.loc[~base.index.duplicated(keep='last')].sort_index()
    return base.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype('float32')


def final_columns(cols):
    if _USER_SET is not None and len(_USER_SET) > 0:
        rest = [c for c in cols if c not in _USER_SET]
        return list(_USER_SET) + rest
    return list(cols)