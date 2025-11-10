# features.py
from __future__ import annotations
import numpy as np
import pandas as pd

FINAL_FEATURE_SET = [
    'price_efficiency','volume_anomaly','volatility_regime','liquidity_score',
    'momentum_5','momentum_10','trend_strength','high_low_vol',
    'dev','dev_mom','atr','realized_vol'
]

def _atr(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n//2)).mean()


def build_features_10m(df: pd.DataFrame) -> pd.DataFrame:
    # assumes columns: open, high, low, close, volume
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df.get("volume", pd.Series(1.0, index=df.index)).astype(float)

    # VWAP 15
    vwap = (close * vol).rolling(15, min_periods=1).sum() / vol.rolling(15, min_periods=1).sum()
    dev = (close - vwap) / vwap
    dev_mom = dev - dev.shift(2)

    # base indicators
    vol_ma10 = vol.rolling(10, min_periods=1).mean()
    atr = _atr(high, low, close, n=14)
    realized_vol = close.pct_change().rolling(60, min_periods=10).std().fillna(0.0)

    # price efficiency
    returns = np.log(close).diff().fillna(0.0)
    num = returns.rolling(8, min_periods=4).sum()
    den = returns.abs().rolling(8, min_periods=4).sum() + 1e-9
    price_efficiency = num / den

    # volume anomaly
    vol_ma = vol.rolling(8, min_periods=4).mean()
    vol_std = vol.rolling(8, min_periods=4).std()
    volume_anomaly = (vol - vol_ma) / (vol_std + 1e-9)

    # volatility regime
    volatility = returns.rolling(10, min_periods=6).std().fillna(0.01)
    volatility_regime = volatility / (volatility.mean() + 1e-9)

    # high-low vol
    high_low_vol = (high - low).rolling(10, min_periods=6).mean() / close

    # liquidity score
    liquidity_score = (close * vol) / (volatility + 1e-9)

    # momentum & trend
    momentum_5 = close.pct_change(5)
    momentum_10 = close.pct_change(10)
    trend_strength = (close - close.rolling(20, min_periods=10).mean()) / (
                close.rolling(20, min_periods=10).std() + 1e-9)

    feats = pd.DataFrame({
        "price_efficiency": price_efficiency,
        "volume_anomaly": volume_anomaly,
        "volatility_regime": volatility_regime,
        "liquidity_score": liquidity_score,
        "momentum_5": momentum_5,
        "momentum_10": momentum_10,
        "trend_strength": trend_strength,
        "high_low_vol": high_low_vol,
        "dev": dev,
        "dev_mom": dev_mom,
        "atr": atr,
        "realized_vol": realized_vol
    }, index=df.index)

    # Улучшенная обработка NaN и бесконечных значений
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # Заполняем NaN медианными значениями по колонкам
    for col in feats.columns:
        if feats[col].isna().any():
            feats[col] = feats[col].fillna(feats[col].median())

    # Winsorization (обрезание выбросов)
    for col in feats.columns:
        ql, qh = feats[col].quantile(0.05), feats[col].quantile(0.95)
        if np.isfinite(ql) and np.isfinite(qh) and ql < qh:
            feats[col] = feats[col].clip(ql, qh)

    feats = feats.astype("float32")
    return feats


def make_regime_labels(close: pd.Series) -> pd.Series:
    volatility = close.pct_change().rolling(20, min_periods=10).std().fillna(0.01)

    # Создаем метки, но заменяем NaN на значение по умолчанию (1 - средняя волатильность)
    labels = pd.cut(volatility, bins=[0, 0.004, 0.012, 1], labels=[0, 1, 2], include_lowest=True)

    # Заполняем оставшиеся NaN значения
    labels = labels.fillna(1)  # Заменяем NaN на 1 (средний режим)

    return labels.astype("int8")


def make_signal_labels(close: pd.Series, horizon: int = 6, thr: float = 0.003) -> pd.Series:
    future_returns = close.pct_change(horizon).shift(-horizon)

    # Создаем бинарные метки, но обрабатываем NaN
    lab = ((future_returns > thr) | (future_returns < -thr)).astype("int8")

    # Заполняем NaN значения (последние horizon баров будут NaN из-за shift)
    lab = lab.fillna(0).astype("int8")  # Заменяем NaN на 0 (нет сигнала)

    return lab