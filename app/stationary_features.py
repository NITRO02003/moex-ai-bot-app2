# app/stationary_features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def make_features_stationary(features: pd.DataFrame) -> pd.DataFrame:
    """Преобразуем фичи к стационарным"""
    stationary = features.copy()

    for col in stationary.columns:
        # Пропускаем бинарные и уже стационарные фичи
        if stationary[col].nunique() <= 2:
            continue

        # Проверяем стационарность через разности
        original_autocorr = stationary[col].autocorr()

        # Применяем разности первого порядка
        diff1 = stationary[col].diff().dropna()
        diff1_autocorr = diff1.autocorr() if len(diff1) > 1 else 1

        # Если разность уменьшает автокорреляцию - используем ее
        if abs(diff1_autocorr) < abs(original_autocorr) * 0.8:
            stationary[col] = stationary[col].diff()

        # Логарифмическое преобразование для положительных фич
        if stationary[col].min() > 0:
            stationary[col] = np.log1p(stationary[col])

    # Заполняем NaN после дифференцирования
    stationary = stationary.ffill().bfill()

    return stationary


def calculate_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаем стационарные фичи с техническими индикаторами"""
    from .features import build_features_10m

    # Базовые фичи
    base_feats = build_features_10m(df)

    # Добавляем returns вместо абсолютных цен
    close = df['close']
    returns = close.pct_change().fillna(0)

    # Стационарные версии фич
    stationary_feats = make_features_stationary(base_feats)

    # Добавляем технические индикаторы на returns
    stationary_feats['returns'] = returns
    stationary_feats['returns_volatility'] = returns.rolling(20).std()
    stationary_feats['returns_skew'] = returns.rolling(30).skew()
    stationary_feats['returns_kurtosis'] = returns.rolling(30).kurt()

    # RSI на returns
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    stationary_feats['rsi'] = rsi(close, 14)

    return stationary_feats.dropna()


def test_stationarity():
    """Тест стационарности новых фич"""
    from .utils import load_all
    from statsmodels.tsa.stattools import adfuller

    data = load_all("data", ["SBER"])
    df = data["SBER"]

    # Старые фичи
    from .features import build_features_10m
    old_feats = build_features_10m(df)

    # Новые стационарные фичи
    new_feats = calculate_stationary_features(df)

    print("=== STATIONARITY TEST ===")
    for col in old_feats.columns[:3]:  # Тестируем первые 3 фичи
        if col in new_feats.columns:
            old_stat = adfuller(old_feats[col].dropna())[1]
            new_stat = adfuller(new_feats[col].dropna())[1]
            print(f"{col}: old p-value={old_stat:.4f}, new p-value={new_stat:.4f}")