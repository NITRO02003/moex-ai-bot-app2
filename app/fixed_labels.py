# app/fixed_labels.py
import pandas as pd
import numpy as np


def make_better_labels(close: pd.Series, horizon: int = 6, volatility_threshold: float = 0.002) -> pd.Series:
    """
    Исправленные метки - предсказываем направление с учетом волатильности
    """
    # future returns
    future_returns = close.pct_change(horizon).shift(-horizon)

    # Адаптивный порог на основе волатильности
    volatility = close.pct_change().rolling(20).std().fillna(0.01)
    adaptive_threshold = volatility * 2  # 2 стандартных отклонения

    # Создаем три класса: -1 (sell), 0 (hold), 1 (buy)
    labels = pd.Series(0, index=close.index)  # по умолчанию hold

    # Buy сигналы: будущий рост больше порога
    buy_mask = future_returns > adaptive_threshold
    labels[buy_mask] = 1

    # Sell сигналы: будущее падение больше порога
    sell_mask = future_returns < -adaptive_threshold
    labels[sell_mask] = -1

    # Убедимся, что у нас сбалансированные классы
    print(f"Label distribution: {labels.value_counts().to_dict()}")

    return labels.astype("int8")


def test_label_quality():
    """Тестируем качество новых меток"""
    from .utils import load_all

    data = load_all("data", ["SBER"])
    close = data["SBER"]['close']

    old_labels = make_better_labels(close)
    print(f"Old labels: {old_labels.value_counts()}")

    # Проверяем информативность
    future_returns = close.pct_change(6).shift(-6)

    buy_returns = future_returns[old_labels == 1].mean()
    sell_returns = future_returns[old_labels == -1].mean()
    hold_returns = future_returns[old_labels == 0].mean()

    print(f"Buy returns: {buy_returns:.4f}")
    print(f"Sell returns: {sell_returns:.4f}")
    print(f"Hold returns: {hold_returns:.4f}")