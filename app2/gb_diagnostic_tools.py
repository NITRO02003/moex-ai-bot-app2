# gb_diagnostic_tools.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app2 import data as D
from app2.model_improvement import create_improved_features


def analyze_feature_distribution(symbols: list):
    """Анализ распределения фич для диагностики"""
    print("📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ ФИЧ:")

    for symbol in symbols:
        print(f"\n🔍 {symbol}:")
        prices = D.load_csv(symbol)

        if prices.empty:
            print("   Нет данных")
            continue

        X = create_improved_features(prices)

        # Анализ основных статистик
        print(f"   Всего фич: {len(X.columns)}")
        print(f"   Баров: {len(X)}")

        # Проверяем наличие NaN и Inf
        nan_count = X.isna().sum().sum()
        inf_count = np.isinf(X.values).sum()

        print(f"   NaN значений: {nan_count}")
        print(f"   Inf значений: {inf_count}")

        # Статистики по предсказаниям
        if hasattr(analyze_feature_distribution, 'gb_model'):
            predictions = analyze_feature_distribution.gb_model.predict_proba(X.values)[:, 1]
            print(f"   Предсказания: min={predictions.min():.3f}, max={predictions.max():.3f}, "
                  f"mean={predictions.mean():.3f}")
            print(f"   >0.6: {(predictions > 0.6).mean():.1%}, "
                  f">0.7: {(predictions > 0.7).mean():.1%}")


def debug_signal_generation(symbol: str, threshold: float = 0.55):
    """Отладка генерации сигналов для конкретного тикера"""
    print(f"\n🐛 ОТЛАДКА ГЕНЕРАЦИИ СИГНАЛОВ: {symbol}")

    prices = D.load_csv(symbol)
    if prices.empty:
        print("   Нет данных")
        return

    from gb_backtest_system import GBBacktestSystem
    system = GBBacktestSystem()

    # Загружаем и обучаем модель
    prices_data = system.load_and_prepare_data([symbol], '2022-01-01', '2024-12-31')
    if not system.train_gb_model(prices_data):
        return

    X = create_improved_features(prices)
    predictions = system.gb_predict(X, prices['close'].astype(float))

    # Анализ предсказаний
    print(f"   Всего баров: {len(predictions)}")
    print(f"   Диапазон предсказаний: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"   Среднее: {predictions.mean():.3f}")

    # Анализ порогов
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    for th in thresholds:
        long_signals = (predictions > th).sum()
        short_signals = (predictions < (1 - th)).sum()
        total_signals = long_signals + short_signals

        print(f"   Порог {th}: {total_signals} сигналов "
              f"(L: {long_signals}, S: {short_signals}) - {total_signals / len(predictions):.1%}")


def compare_features_vs_predictions(symbol: str):
    """Сравнение фич с предсказаниями"""
    print(f"\n🔬 СРАВНЕНИЕ ФИЧ И ПРЕДСКАЗАНИЙ: {symbol}")

    prices = D.load_csv(symbol)
    if prices.empty:
        return

    from gb_backtest_system import GBBacktestSystem
    system = GBBacktestSystem()

    prices_data = system.load_and_prepare_data([symbol], '2022-01-01', '2024-12-31')
    system.train_gb_model(prices_data)

    X = create_improved_features(prices)
    predictions = system.gb_predict(X, prices['close'].astype(float))

    # Корреляция фич с предсказаниями
    correlations = []
    for col in X.columns:
        corr = np.corrcoef(X[col], predictions)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))

    # Сортируем по убыванию корреляции
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("   Топ-10 фич по корреляции с предсказаниями:")
    for feat, corr in correlations[:10]:
        print(f"      {feat}: {corr:.3f}")


if __name__ == "__main__":
    symbols = ['SBER', 'GAZP', 'GMKN', 'LKOH', 'ROSN', 'YNDX']

    # Запускаем диагностику
    analyze_feature_distribution(symbols)
    debug_signal_generation('SBER', 0.55)
    debug_signal_generation('GAZP', 0.55)
    compare_features_vs_predictions('SBER')