import pandas as pd
from app2 import data as D


def check_data_quality():
    """Проверка качества загруженных данных"""
    print("=== ПРОВЕРКА КАЧЕСТВА ДАННЫХ ===")

    symbols = ['SBER', 'GAZP', 'LKOH', 'GMKN', 'ROSN']

    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        prices = D.load_csv(symbol)

        if prices.empty:
            print("   Нет данных")
            continue

        close = prices['close'].astype(float)

        print(f"   Период: {len(prices)} баров")
        print(f"   Дата начала: {prices.index[0]}")
        print(f"   Дата окончания: {prices.index[-1]}")
        print(f"   Цена закрытия: от {close.min():.2f} до {close.max():.2f}")
        print(f"   Средняя цена: {close.mean():.2f}")

        # Проверяем пропуски
        missing_data = close.isna().sum()
        print(f"   Пропущенных данных: {missing_data}")

        # Проверяем аномалии
        returns = close.pct_change()
        extreme_returns = (returns.abs() > 0.1).sum()  # Более 10% за бар
        print(f"   Экстремальных движений (>10%): {extreme_returns}")

        # Статистика доходностей
        if len(returns) > 1:
            print(f"   Средняя дневная доходность: {returns.mean() * 100:.4f}%")
            print(f"   Волатильность: {returns.std() * 100:.4f}%")


if __name__ == "__main__":
    check_data_quality()