# app/data_optimizer.py
import pandas as pd
import numpy as np
from pathlib import Path


def optimize_data_quality():
    """Оптимизация качества данных для обучения"""
    print("=== DATA QUALITY OPTIMIZATION ===")

    data_dir = Path("data")
    symbols = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN"]

    recommendations = []

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

            print(f"\n{symbol} DATA ANALYSIS:")
            print(f"  Rows: {len(df)}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")

            # Анализ качества
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            trend = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)

            print(f"  Volatility: {volatility:.4f}")
            print(f"  Total trend: {trend:.2%}")

            # Проверка на стационарность
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(df['close'].dropna())
            is_stationary = adf_result[1] < 0.05
            print(f"  Stationary: {'✅' if is_stationary else '❌'} (p-value: {adf_result[1]:.4f})")

            if not is_stationary:
                recommendations.append(f"Внести стационарность в данные {symbol}")

            # Проверка автокорреляции
            autocorr = df['close'].autocorr(lag=1)
            print(f"  Autocorrelation: {autocorr:.4f}")

            if abs(autocorr) > 0.9:
                recommendations.append(f"Высокая автокорреляция у {symbol}")

    print(f"\nRECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  • {rec}")

    if not recommendations:
        print("  ✅ Данные в хорошем состоянии")


if __name__ == "__main__":
    optimize_data_quality()