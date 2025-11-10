# app/check_data_quality.py
import pandas as pd
from pathlib import Path


def check_data_quality():
    data_dir = Path("data")
    symbols = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN"]

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)

            print(f"\n=== {symbol} DATA QUALITY ===")
            print(f"Rows: {len(df)}")
            print(f"NaN values: {df.isna().sum().sum()}")
            print(f"Infinite values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")

            # Проверяем основные статистики
            print(
                f"Close stats: min={df['close'].min():.2f}, max={df['close'].max():.2f}, mean={df['close'].mean():.2f}")
            print(
                f"Volume stats: min={df['volume'].min():.0f}, max={df['volume'].max():.0f}, mean={df['volume'].mean():.0f}")

            # Проверяем волатильность (процентные изменения)
            returns = df['close'].pct_change().dropna()
            print(f"Returns stats: min={returns.min():.4f}, max={returns.max():.4f}, std={returns.std():.4f}")

            # Проверяем, есть ли аномальные значения
            if returns.abs().max() > 0.1:  # Если изменение больше 10%
                print(f"WARNING: Large price movement detected: {returns.abs().max():.4f}")


if __name__ == "__main__":
    check_data_quality()