# app/test_features.py
import pandas as pd
import os
from pathlib import Path
from app.features import build_features_10m


def test_features():
    data_dir = Path("data")
    symbols = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN"]

    for symbol in symbols:
        path = data_dir / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Убедимся, что индекс в UTC
            df.index = pd.to_datetime(df.index, utc=True)

            print(f"\n=== Testing features for {symbol} ===")
            print(f"Input data: {len(df)} rows")
            print(f"Date range: {df.index.min()} to {df.index.max()}")

            # Строим фичи
            features = build_features_10m(df)

            print(f"Output features: {len(features)} rows")
            print(f"Feature columns: {list(features.columns)}")
            print(f"Feature dtypes:\n{features.dtypes}")
            print(f"Sample features (first 3):")
            print(features.head(3))

            # Проверяем на NaN
            nan_count = features.isnull().sum().sum()
            print(f"NaN values: {nan_count}")

            if nan_count > 0:
                print("WARNING: Features contain NaN values!")
                # Показать, где именно NaN
                nan_columns = features.isnull().sum()
                nan_columns = nan_columns[nan_columns > 0]
                print(f"Columns with NaN: {nan_columns}")

        else:
            print(f"File not found: {symbol}")


if __name__ == "__main__":
    test_features()