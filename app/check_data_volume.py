# app/check_data_volume.py
import pandas as pd
from pathlib import Path


def check_data_volume():
    data_dir = Path("data")
    symbols = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN"]

    print("=== DATA VOLUME CHECK ===")

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            days = (df.index.max() - df.index.min()).days if len(df) > 1 else 0
            bars_per_day = len(df) / days if days > 0 else 0

            print(f"\n{symbol}:")
            print(f"  Rows: {len(df)}")
            print(f"  Period: {df.index.min()} to {df.index.max()}")
            print(f"  Days: {days}")
            print(f"  Avg bars/day: {bars_per_day:.1f}")
            print(f"  Expected total: ~{days * 30:.0f} rows (30 bars/day)")

            if len(df) < days * 20:  # Меньше 20 баров в день - подозрительно
                print(f"  ⚠️  WARNING: Low data density!")
        else:
            print(f"\n{symbol}: ❌ File not found")


if __name__ == "__main__":
    check_data_volume()