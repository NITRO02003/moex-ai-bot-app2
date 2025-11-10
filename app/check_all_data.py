# app/check_all_data.py
import pandas as pd
import os
from pathlib import Path


def check_all_data():
    # Определяем правильный путь к данным
    # Если скрипт запускается из корня проекта: python -m app.check_all_data
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # Альтернативно: если данные в app/data/
    alt_data_dir = Path(__file__).parent / "data"

    symbols = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN"]

    print("=== DATA QUALITY CHECK ===")
    print(f"Looking in: {data_dir}")
    print(f"Alternative: {alt_data_dir}")

    # Проверим оба возможных расположения
    used_data_dir = None
    if data_dir.exists():
        used_data_dir = data_dir
        print(f"✅ Using main data directory: {data_dir}")
    elif alt_data_dir.exists():
        used_data_dir = alt_data_dir
        print(f"✅ Using alternative data directory: {alt_data_dir}")
    else:
        print("❌ No data directory found!")
        return

    for symbol in symbols:
        path = used_data_dir / f"{symbol}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty:
                    print(f"\n✅ {symbol}:")
                    print(f"   Rows: {len(df)}")
                    print(f"   Period: {df.index.min()} to {df.index.max()}")
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Sample OHLC:")
                    print(
                        f"     First: O{df['open'].iloc[0]:.2f} H{df['high'].iloc[0]:.2f} L{df['low'].iloc[0]:.2f} C{df['close'].iloc[0]:.2f} V{df['volume'].iloc[0]:.0f}")
                    print(
                        f"     Last:  O{df['open'].iloc[-1]:.2f} H{df['high'].iloc[-1]:.2f} L{df['low'].iloc[-1]:.2f} C{df['close'].iloc[-1]:.2f} V{df['volume'].iloc[-1]:.0f}")
                else:
                    print(f"\n⚠️  {symbol}: EMPTY DataFrame")
            except Exception as e:
                print(f"\n❌ {symbol}: Error reading file - {e}")
        else:
            print(f"\n❌ {symbol}: File not found at {path}")


if __name__ == "__main__":
    check_all_data()