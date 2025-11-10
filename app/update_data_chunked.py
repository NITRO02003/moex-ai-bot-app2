# app/update_data_chunked.py
from __future__ import annotations
import os, argparse
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from pathlib import Path

from .data_fetch import fetch_moex_candles_10m_paginated, DEFAULT_BOARDS
from .utils import ensure_dtindex_utc


def load_data_by_months(symbols: List[str], start_date: str, end_date: str,
                        data_dir: str = "data", verbose: bool = False):
    """Загрузка данных по месяцам с пагинацией"""

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Создаем список месяцев для загрузки
    current = start_dt
    months = []

    while current <= end_dt:
        # Начало месяца
        month_start = current.replace(day=1)
        # Конец месяца
        if current.month == 12:
            month_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)

        # Ограничиваем конечной датой
        if month_end > end_dt:
            month_end = end_dt

        months.append((month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))

        # Следующий месяц
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    print(f"Will load data for {len(months)} months: {months}")

    for symbol in symbols:
        print(f"\n=== Loading {symbol} ===")
        all_data = []

        for i, (month_start, month_end) in enumerate(months):
            print(f"Month {i + 1}/{len(months)}: {month_start} to {month_end}")

            monthly_data = fetch_moex_candles_10m_paginated(
                symbol,
                start=month_start,
                end=month_end,
                boards=DEFAULT_BOARDS,
                verbose=verbose
            )

            if not monthly_data.empty:
                all_data.append(monthly_data)
                print(f"  → {len(monthly_data)} rows")
            else:
                print(f"  → No data")

            # Пауза между месяцами
            import time
            time.sleep(1)

        if all_data:
            combined = pd.concat(all_data).sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]

            # Сохраняем
            output_path = Path(data_dir) / f"{symbol}.csv"
            combined = ensure_dtindex_utc(combined)
            combined.to_csv(output_path)
            print(f"✅ {symbol}: {len(combined)} total rows saved to {output_path}")
        else:
            print(f"❌ {symbol}: No data collected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["SBER", "GAZP", "LKOH", "GMKN", "ROSN"])
    ap.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    load_data_by_months(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        data_dir=args.data_dir,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()