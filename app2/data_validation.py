# data_validation.py
from app2 import data as D, diagnostics as DX
import pandas as pd


def validate_data_coverage():
    """Проверяем покрытие данных за требуемый период"""
    symbols = ['GAZP', 'GMKN', 'LKOH', 'ROSN', 'SBER', 'YNDX']
    start_date = '2022-01-15'
    end_date = '2024-12-31'

    print("=== ПРОВЕРКА ДАННЫХ ===")
    print(f"Требуемый период: {start_date} - {end_date}")
    print()

    coverage_report = {}

    for symbol in symbols:
        print(f"🔍 Проверяем {symbol}...")
        try:
            prices = D.load_csv(symbol)

            if prices.empty:
                print(f"   ❌ Нет данных")
                coverage_report[symbol] = {'status': 'NO_DATA', 'bars': 0}
                continue

            first_dt = prices.index.min()
            last_dt = prices.index.max()
            first_date = first_dt.strftime('%Y-%m-%d')
            last_date = last_dt.strftime('%Y-%m-%d')
            total_bars = len(prices)

            # Проверяем покрытие
            coverage_start = first_date <= start_date
            coverage_end = last_date >= end_date

            status = "✅" if coverage_start and coverage_end else "⚠️"

            print(f"   {status} Данные: {first_date} - {last_date}")
            print(f"   📊 Баров: {total_bars:,}")

            if not coverage_start:
                print(f"   ⚠️  Не хватает данных с начала")
            if not coverage_end:
                print(f"   ⚠️  Не хватает данных до конца")

            coverage_report[symbol] = {
                'status': 'OK' if coverage_start and coverage_end else 'PARTIAL',
                'first_date': first_date,
                'last_date': last_date,
                'total_bars': total_bars,
                'coverage_start': coverage_start,
                'coverage_end': coverage_end
            }

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            coverage_report[symbol] = {'status': 'ERROR', 'error': str(e)}

    print("\n" + "=" * 50)
    print("📋 СВОДКА ПОКРЫТИЯ ДАННЫХ:")

    ok_count = sum(1 for r in coverage_report.values() if r.get('status') == 'OK')
    partial_count = sum(1 for r in coverage_report.values() if r.get('status') == 'PARTIAL')

    print(f"✅ Полное покрытие: {ok_count} тикеров")
    print(f"⚠️  Частичное покрытие: {partial_count} тикеров")
    print(f"❌ Нет данных: {len(symbols) - ok_count - partial_count} тикеров")

    return coverage_report


if __name__ == "__main__":
    coverage = validate_data_coverage()