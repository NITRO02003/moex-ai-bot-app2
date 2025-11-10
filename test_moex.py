# test_moex.py
import requests
from datetime import datetime


def test_moex_api():
    symbols = ["SBER", "GAZP", "LKOH"]

    for symbol in symbols:
        # Тест 1: Без параметров (последние данные)
        url1 = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json?interval=10&limit=10"

        # Тест 2: С коротким периодом
        url2 = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json?interval=10&from=2024-09-01&till=2024-09-02"

        print(f"\n=== Testing {symbol} ===")

        for i, url in enumerate([url1, url2], 1):
            try:
                print(f"URL {i}: {url}")
                r = requests.get(url, timeout=10)
                data = r.json()
                candles = data.get("candles", {}).get("data", [])
                print(f"Result: {len(candles)} candles")

                if candles:
                    print(f"First candle: {candles[0]}")

            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    test_moex_api()