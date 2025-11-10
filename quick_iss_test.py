# quick_iss_test.py
import requests
import pandas as pd

def moex_1m(ticker: str, date: str):
    """
    ticker: 'SBER', 'GAZP', ...
    date: 'YYYY-MM-DD' (торговый день MOEX)
    Возвращает DataFrame 1-мин свечей за указанный день.
    """
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
    params = {
        "interval": 1,                    # 1 мин
        "from": f"{date} 00:00:00",
        "till": f"{date} 23:59:59",
        "iss.only": "candles",
        "iss.json": 1,
        "iss.meta": 0,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    cols = j["candles"]["columns"]
    data = j["candles"]["data"]
    df = pd.DataFrame(data, columns=cols)
    # приведение колонок к понятным именам
    if not df.empty:
        df.rename(columns={
            "open":"open","close":"close","high":"high","low":"low",
            "volume":"volume","begin":"begin","end":"end"
        }, inplace=True)
        df["begin"] = pd.to_datetime(df["begin"])
        df["end"]   = pd.to_datetime(df["end"])
        df = df.sort_values("begin").reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Сегодня суббота, 2025-11-08 — биржа закрыта.
    # Возьмём последнюю пятницу: 2025-11-07 (пример).
    df = moex_1m("SBER", date="2025-11-07")
    print("Свечей:", len(df))
    print(df.head())
