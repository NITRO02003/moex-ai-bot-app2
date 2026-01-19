import requests, pandas as pd

def moex_1m(ticker: str, start: str):
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
    params = {"interval": 1, "from": start}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    cols = j["candles"]["columns"]
    data = j["candles"]["data"]
    df = pd.DataFrame(data, columns=cols)
    return df

if __name__ == "__main__":
    df = moex_1m("SBER", start="2025-11-07")
    print("ISS свечей:", len(df))
    print(df.head())
