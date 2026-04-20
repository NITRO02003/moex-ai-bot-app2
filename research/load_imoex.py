import requests
import pandas as pd

url = "https://iss.moex.com/iss/engines/stock/markets/index/securities/IMOEX/candles.json"
params = {
    "interval": 10,
    "from": "2020-01-01",
    "till": "2026-01-01"
}

data = []
start = 0

while True:
    params["start"] = start
    r = requests.get(url, params=params).json()
    candles = r["candles"]["data"]
    if not candles:
        break
    data.extend(candles)
    start += len(candles)

cols = r["candles"]["columns"]
df = pd.DataFrame(data, columns=cols)

df.to_csv("data/IMOEX_10min.csv", index=False)
print("done")