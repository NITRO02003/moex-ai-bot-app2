# quick_candles_test.py
import os, sys, datetime as dt
import pandas as pd
import pytz
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import quotation_to_decimal

TOKEN = os.environ.get("TINKOFF_TOKEN")
ACCOUNT_ID = os.environ.get("ACCOUNT_ID")

if not TOKEN:
    print("Нет переменной TINKOFF_TOKEN"); sys.exit(1)

MSK = pytz.timezone("Europe/Moscow")
UTC = pytz.UTC

def last_business_day_msk(now=None):
    now = now or dt.datetime.now(MSK)
    d = now.date()
    # шаг назад до буднего дня (простая эвристика; без учёта праздников)
    while d.weekday() >= 5:  # 5=сб, 6=вс
        d = d - dt.timedelta(days=1)
    return d

def msk_interval_to_utc(day, start_hm="10:00", end_hm="18:45"):
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    start_msk = MSK.localize(dt.datetime(day.year, day.month, day.day, sh, sm))
    end_msk   = MSK.localize(dt.datetime(day.year, day.month, day.day, eh, em))
    return start_msk.astimezone(UTC), end_msk.astimezone(UTC)

def get_figi(client, ticker):
    r = client.instruments.find_instrument(query=ticker)
    for it in r.instruments:
        if it.ticker == ticker and it.figi:
            return it.figi
    raise RuntimeError(f"FIGI not found for {ticker}")

def fetch_1m(client, ticker, start_utc, end_utc):
    figi = get_figi(client, ticker)
    resp = client.market_data.get_candles(
        figi=figi, from_=start_utc, to=end_utc,
        interval=CandleInterval.CANDLE_INTERVAL_1_MIN
    )
    rows = []
    for c in resp.candles:
        rows.append({
            "time": c.time,
            "open": float(quotation_to_decimal(c.open)),
            "high": float(quotation_to_decimal(c.high)),
            "low":  float(quotation_to_decimal(c.low)),
            "close":float(quotation_to_decimal(c.close)),
            "volume": c.volume,
        })
    return pd.DataFrame(rows)

def main():
    ticker = "SBER"
    day = last_business_day_msk()
    start_utc, end_utc = msk_interval_to_utc(day, "10:00", "18:45")
    print(f"Запрашиваем {ticker}: {day} 10:00–18:45 MSK → UTC {start_utc}..{end_utc}")

    with Client(TOKEN) as c:
        df = fetch_1m(c, ticker, start_utc, end_utc)

    print("Свечей:", len(df))
    if len(df):
        print(df.head(5))
        p = r".\data\raw"; os.makedirs(p, exist_ok=True)
        out = os.path.join(p, f"{ticker}_1m_{day}.parquet")
        df.to_parquet(out, index=False)
        print("Сохранено в:", out)
    else:
        print("Пусто. Это бывает в выходные/праздники или если окно времени вне сессии.")

if __name__ == "__main__":
    main()
