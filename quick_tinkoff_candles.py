import os
import datetime as dt
import pandas as pd
import pytz
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import quotation_to_decimal

TINKOFF_TOKEN = os.environ["TINKOFF_TOKEN"]
ACCOUNT_ID = os.environ["ACCOUNT_ID"]

MSK = pytz.timezone("Europe/Moscow")
UTC = pytz.utc

def last_trading_day_msk(now_msk: dt.datetime) -> dt.date:
    # если суббота/воскресенье — откатываемся на пятницу
    wd = now_msk.weekday()  # 0=Mon ... 6=Sun
    if wd == 5:  # Sat
        return (now_msk - dt.timedelta(days=1)).date()
    if wd == 6:  # Sun
        return (now_msk - dt.timedelta(days=2)).date()
    return now_msk.date()

def main(ticker="SBER"):
    now_msk = dt.datetime.now(MSK)
    day = last_trading_day_msk(now_msk)

    # окно торгов TQBR: 10:00–18:45 МСК (берём с запасом до 18:50)
    start_msk = MSK.localize(dt.datetime.combine(day, dt.time(10, 0, 0)))
    end_msk   = MSK.localize(dt.datetime.combine(day, dt.time(18, 50, 0)))
    start_utc = start_msk.astimezone(UTC).replace(tzinfo=None)
    end_utc   = end_msk.astimezone(UTC).replace(tzinfo=None)

    with Client(TINKOFF_TOKEN) as c:
        # найдём FIGI по тикеру
        resp = c.instruments.find_instrument(query=ticker)
        figi = None
        for it in resp.instruments:
            if it.ticker == ticker and it.figi:
                figi = it.figi
                break
        if not figi:
            raise RuntimeError(f"FIGI не найден для {ticker}")

        candles = c.market_data.get_candles(
            figi=figi,
            from_=start_utc,
            to=end_utc,
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
        ).candles

        rows = []
        for x in candles:
            rows.append({
                "time": x.time,
                "open": float(quotation_to_decimal(x.open)),
                "high": float(quotation_to_decimal(x.high)),
                "low":  float(quotation_to_decimal(x.low)),
                "close":float(quotation_to_decimal(x.close)),
                "volume": x.volume,
            })
        df = pd.DataFrame(rows)
        print(f"Тикер: {ticker}, дата: {day}, свечей: {len(df)}")
        if not df.empty:
            print(df.head(3))
            print(df.tail(3))

if __name__ == "__main__":
    main("SBER")
