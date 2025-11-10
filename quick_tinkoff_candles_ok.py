from datetime import timedelta
from tinkoff.invest import Client, CandleInterval, InstrumentIdType
from tinkoff.invest.utils import now

TOKEN = None
ACCOUNT_ID = None  # не нужен для свечей, но оставим для единообразия

import os
TOKEN = os.environ.get("TINKOFF_TOKEN")
ACCOUNT_ID = os.environ.get("ACCOUNT_ID")

assert TOKEN, "Переменная окружения TINKOFF_TOKEN не задана"

TICKER = "SBER"
CLASS_CODE = "TQBR"  # акции на Мосбирже

with Client(TOKEN) as c:
    # 1) Находим FIGI именно акции SBER на TQBR
    share = c.instruments.share_by(
        id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
        class_code=CLASS_CODE,
        id=TICKER
    )
    figi = share.instrument.figi

    # 2) Берём строго последние 24 часа в UTC
    to = now()                 # UTC
    frm = to - timedelta(hours=24)

    # 3) Запрашиваем минутки (диапазон <= 1 сутки!)
    r = c.market_data.get_candles(
        figi=figi,
        from_=frm,
        to=to,
        interval=CandleInterval.CANDLE_INTERVAL_1_MIN
    )

    candles = r.candles
    print(f"FIGI: {figi}, минутных свечей: {len(candles)}")
    if candles:
        c0, c1 = candles[0], candles[-1]
        print("Первая:", c0.time, float(c0.open.units + c0.open.nano/1e9))
        print("Последняя:", c1.time, float(c1.close.units + c1.close.nano/1e9))
