# data_fetch.py - ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ
from __future__ import annotations
from typing import Optional, Dict, Any, Iterable
import time
import pandas as pd
import requests

MOEX_ISS_BASE = "https://iss.moex.com/iss"
UA = "curl/8.5.0"
DEFAULT_BOARDS = ("TQBR", "TQTF", "TQTD", "FQBR")


def _clean_dt(s: Optional[str], fallback_time: str) -> Optional[str]:
    """Очистка и форматирование даты"""
    if not s:
        return None
    s = str(s).strip()
    # если пришла только дата — добавляем время как в твоем примере
    if len(s) <= 10 and s.count("-") == 2:
        return f"{s} {fallback_time}"
    return s


def _mk_session() -> requests.Session:
    """Создание HTTP сессии"""
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": UA,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    return sess


def _parse_candles(payload: Dict[str, Any]) -> pd.DataFrame:
    """Парсинг данных свечей"""
    try:
        block = payload.get("candles", {})
        data = block.get("data", [])
        cols = block.get("columns", [])

        if not data:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

        # Создаем DataFrame
        df = pd.DataFrame(data, columns=cols)

        # Преобразуем временную метку
        df['datetime'] = pd.to_datetime(df['begin'], utc=True)

        # Функция для конвертации российских чисел
        def safe_convert(value):
            if isinstance(value, (int, float)):
                return value
            try:
                # Заменяем запятую на точку, убираем пробелы
                clean_val = str(value).replace(' ', '').replace(',', '.')
                return float(clean_val)
            except (ValueError, TypeError):
                return float('nan')

        # Конвертируем все числовые колонки
        ohlc_data = []
        for i, row in enumerate(data):
            try:
                ohlc_data.append({
                    'datetime': pd.to_datetime(row[cols.index('begin')], utc=True),
                    'open': safe_convert(row[cols.index('open')]),
                    'high': safe_convert(row[cols.index('high')]),
                    'low': safe_convert(row[cols.index('low')]),
                    'close': safe_convert(row[cols.index('close')]),
                    'volume': safe_convert(row[cols.index('volume')])
                })
            except (ValueError, IndexError) as e:
                continue

        if not ohlc_data:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

        result = pd.DataFrame(ohlc_data).set_index('datetime')
        result = result.dropna(subset=['open', 'high', 'low', 'close'])

        return result

    except Exception as e:
        print(f"[_parse_candles] Error: {e}")
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))


def _get_raw(sess: requests.Session, url: str, verbose: bool = False) -> Dict[str, Any]:
    """Выполнение HTTP запроса"""
    if verbose:
        print(f"[fetch] GET {url}")

    r = sess.get(url, timeout=30)

    if r.status_code in (429, 503):
        time.sleep(0.5)
        r = sess.get(url, timeout=30)
        if verbose:
            print(f"[fetch] RETRY -> {r.status_code}")

    r.raise_for_status()

    try:
        return r.json()
    except Exception:
        # На случай если вернулся HTML/ошибка прокси
        txt = r.text[:400]
        if verbose:
            print(f"[fetch] non-JSON response head:\n{txt}")
        raise


def _url_with_dates(board: str, symbol: str, start: Optional[str], end: Optional[str]) -> str:
    """Формирование URL с датами"""
    base = f"{MOEX_ISS_BASE}/engines/stock/markets/shares/boards/{board}/securities/{symbol}/candles.json"
    qs = ["interval=10"]
    if start:
        qs.append(f"from={start.replace(' ', '%20')}")
    if end:
        qs.append(f"till={end.replace(' ', '%20')}")
    return base + "?" + "&".join(qs)


def fetch_moex_candles_10m(symbol: str,
                           start: Optional[str] = None,
                           end: Optional[str] = None,
                           boards: Iterable[str] = DEFAULT_BOARDS,
                           verbose: bool = False) -> pd.DataFrame:
    """
    Основная функция загрузки 10-минутных свечей
    """
    start = _clean_dt(start, "09:50:00")
    end = _clean_dt(end, "23:59:59")
    sess = _mk_session()

    # 1) Попробуем получить данные по датам
    for b in boards:
        url = _url_with_dates(b, symbol, start, end)
        try:
            if verbose:
                print(f"[fetch] Trying {b}: {url}")

            payload = _get_raw(sess, url, verbose=verbose)
            df = _parse_candles(payload)

            if not df.empty:
                if verbose:
                    print(f"[fetch] SUCCESS {symbol} on {b}: {len(df)} rows")
                return df

        except Exception as e:
            if verbose:
                print(f"[fetch] Error on {b}: {e}")

    # 2) Фоллбек - последние доступные данные
    for b in boards:
        url = f"{MOEX_ISS_BASE}/engines/stock/markets/shares/boards/{b}/securities/{symbol}/candles.json?interval=10&limit=500"
        try:
            payload = _get_raw(sess, url, verbose=verbose)
            df = _parse_candles(payload)
            if not df.empty:
                if verbose:
                    print(f"[fetch] FALLBACK {symbol}: {len(df)} rows")
                return df
        except Exception as e:
            if verbose:
                print(f"[fetch] Fallback error: {e}")

    if verbose:
        print(f"[fetch] No data for {symbol}")
    return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))


def fetch_moex_candles_10m_paginated(symbol: str,
                                     start: Optional[str] = None,
                                     end: Optional[str] = None,
                                     boards: Iterable[str] = DEFAULT_BOARDS,
                                     verbose: bool = False) -> pd.DataFrame:
    """
    Загрузка данных с пагинацией - обход ограничения 500 строк от MOEX
    """
    start = _clean_dt(start, "09:50:00")
    end = _clean_dt(end, "23:59:59")
    sess = _mk_session()

    all_frames = []

    for b in boards:
        offset = 0
        limit = 500  # MOEX всегда возвращает 500, независимо от лимита
        board_frames = []

        if verbose:
            print(f"[paginated] Starting pagination for {symbol} on {b}")

        while True:
            # Формируем URL с пагинацией
            url = _url_with_dates(b, symbol, start, end)
            if offset > 0:
                url += f"&start={offset}"

            try:
                if verbose:
                    print(f"[paginated] Request: offset={offset}, url={url}")

                payload = _get_raw(sess, url, verbose=verbose)
                df = _parse_candles(payload)

                if df.empty:
                    if verbose:
                        print(f"[paginated] No more data at offset {offset}")
                    break

                board_frames.append(df)
                if verbose:
                    print(f"[paginated] Got {len(df)} rows at offset {offset}")

                # Если получили меньше 500 строк - значит это последняя страница
                if len(df) < 500:
                    if verbose:
                        print(f"[paginated] Last page detected: {len(df)} < 500")
                    break

                offset += 500
                time.sleep(0.3)  # Пауза между запросами

            except Exception as e:
                if verbose:
                    print(f"[paginated] Error at offset {offset}: {e}")
                break

        if board_frames:
            board_data = pd.concat(board_frames).sort_index()
            all_frames.append(board_data)
            if verbose:
                print(f"[paginated] Board {b} total: {len(board_data)} rows")
            break  # Если нашли данные на одной площадке, переходим к след. символу

    if all_frames:
        result = pd.concat(all_frames).sort_index()
        # Убираем дубликаты на случай перекрытия данных
        result = result[~result.index.duplicated(keep='first')]
        if verbose:
            print(f"[paginated] FINAL {symbol}: {len(result)} rows from {result.index.min()} to {result.index.max()}")
        return result
    else:
        if verbose:
            print(f"[paginated] No data for {symbol}")
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))