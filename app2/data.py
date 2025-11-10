from __future__ import annotations
from typing import Optional, Iterable, Dict
import pandas as pd
from .paths import DATA_DIR

try:
    from app.data_fetch import fetch_moex_candles_10m as _fetch10
    HAVE_EXT = True
except Exception:
    HAVE_EXT = False
    import requests
    MOEX = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards'
    def _fetch10(symbol: str, start: Optional[str]=None, end: Optional[str]=None, verbose: bool=False) -> pd.DataFrame:
        params = {'interval':10}
        if start: params['from'] = start if ' ' in start else f'{start} 09:50:00'
        if end:   params['till'] = end if ' ' in end else f'{end} 23:59:59'
        url = f"{MOEX}/TQBR/securities/{symbol}/candles.json"
        if verbose:
            print(f"[fetch] Trying TQBR: {url}?interval=10&from={params.get('from')}&till={params.get('till')}")
            print(f"[fetch] GET {url}", flush=True)
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        data = js.get('candles',{}).get('data',[])
        cols = js.get('candles',{}).get('columns',[])
        if not data: 
            return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
        df = pd.DataFrame(data, columns=cols)
        idx = pd.to_datetime(df['begin'], errors='coerce', utc=True)
        out = df.set_index(idx)[['open','high','low','close','volume']].dropna().sort_index()
        return out

def _concat_nonempty(parts):
    parts = [x for x in parts if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep='last')]
    return out.sort_index()

def fetch_range(symbol: str, start: str | None, end: str | None, interval: str = '10min', verbose: bool = False) -> pd.DataFrame:
    """Загрузка свечей с MOEX ISS пачками по 500 записей."""
    if interval != '10min':
        raise ValueError("Поддерживается только interval='10min'")
    import requests
    MOEX = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards'
    url = f"{MOEX}/TQBR/securities/{symbol}/candles.json"

    def norm_dt(s: str | None, is_from: bool) -> str | None:
        if not s:
            return None
        if ' ' in s:
            return s
        return f"{s} 09:50:00" if is_from else f"{s} 23:59:59"

    cursor = norm_dt(start, True)
    till   = norm_dt(end, False)
    parts = []

    while True:
        params = {"interval": 10}
        if cursor: params["from"] = cursor
        if till:   params["till"] = till
        if verbose:
            print(f"[fetch] Trying TQBR: {url}?interval=10&from={params.get('from')}&till={params.get('till')}")
            print(f"[fetch] GET {url}")
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        data = js.get("candles", {}).get("data", [])
        cols = js.get("candles", {}).get("columns", [])
        if not data:
            break

        df = pd.DataFrame(data, columns=cols)
        idx = pd.to_datetime(df["begin"], errors="coerce", utc=True)
        chunk = df.set_index(idx)[["open", "high", "low", "close", "volume"]].dropna().sort_index()
        parts.append(chunk)

        # курсор -> последняя begin + 1 сек, пока <500 — завершаем
        last_begin = str(df["begin"].iloc[-1])
        cursor = (pd.to_datetime(last_begin) + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        if verbose:
            print(f"[fetch] chunk rows={len(df)} next_from={cursor}")
        if len(df) < 500:
            break

    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out

def save_csv(symbol: str, df: pd.DataFrame) -> str:
    path = DATA_DIR / f'{symbol}.csv'
    df.to_csv(path)
    return str(path)

def load_csv(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f'{symbol}.csv'
    if not path.exists():
        return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
    raw = pd.read_csv(path)
    idx = pd.to_datetime(raw.iloc[:,0], errors='coerce', utc=True)
    body = raw.drop(columns=[raw.columns[0]])
    body.index = idx
    return body.sort_index()
