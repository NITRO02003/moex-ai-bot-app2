
# app/realtime.py
from __future__ import annotations
import os, time
from typing import Iterable, Dict
import pandas as pd

from .data_fetch import fetch_moex_candles_10m
from .utils import ensure_dtindex_utc

DATA_DIR = "data"

def merge_append_csv(path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            if not old.empty:
                if old.columns[0].lower() in ("", "unnamed: 0", "index"):
                    old = old.drop(columns=[old.columns[0]])
                # if first remaining column is unnamed timestamp
                if not isinstance(old.index, pd.DatetimeIndex):
                    old.index = pd.to_datetime(old.iloc[:,0], errors="coerce", utc=True)
                    old = old.drop(columns=[old.columns[0]])
                old = ensure_dtindex_utc(old)
            else:
                old = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        except Exception:
            old = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    else:
        old = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    if new_df is None or new_df.empty:
        return old
    new_df = ensure_dtindex_utc(new_df)
    df = pd.concat([old, new_df]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_csv(path)
    return df

def fetch_latest(symbols: Iterable[str], start: str, end: str, verbose: bool=False) -> Dict[str, pd.DataFrame]:
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {}
    for s in symbols:
        df = fetch_moex_candles_10m(s, start=start, end=end, verbose=verbose)
        csv_path = os.path.join(DATA_DIR, f"{s}.csv")
        merged = merge_append_csv(csv_path, df)
        out[s] = merged
        if verbose:
            print(f"[rt] {s}: merged rows={len(merged)}")
    return out
