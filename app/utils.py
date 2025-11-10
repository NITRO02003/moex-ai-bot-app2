# app/utils.py
from __future__ import annotations
from typing import Optional, Union, Dict, Iterable
import os
import pandas as pd

DateLike = Optional[Union[str, pd.Timestamp]]

__all__ = ["ensure_dtindex_utc","to_utc_ts","slice_by_date","load_all"]

def ensure_dtindex_utc(df: Optional[pd.DataFrame], index_col_guess: Optional[str] = None) -> pd.DataFrame:
    """Всегда возвращает DataFrame с DatetimeIndex (UTC). Пустой — тоже валидный."""
    if df is None:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    if not isinstance(df.index, pd.DatetimeIndex):
        candidates = []
        if index_col_guess and index_col_guess in df.columns:
            candidates.append(index_col_guess)
        for c in ("dt","datetime","timestamp","time","date","begin"):
            if c in df.columns and c not in candidates:
                candidates.append(c)
        parsed = None
        for col in candidates:
            try:
                parsed = pd.to_datetime(df[col], errors="raise", utc=True)
                df = df.drop(columns=[col]).copy()
                df.index = parsed
                break
            except Exception:
                parsed = None
        if parsed is None:
            try:
                df.index = pd.to_datetime(df.index, errors="raise", utc=True)
            except Exception:
                return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()

def to_utc_ts(x: DateLike) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0] if len(ts) else pd.NaT
    if pd.isna(ts):
        return None
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts

def slice_by_date(df: pd.DataFrame, start: DateLike=None, end: DateLike=None) -> pd.DataFrame:
    if df is None or df.empty:
        return ensure_dtindex_utc(df)
    df = ensure_dtindex_utc(df)
    s = to_utc_ts(start); e = to_utc_ts(end)
    if s is not None: df = df.loc[df.index >= s]
    if e is not None: df = df.loc[df.index <= e]
    return df

def _read_one_csv(path: str) -> pd.DataFrame:
    import pandas as pd, os
    if not os.path.exists(path):
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python")
    if df is None or df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    if df.columns[0].lower() in ("", "index", "unnamed: 0"):
        df = df.drop(columns=[df.columns[0]])
    for c in ("dt","datetime","timestamp","time","date","begin"):
        if c in df.columns:
            try:
                idx = pd.to_datetime(df[c], errors="coerce", utc=True)
                df = df.drop(columns=[c]).copy()
                df.index = idx
                df = df.loc[~df.index.isna()].sort_index()
                return ensure_dtindex_utc(df)
            except Exception:
                pass
    try:
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df.loc[~df.index.isna()].sort_index()
    except Exception:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    return ensure_dtindex_utc(df)

def load_all(data_dir: str, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}.csv")
        try:
            df = _read_one_csv(path)
        except Exception:
            df = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        out[sym] = ensure_dtindex_utc(df)
    return out