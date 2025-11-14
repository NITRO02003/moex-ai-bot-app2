from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..utils.timeparse import parse_datetime_series
from ..utils.io import read_csv_flexible

def load_bars(symbol: str, base_dir: str = "data") -> pd.DataFrame | None:
    path = Path(base_dir) / f"{symbol}.csv"
    if not path.exists():
        return None
    df = read_csv_flexible(path)
    if df is None or df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    dt = None
    if "tradedate" in df.columns and any(c in df.columns for c in ["time","tradetime","systime"]):
        tcol = "time" if "time" in df.columns else "tradetime" if "tradetime" in df.columns else "systime"
        combo = df["tradedate"].astype(str).str.strip() + " " + df[tcol].astype(str).str.strip()
        dt = pd.to_datetime(combo, errors="coerce")
        if dt.notna().sum() == 0:
            combo = df["tradedate"].astype(str).str.strip() + " " + df[tcol].astype(str).str.zfill(8)
            dt = pd.to_datetime(combo, errors="coerce")
    if dt is None:
        for c in ["dt","datetime","datetime_iso","datetime_utc","timestamp","time","date","begin","end","candle_begin_time","candle_end_time"]:
            if c in df.columns:
                dt = parse_datetime_series(df[c]); 
                if dt is not None: break
    if dt is None:
        for c in df.columns:
            if any(k in c for k in ("time","date","dt")):
                dtest = parse_datetime_series(df[c])
                if dtest is not None:
                    dt = dtest; break
    if dt is None or dt.isna().all():
        raise KeyError("load_bars: no recognizable datetime column")

    df["dt"] = pd.to_datetime(dt)
    df = df.dropna(subset=["dt"])

    o_col = pick("open","o","price_open","openprice")
    h_col = pick("high","h","price_high","highprice")
    l_col = pick("low","l","price_low","lowprice")
    c_col = pick("close","c","last","price","settle","closeprice","pr_close","value")
    v_col = pick("volume","vol","qty","turnover","val")

    if c_col is None:
        if o_col is None:
            raise KeyError("load_bars: cannot find close/price column")
        c_col = o_col

    df["close"] = pd.to_numeric(df[c_col], errors="coerce")
    df["open"]  = pd.to_numeric(df[o_col], errors="coerce") if o_col else df["close"]
    df["high"]  = pd.to_numeric(df[h_col], errors="coerce") if h_col else df["close"]
    df["low"]   = pd.to_numeric(df[l_col], errors="coerce") if l_col else df["close"]
    df["volume"] = pd.to_numeric(df[v_col], errors="coerce").fillna(0.0) if v_col else 0.0

    df = df.dropna(subset=["close"]).sort_values("dt").reset_index(drop=True)
    return df[["dt","open","high","low","close","volume"]]
