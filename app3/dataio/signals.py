from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils.timeparse import parse_datetime_series
from ..utils.io import read_csv_flexible

def load_signals(symbol: str, base_dir: str = "signals") -> pd.DataFrame:
    path = Path(base_dir) / f"{symbol}.csv"
    if not path.exists():
        from ..dataio.loader import load_bars
        bars = load_bars(symbol)
        if bars is None or bars.empty:
            return pd.DataFrame(columns=["dt","side"])
        s = bars[["dt","close"]].copy()
        fast = s["close"].ewm(span=5, adjust=False).mean()
        slow = s["close"].ewm(span=25, adjust=False).mean()
        side = np.sign(fast - slow).astype(int)
        side_shift = side.shift(1).fillna(0).astype(int)
        chg = side != side_shift
        s = s.loc[chg].copy()
        s["side"] = side.loc[chg].astype(int)
        s["conf"] = 0.75
        return s[["dt","side","conf"]]

    s = read_csv_flexible(path)
    s.columns = [c.lower() for c in s.columns]

    dt = None
    if "tradedate" in s.columns and any(c in s.columns for c in ["time","tradetime","systime"]):
        tcol = "time" if "time" in s.columns else "tradetime" if "tradetime" in s.columns else "systime"
        combo = s["tradedate"].astype(str).str.strip() + " " + s[tcol].astype(str).str.strip()
        dt = pd.to_datetime(combo, errors="coerce")
        if dt.notna().sum() == 0:
            combo = s["tradedate"].astype(str).str.strip() + " " + s[tcol].astype(str).str.zfill(8)
            dt = pd.to_datetime(combo, errors="coerce")
    if dt is None:
        for c in ["dt","datetime","datetime_iso","datetime_utc","timestamp","time","date","begin","end","candle_begin_time","candle_end_time"]:
            if c in s.columns:
                dt = parse_datetime_series(s[c])
                if dt is not None: break
    if dt is None:
        for c in s.columns:
            if any(k in c for k in ("time","date","dt")):
                dtest = parse_datetime_series(s[c])
                if dtest is not None:
                    dt = dtest; break
    if dt is None or dt.isna().all():
        raise KeyError("load_signals: no recognizable datetime column")

    s["dt"] = pd.to_datetime(dt)
    if "side" not in s.columns:
        if "score" in s.columns:
            s["side"] = np.sign(s["score"]).astype(int)
        elif "signal" in s.columns:
            s["side"] = np.sign(s["signal"]).astype(int)
        else:
            s["side"] = 0
    s["side"] = s["side"].astype(int).clip(-1, 1)

    s = s.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    cols = ["dt","side"] + [c for c in s.columns if c not in ("dt","side")]
    return s[cols]
