from __future__ import annotations
import pandas as pd

def parse_datetime_series(ser: pd.Series) -> pd.Series | None:
    """Supports epoch seconds/milliseconds and string timestamps. Returns Series or None."""
    if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
        v = pd.to_numeric(ser, errors="coerce").dropna()
        if not v.empty:
            mx = float(v.iloc[0])
            unit = "ms" if mx > 1e12 else "s"
            dt = pd.to_datetime(ser, unit=unit, utc=False, errors="coerce")
            if dt.notna().mean() > 0.5:
                return dt
    dt = pd.to_datetime(ser, utc=False, errors="coerce")
    if dt.notna().mean() > 0.5:
        return dt
    return None
