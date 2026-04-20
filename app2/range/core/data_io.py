import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ...paths import ROOT


def _load_range_config(path: str) -> Dict[str, Any]:
    """
    Load range configuration from a JSON file. The function merges the base
    parameter dictionary with any risk profile overrides defined in the same
    file. The JSON structure is expected to mirror the RangeV3 config
    convention used throughout this project.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    range_cfg = cfg.get("RangeV3", {})
    params = dict(range_cfg.get("params", {}))
    profile_name = str(range_cfg.get("risk_profile", "") or "")
    profiles = range_cfg.get("risk_profiles", {})
    if profile_name and isinstance(profiles, dict):
        overrides = profiles.get(profile_name)
        if isinstance(overrides, dict):
            for key in overrides.keys():
                if key in params:
                    del params[key]
            params.update(overrides)
    return params


def _list_available_symbols(interval: str) -> List[str]:
    """Return a sorted list of symbols for which OHLCV data is available."""
    symbols: List[str] = []
    processed_dir = ROOT / "processed"
    data_dir = ROOT / "data"
    if processed_dir.exists():
        for p in processed_dir.glob(f"*_{interval}.csv"):
            name = p.name
            if name.endswith(f"_{interval}.csv"):
                symbols.append(name[: -len(f"_{interval}.csv")])
    if data_dir.exists():
        for p in data_dir.glob("*.csv"):
            symbols.append(p.stem)
    return sorted(set(symbols))


def _find_data_path(symbol: str, interval: str) -> str:
    """Prefer processed/{symbol}_{interval}.csv and fallback to data/{symbol}.csv."""
    fname_processed = ROOT / "processed" / f"{symbol}_{interval}.csv"
    fname_data = ROOT / "data" / f"{symbol}.csv"
    if fname_processed.exists():
        return str(fname_processed)
    if fname_data.exists():
        return str(fname_data)
    raise FileNotFoundError(
        f"Cannot find data for {symbol}: tried {fname_processed} and {fname_data}"
    )


def _load_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    """Load OHLCV data and enforce the active data contract for core."""
    path = _find_data_path(symbol, interval)
    df = pd.read_csv(path)

    dt_col = None
    for c in df.columns:
        lc = c.lower()
        if "time" in lc or "date" in lc or lc in {"begin", "end", "timestamp", "ts"}:
            dt_col = c
            break
    if dt_col is None:
        dt_col = df.columns[0]

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True).set_index(dt_col)

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        dup_count = int(df.index.duplicated().sum())
        raise ValueError(
            f"Data for {symbol} contains {dup_count} duplicate timestamps in {path}"
        )

    rename_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("open"):
            rename_map[c] = "open"
        elif lc.startswith("high"):
            rename_map[c] = "high"
        elif lc.startswith("low"):
            rename_map[c] = "low"
        elif lc.startswith("close"):
            rename_map[c] = "close"
        elif lc.startswith("vol"):
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    required_price_cols = ["open", "high", "low", "close"]
    for col in required_price_cols:
        if col not in df.columns:
            raise ValueError(
                f"Data for {symbol} missing required column '{col}' in {path}"
            )
    if "volume" not in df.columns:
        df["volume"] = 0.0

    required_cols = ["open", "high", "low", "close", "volume"]
    df = df.dropna(subset=required_cols).copy()

    if len(df) < 10:
        raise ValueError(
            f"Data for {symbol} has too few valid rows (<10) in {path}: {len(df)}"
        )
    if (df[required_price_cols] <= 0).any().any():
        raise ValueError(f"Data for {symbol} contains non-positive OHLC values in {path}")
    if (df["volume"] < 0).any():
        raise ValueError(f"Data for {symbol} contains negative volume in {path}")

    invalid_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    if bool(invalid_ohlc.any()):
        raise ValueError(f"Data for {symbol} contains invalid OHLC rows in {path}")

    return df
