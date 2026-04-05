import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

def _load_range_config(path: str) -> Dict[str, Any]:
    """
    Load range configuration from a JSON file. The function merges the base
    parameter dictionary with any risk profile overrides defined in the same
    file. The JSON structure is expected to mirror the RangeV3 config
    convention used throughout this project.

    Parameters
    ----------
    path : str
        Filesystem path to a JSON configuration file.

    Returns
    -------
    dict
        A flat dictionary of parameter values after applying any profile
        overrides. If the file is missing or malformed, an exception is
        propagated to the caller.
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
            params.update(overrides)
    return params


def _list_available_symbols(interval: str) -> List[str]:
    """
    Return a sorted list of symbols for which OHLCV data is available.

    The function first looks in the processed/ directory for files named
    {symbol}_{interval}.csv and then in the data/ directory for generic
    {symbol}.csv files. The combined set of symbols is deduplicated and
    sorted alphabetically.

    Parameters
    ----------
    interval : str
        The bar interval suffix used in processed file names (e.g. "5m").

    Returns
    -------
    list[str]
        A sorted list of symbol names.
    """
    symbols: List[str] = []
    processed_dir = Path("processed")
    data_dir = Path("data")
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
    """
    Determine the best path to the OHLCV CSV for a given symbol and interval.

    The function prefers the processed/{symbol}_{interval}.csv file if it
    exists; otherwise it falls back to data/{symbol}.csv. If neither file
    exists, it raises FileNotFoundError.

    Parameters
    ----------
    symbol : str
        The ticker symbol to look up.
    interval : str
        The bar interval suffix used in processed file names.

    Returns
    -------
    str
        Path to the CSV file containing OHLCV data.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    fname_processed = os.path.join("processed", f"{symbol}_{interval}.csv")
    fname_data = os.path.join("data", f"{symbol}.csv")
    if os.path.exists(fname_processed):
        return fname_processed
    if os.path.exists(fname_data):
        return fname_data
    raise FileNotFoundError(
        f"Cannot find data for {symbol}: tried {fname_processed} and {fname_data}"
    )


def _load_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    """
    Load OHLCV data for a given symbol and interval.

    The function locates the appropriate CSV using `_find_data_path`, loads
    it into a DataFrame, infers and normalizes the datetime column, and
    ensures the required OHLC columns exist. If the volume column is
    missing, it is synthesized as zeros.

    Parameters
    ----------
    symbol : str
        The ticker symbol to load.
    interval : str
        The bar interval suffix used in processed file names.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by datetime with columns: open, high, low, close,
        volume and any other original columns preserved.

    Raises
    ------
    ValueError
        If required OHLC columns are missing.
    """
    path = _find_data_path(symbol, interval)
    df = pd.read_csv(path)

    # attempt to infer the datetime column by searching for common names
    dt_col = None
    for c in df.columns:
        lc = c.lower()
        if "time" in lc or "date" in lc or "dt" in lc:
            dt_col = c
            break
    if dt_col is None:
        for c in df.columns:
            lc = c.lower()
            if lc in ("begin", "end", "timestamp", "ts"):
                dt_col = c
                break
    if dt_col is None:
        # fallback to the first column if nothing matches
        dt_col = df.columns[0]

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True).set_index(dt_col)

    # normalize common OHLC column names to canonical lower-case names
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

    # ensure required columns are present
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(
                f"Data for {symbol} missing required column '{col}' in {path}"
            )
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df