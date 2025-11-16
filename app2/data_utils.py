"""Utility functions for loading and resampling price data.

This module centralises common data operations used across the backtester
and CLI commands: loading CSV price files for a given symbol and
resampling OHLCV data to arbitrary intervals with timezone conversion.

By placing these helpers here, we avoid circular imports between
`cli.py`, `data_pipeline.py` and `forward_test.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd


def load_prices(symbol: str, search_dirs: Iterable[str] = ("data", "")) -> pd.DataFrame:
    """Load price data for a given symbol from a CSV file.

    Parameters
    ----------
    symbol : str
        Ticker symbol to load, e.g. ``"SBER"``.
    search_dirs : iterable of str, default ("data", "")
        Directories to search for ``{symbol}.csv`` in order.  The empty
        string denotes the current working directory.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the price data.  If the file cannot be
        found, raises ``FileNotFoundError``.
    """
    for d in search_dirs:
        candidate = Path(d) / f"{symbol}.csv" if d else Path(f"{symbol}.csv")
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"Price file for {symbol} not found in any of: {list(search_dirs)}")


def resample_prices(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate a price dataframe to a different interval.

    This function is equivalent to the private ``_resample_prices`` in
    ``cli.py`` but is placed here to avoid circular imports.  It
    interprets a column named ``begin`` as the timestamp; otherwise it
    uses the existing index.  Timestamps are parsed as UTC and
    converted to Europe/Moscow timezone before resampling.  OHLCV
    columns are aggregated accordingly.

    Parameters
    ----------
    df : pandas.DataFrame
        Input OHLCV dataframe with columns ``open``, ``high``, ``low``,
        ``close`` and optionally ``volume``.  A column named ``begin`` is
        treated as the timestamp; otherwise the existing index is used.
    interval : str
        Pandas offset alias such as ``'10min'``, ``'30min'`` or ``'1h'``.

    Returns
    -------
    pandas.DataFrame
        Resampled dataframe with aggregated OHLCV columns and a ``begin``
        column for the new interval.
    """
    norm_interval = interval.lower()
    # Do nothing for the default 10‑minute interval except timezone conversion
    if norm_interval in ("10min", "10t", "10m"):
        if "begin" in df.columns:
            ts = pd.to_datetime(df["begin"], utc=True).dt.tz_convert("Europe/Moscow")
            out = df.copy()
            out["begin"] = ts
            return out
        return df
    # Copy to avoid modifying original
    dfx = df.copy()
    # Determine time column or index
    if "begin" in dfx.columns:
        idx = pd.to_datetime(dfx["begin"], utc=True)
        dfx = dfx.set_index(idx)
        dfx = dfx.drop(columns=["begin"])
    else:
        # Use existing index as timestamps
        dfx.index = pd.to_datetime(dfx.index, utc=True)
    # Aggregate using OHLCV semantics
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in dfx.columns:
        ohlc["volume"] = "sum"
    resampled = dfx.resample(norm_interval, label="right", closed="right").agg(ohlc)
    # Drop rows with missing values
    resampled = resampled.dropna(how="any")
    # Convert index to Europe/Moscow timezone
    resampled.index = resampled.index.tz_convert("Europe/Moscow")
    # Reset index to begin column
    resampled = resampled.reset_index().rename(columns={"index": "begin"})
    return resampled