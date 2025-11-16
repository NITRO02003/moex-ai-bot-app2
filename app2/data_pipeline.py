"""Data processing pipeline for MOEX AI bot.

This module provides functions to convert raw CSV price data into a cleaned
and resampled form.  Raw files are expected to reside in a directory such
as ``data`` with names like ``SBER.csv``.  Each file should contain at
least columns ``open``, ``high``, ``low``, ``close``, ``volume`` and
optionally a timestamp column named ``begin``.  If no ``begin`` column is
present, the first column is assumed to hold timestamps.

Processed data are written to an output directory (by default ``processed``)
using filenames of the form ``{symbol}_{interval}.csv``.  Timestamps are
parsed as UTC and then converted to Europe/Moscow timezone.  Missing bars
are dropped.

Example usage::

    from app2.data_pipeline import process_all
    process_all(["SBER", "GAZP"], input_dir="data", output_dir="processed",
                intervals=["10min", "30min", "1h"])

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# Import the centralised resampling function from data_utils instead of CLI
from .data_utils import resample_prices


def _load_raw(symbol: str, input_dir: str) -> Optional[pd.DataFrame]:
    """Load a raw CSV for a symbol from ``input_dir``.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. ``"SBER"``.
    input_dir : str
        Directory where raw CSV files are stored.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing raw OHLCV data with a ``begin`` column if
        present.  If the file is not found or cannot be read, returns None.
    """
    path = Path(input_dir) / f"{symbol}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df


def process_symbol(
    symbol: str,
    intervals: Iterable[str] = ("10min",),
    input_dir: str = "data",
    output_dir: str = "processed",
) -> List[str]:
    """Process a single symbol and write resampled CSV files.

    This helper reads the raw CSV for ``symbol`` from ``input_dir``,
    resamples it to each interval in ``intervals`` using ``_resample_prices``
    (which converts timestamps to Europe/Moscow timezone and aggregates
    OHLCV data) and writes the result to ``output_dir`` with filenames
    ``{symbol}_{interval}.csv``.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    intervals : Iterable[str], default ("10min",)
        Bar intervals to resample to, e.g. ``["10min", "30min", "1h"]``.
    input_dir : str, default ``"data"``
        Directory containing raw CSV files.
    output_dir : str, default ``"processed"``
        Directory where processed CSV files will be written.

    Returns
    -------
    list of str
        List of file paths that were created.  If no raw data could be
        loaded for the symbol, returns an empty list.
    """
    raw_df = _load_raw(symbol, input_dir)
    if raw_df is None:
        return []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    for interval in intervals:
        try:
            # Delegate resampling to the data utilities module to avoid CLI circular dependency
            processed = resample_prices(raw_df, interval)
        except Exception:
            continue
        out_path = out_dir / f"{symbol}_{interval}.csv"
        processed.to_csv(out_path, index=False)
        created.append(str(out_path))
    return created


def process_all(
    symbols: Iterable[str],
    intervals: Iterable[str] = ("10min",),
    input_dir: str = "data",
    output_dir: str = "processed",
) -> Dict[str, List[str]]:
    """Process multiple symbols and return a mapping of outputs.

    Parameters
    ----------
    symbols : iterable of str
        Iterable of ticker symbols to process.
    intervals : iterable of str, default ("10min",)
        Bar intervals to resample to for each symbol.
    input_dir : str, default ``"data"``
        Directory of raw CSV files.
    output_dir : str, default ``"processed"``
        Directory to write processed CSV files.

    Returns
    -------
    dict
        Mapping from each symbol to a list of created file paths.  Symbols
        for which no input data could be loaded will have an empty list.
    """
    result: Dict[str, List[str]] = {}
    for sym in symbols:
        result[sym] = process_symbol(sym, intervals, input_dir, output_dir)
    return result