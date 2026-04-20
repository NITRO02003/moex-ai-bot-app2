from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..paths import ROOT
from ..utils import load_symbols

REQUIRED_COLUMNS = ["begin", "open", "high", "low", "close", "volume"]


@dataclass
class FileReport:
    symbol: str
    source: str
    interval: str
    path: str
    rows: int
    missing_columns: list[str]
    nan_rows: int
    duplicate_begin: int
    non_monotonic_begin: int
    invalid_begin_rows: int
    invalid_price_rows: int
    invalid_volume_rows: int
    invalid_ohlc_rows: int
    zero_range_rows: int
    unexpected_gap_rows: int
    timeframe_expected_minutes: Optional[int]
    timeframe_mode_minutes: Optional[float]
    timezone: Optional[str]
    status: str
    error: Optional[str]


def _infer_minutes(delta: pd.Series) -> Optional[float]:
    if delta.empty:
        return None
    mode_val = delta.mode()
    if mode_val.empty:
        return None
    return float(mode_val.iloc[0] / 60.0)


def _expected_minutes(interval: str) -> Optional[int]:
    name = interval.strip().lower()
    if name == "10min":
        return 10
    if name == "30min":
        return 30
    if name == "1h":
        return 60
    return None


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "begin" in df.columns:
        dt_col = "begin"
    elif "datetime" in df.columns:
        dt_col = "datetime"
    else:
        dt_col = df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    if dt_col != "begin":
        df = df.rename(columns={dt_col: "begin"})
    return df


def _validate_file(symbol: str, source: str, interval: str, path: Path) -> FileReport:
    try:
        df = _load_csv(path)
    except Exception as exc:
        return FileReport(symbol, source, interval, str(path), 0, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, _expected_minutes(interval), None, None, "fail", str(exc))

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        return FileReport(symbol, source, interval, str(path), len(df), missing_columns, 0, 0, 0, 0, 0, 0, 0, 0, 0, _expected_minutes(interval), None, None, "fail", None)

    work = df.copy()
    work = work.sort_values("begin").reset_index(drop=True)

    nan_rows = int(work[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    duplicate_begin = int(work["begin"].duplicated().sum())
    invalid_begin_rows = int(work["begin"].isna().sum())

    non_monotonic_begin = 0
    if len(work) > 1:
        diffs = work["begin"].diff().dt.total_seconds()
        non_monotonic_begin = int((diffs.fillna(1) <= 0).sum() - 1)
        non_monotonic_begin = max(non_monotonic_begin, 0)
    else:
        diffs = pd.Series(dtype=float)

    invalid_price_rows = int(((work[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
    invalid_volume_rows = int((work["volume"] < 0).sum())
    invalid_ohlc_rows = int(((work["high"] < work[["open", "close", "low"]].max(axis=1)) | (work["low"] > work[["open", "close", "high"]].min(axis=1)) | (work["high"] < work["low"])).sum())
    zero_range_rows = int((work["high"] == work["low"]).sum())

    tf_expected = _expected_minutes(interval)
    timeframe_mode_minutes = _infer_minutes(diffs.dropna())

    unexpected_gap_rows = 0
    if tf_expected and not diffs.dropna().empty:
        step = tf_expected * 60
        gap_mask = (diffs.dropna() % step) != 0
        unexpected_gap_rows = int(gap_mask.sum())

    tz = None
    try:
        tz = str(work["begin"].dt.tz)
    except Exception:
        tz = None

    status = "ok"
    if any([missing_columns, nan_rows, duplicate_begin, non_monotonic_begin, invalid_begin_rows, invalid_price_rows, invalid_volume_rows, invalid_ohlc_rows]):
        status = "fail"
    elif zero_range_rows > 0 or unexpected_gap_rows > 0:
        status = "warn"

    return FileReport(symbol, source, interval, str(path), len(work), missing_columns, nan_rows, duplicate_begin, non_monotonic_begin, invalid_begin_rows, invalid_price_rows, invalid_volume_rows, invalid_ohlc_rows, zero_range_rows, unexpected_gap_rows, tf_expected, timeframe_mode_minutes, tz, status, None)


def _iter_targets(symbols: list[str], intervals: list[str], source: str):
    if source in {"data", "both"}:
        for symbol in symbols:
            path = ROOT / "data" / f"{symbol}.csv"
            if path.exists():
                yield symbol, "data", "raw", path
    if source in {"processed", "both"}:
        for symbol in symbols:
            for interval in intervals:
                path = ROOT / "processed" / f"{symbol}_{interval}.csv"
                if path.exists():
                    yield symbol, "processed", interval, path


def main(args):
    symbols = load_symbols(args.symbols)
    intervals = args.intervals
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for symbol, source, interval, path in _iter_targets(symbols, intervals, args.source):
        reports.append(asdict(_validate_file(symbol, source, interval, path)))

    if not reports:
        raise SystemExit("[data-sanity] no files found for requested symbols/source")

    summary_df = pd.DataFrame(reports).sort_values(["status", "source", "symbol", "interval"])
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    issues_df = summary_df[summary_df["status"] != "ok"].copy()
    issues_df.to_csv(out_dir / "issues.csv", index=False)

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    counts = summary_df["status"].value_counts().to_dict()
    print(f"[data-sanity] checked_files={len(summary_df)} out_dir={out_dir}")
    print(f"[data-sanity] status_counts={counts}")
    return {"checked_files": int(len(summary_df)), "status_counts": counts, "out_dir": str(out_dir)}


if __name__ == "__main__":
    raise SystemExit("Run via: python -m app2.cli data-sanity ...")
