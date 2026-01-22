from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .range_v3 import (
    RangeV3Params,
    apply_breakout_logic_v3,
    build_range_box_v3,
    detect_range_segments_v3_with_debug,
    generate_signals_v3_for_segment,
)


def _load_range_config(path: str) -> Dict[str, Any]:
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


def _find_data_path(symbol: str, interval: str) -> str:
    fname_processed = os.path.join("processed", f"{symbol}_{interval}.csv")
    fname_data = os.path.join("data", f"{symbol}.csv")
    if os.path.exists(fname_processed):
        return fname_processed
    if os.path.exists(fname_data):
        return fname_data
    raise FileNotFoundError(f"Cannot find data for {symbol}: tried {fname_processed} and {fname_data}")


def _load_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    path = _find_data_path(symbol, interval)
    df = pd.read_csv(path)

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
        dt_col = df.columns[0]

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    df = df.set_index(dt_col)

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

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Data for {symbol} missing required column '{col}' in {path}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def _filter_date(df: pd.DataFrame, date_str: Optional[str]) -> pd.DataFrame:
    if not date_str:
        return df
    day = pd.to_datetime(date_str).date()
    return df[df.index.date == day]


def run_debug_segments(
    symbol: str,
    interval: str,
    config_path: str,
    date_str: Optional[str] = None,
) -> Dict[str, Any]:
    df = _load_ohlcv(symbol, interval)
    params_cfg = _load_range_config(config_path)
    params = RangeV3Params(params_cfg)

    segments, debug_info = detect_range_segments_v3_with_debug(df, params)
    tradable = [s for s in segments if s.quality in ("AAA", "AA")]

    combined = df.copy()
    combined["v3_signal"] = 0
    combined["v3_L"] = pd.NA
    combined["v3_U"] = pd.NA
    combined["v3_M"] = pd.NA
    combined["v3_segment_quality"] = None
    combined["v3_breakout"] = False

    segments_rows: List[Dict[str, Any]] = []
    used_segments: List[Dict[str, Any]] = []

    for seg in tradable:
        box = build_range_box_v3(df, seg, params)
        if box is None:
            continue
        local = generate_signals_v3_for_segment(df, seg, box, params)
        idx_slice = df.index[seg.start_idx : seg.end_idx + 1]
        for col in ["v3_signal", "v3_L", "v3_U", "v3_M", "v3_segment_quality", "v3_breakout"]:
            combined.loc[idx_slice, col] = local.loc[idx_slice, col]
        segments_rows.append(
            {
                "start_dt": str(df.index[seg.start_idx]),
                "end_dt": str(df.index[seg.end_idx]),
                "quality": seg.quality,
                "L": float(box[0]),
                "U": float(box[1]),
                "H": float(box[2]),
                "M": float(box[3]),
            }
        )
        used_segments.append(
            {"start": int(seg.start_idx), "end": int(seg.end_idx), "quality": seg.quality}
        )

    combined = apply_breakout_logic_v3(combined, params)
    combined_filtered = _filter_date(combined, date_str)

    debug_info["segments_tradable"] = len(tradable)
    debug_info["segments_used"] = used_segments
    debug_info["symbol"] = symbol
    debug_info["interval"] = interval
    debug_info["date"] = date_str

    segments_df = pd.DataFrame(segments_rows)
    return {
        "segments_df": segments_df,
        "snapshots_df": combined_filtered,
        "debug_info": debug_info,
    }


def main(args):
    result = run_debug_segments(
        symbol=args.symbol,
        interval=args.interval,
        config_path=args.config_range,
        date_str=args.date,
    )
    out_dir = os.path.dirname(args.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    segments_path = f"{args.out_prefix}_segments.csv"
    snapshots_path = f"{args.out_prefix}_snapshots.csv"
    debug_path = f"{args.out_prefix}_debug.json"

    result["segments_df"].to_csv(segments_path, index=False)
    result["snapshots_df"].to_csv(snapshots_path)
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(result["debug_info"], f, ensure_ascii=False, indent=2)

    print(f"[range-debug-segments] saved segments to {segments_path}")
    print(f"[range-debug-segments] saved snapshots to {snapshots_path}")
    print(f"[range-debug-segments] saved debug to {debug_path}")
    return result
