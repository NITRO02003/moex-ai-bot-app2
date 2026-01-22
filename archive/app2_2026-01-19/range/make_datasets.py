from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .dataset import (
    make_entry_snapshots,
    make_intrade_timeseries,
    snapshots_to_features,
    v3_trades_to_frame,
)
from .feature_sweep import add_derived_features
from .features import add_basic_range_features


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _build_base(out_prefix: str, symbol: str, interval: str, tag: str) -> str:
    return f"{out_prefix}_{symbol}_{interval}_{tag}"


def _load_symbol_artifacts(
    out_prefix: str, symbol: str, interval: str, tag: str
) -> Optional[Dict[str, pd.DataFrame]]:
    base = _build_base(out_prefix, symbol, interval, tag)
    trades_path = f"{base}_trades.csv"
    snaps_path = f"{base}_snapshots.csv"
    if not os.path.exists(trades_path) or not os.path.exists(snaps_path):
        return None
    trades_df = _read_csv(trades_path)
    snaps_df = _read_csv(snaps_path)
    return {"trades": trades_df, "snapshots": snaps_df}


def _prepare_snapshot_features(snaps_df: pd.DataFrame) -> pd.DataFrame:
    feats = snapshots_to_features(snaps_df)
    required = {"open", "high", "low", "close"}
    if required.issubset(feats.columns):
        feats = add_basic_range_features(feats, ma_len=20, ma_mode="ema")
    feats = add_derived_features(feats)
    return feats


def build_entry_dataset(
    symbols: List[str],
    interval: str,
    out_prefix: str,
    tag: str,
) -> pd.DataFrame:
    all_snaps: List[pd.DataFrame] = []
    for sym in symbols:
        artifacts = _load_symbol_artifacts(out_prefix, sym, interval, tag)
        if artifacts is None:
            continue
        trades_norm = v3_trades_to_frame(artifacts["trades"], symbol=sym)
        feats = _prepare_snapshot_features(artifacts["snapshots"])
        snaps = make_entry_snapshots(features_df=feats, trades_df=trades_norm)
        if not snaps.empty:
            all_snaps.append(snaps)
    if not all_snaps:
        return pd.DataFrame()
    return pd.concat(all_snaps, ignore_index=True)


def build_intrade_dataset(
    symbols: List[str],
    interval: str,
    out_prefix: str,
    tag: str,
    exit_improve_threshold: float = 0.0,
    exit_min_bars: int = 0,
) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []
    for sym in symbols:
        artifacts = _load_symbol_artifacts(out_prefix, sym, interval, tag)
        if artifacts is None:
            continue
        trades_norm = v3_trades_to_frame(artifacts["trades"], symbol=sym)
        feats = _prepare_snapshot_features(artifacts["snapshots"])
        rows = make_intrade_timeseries(
            features_df=feats,
            trades_df=trades_norm,
            interval=interval,
            exit_improve_threshold=exit_improve_threshold,
            exit_min_bars=exit_min_bars,
        )
        if not rows.empty:
            all_rows.append(rows)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main(args):
    symbols = list(args.symbols)
    interval = args.interval
    out_prefix = args.out_prefix
    tag = args.tag
    mode = getattr(args, "mode", "both")
    exit_improve_threshold = float(getattr(args, "exit_improve_threshold", 0.0))
    exit_min_bars = int(getattr(args, "exit_min_bars", 0))

    entry_df = pd.DataFrame()
    intrade_df = pd.DataFrame()
    if mode in ("entry", "both"):
        entry_df = build_entry_dataset(symbols, interval, out_prefix, tag)
    if mode in ("intrade", "both"):
        intrade_df = build_intrade_dataset(
            symbols,
            interval,
            out_prefix,
            tag,
            exit_improve_threshold=exit_improve_threshold,
            exit_min_bars=exit_min_bars,
        )
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    if mode in ("entry", "both"):
        entry_path = f"{out_prefix}_entry_snapshots.csv"
        entry_df.to_csv(entry_path, index=False)

        meta = {
            "symbols": symbols,
            "interval": interval,
            "out_prefix": out_prefix,
            "tag": tag,
            "entry_rows": int(len(entry_df)),
        }
        meta_path = f"{out_prefix}_entry_snapshots_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[range-v3-make-datasets] saved entry dataset to {entry_path}")
        print(f"[range-v3-make-datasets] saved meta to {meta_path}")

    if mode in ("intrade", "both"):
        intrade_path = f"{out_prefix}_intrade_timeseries.csv"
        intrade_df.to_csv(intrade_path, index=False)

        meta = {
            "symbols": symbols,
            "interval": interval,
            "out_prefix": out_prefix,
            "tag": tag,
            "intrade_rows": int(len(intrade_df)),
            "exit_improve_threshold": exit_improve_threshold,
            "exit_min_bars": exit_min_bars,
        }
        meta_path = f"{out_prefix}_intrade_timeseries_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[range-v3-make-datasets] saved intrade dataset to {intrade_path}")
        print(f"[range-v3-make-datasets] saved meta to {meta_path}")

    return {"entry": entry_df, "intrade": intrade_df}
