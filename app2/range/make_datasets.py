from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..parallel import parallel_map

from .dataset import (
    make_entry_candidates,
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


def _load_range_params(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("RangeV3", {}).get("params", {})


def _build_entry_for_symbol(
    args: tuple[str, str, str, str],
) -> pd.DataFrame:
    sym, interval, out_prefix, tag = args
    artifacts = _load_symbol_artifacts(out_prefix, sym, interval, tag)
    if artifacts is None:
        return pd.DataFrame()
    trades_norm = v3_trades_to_frame(artifacts["trades"], symbol=sym)
    feats = _prepare_snapshot_features(artifacts["snapshots"])
    snaps = make_entry_snapshots(features_df=feats, trades_df=trades_norm)
    return snaps if not snaps.empty else pd.DataFrame()


def _build_entry_candidates_for_symbol(
    args: tuple[str, str, str, str, float, float, int, float, str, float, float, float, bool],
) -> pd.DataFrame:
    (
        sym,
        interval,
        out_prefix,
        tag,
        entry_zone_alpha,
        shadow_pct,
        horizon_bars,
        return_threshold,
        label_mode,
        mfe_threshold,
        mae_threshold,
        entry_quantile,
        quantile_drop_middle,
    ) = args
    artifacts = _load_symbol_artifacts(out_prefix, sym, interval, tag)
    if artifacts is None:
        return pd.DataFrame()
    feats = _prepare_snapshot_features(artifacts["snapshots"])
    rows = make_entry_candidates(
        features_df=feats,
        interval=interval,
        entry_zone_alpha=entry_zone_alpha,
        shadow_pct=shadow_pct,
        horizon_bars=horizon_bars,
        return_threshold=return_threshold,
        label_mode=label_mode,
        mfe_threshold=mfe_threshold,
        mae_threshold=mae_threshold,
        quantile=entry_quantile,
        drop_middle=quantile_drop_middle,
    )
    if not rows.empty and "symbol" not in rows.columns:
        rows["symbol"] = sym
    return rows if not rows.empty else pd.DataFrame()


def _build_intrade_for_symbol(
    args: tuple[str, str, str, str, float, int],
) -> pd.DataFrame:
    sym, interval, out_prefix, tag, exit_improve_threshold, exit_min_bars = args
    artifacts = _load_symbol_artifacts(out_prefix, sym, interval, tag)
    if artifacts is None:
        return pd.DataFrame()
    trades_norm = v3_trades_to_frame(artifacts["trades"], symbol=sym)
    feats = _prepare_snapshot_features(artifacts["snapshots"])
    rows = make_intrade_timeseries(
        features_df=feats,
        trades_df=trades_norm,
        interval=interval,
        exit_improve_threshold=exit_improve_threshold,
        exit_min_bars=exit_min_bars,
    )
    return rows if not rows.empty else pd.DataFrame()


def build_entry_dataset(
    symbols: List[str],
    interval: str,
    out_prefix: str,
    tag: str,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    tasks = [(sym, interval, out_prefix, tag) for sym in symbols]
    all_snaps = [df for df in parallel_map(tasks, _build_entry_for_symbol, n_jobs=n_jobs) if not df.empty]
    if not all_snaps:
        return pd.DataFrame()
    return pd.concat(all_snaps, ignore_index=True)


def build_entry_candidates_dataset(
    symbols: List[str],
    interval: str,
    out_prefix: str,
    tag: str,
    entry_zone_alpha: float,
    shadow_pct: float,
    horizon_bars: int,
    return_threshold: float,
    label_mode: str,
    mfe_threshold: float,
    mae_threshold: float,
    entry_quantile: float,
    quantile_drop_middle: bool,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    tasks = [
        (
            sym,
            interval,
            out_prefix,
            tag,
            entry_zone_alpha,
            shadow_pct,
            horizon_bars,
            return_threshold,
            label_mode,
            mfe_threshold,
            mae_threshold,
            entry_quantile,
            quantile_drop_middle,
        )
        for sym in symbols
    ]
    all_rows = [df for df in parallel_map(tasks, _build_entry_candidates_for_symbol, n_jobs=n_jobs) if not df.empty]
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def build_intrade_dataset(
    symbols: List[str],
    interval: str,
    out_prefix: str,
    tag: str,
    exit_improve_threshold: float = 0.0,
    exit_min_bars: int = 0,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    tasks = [
        (sym, interval, out_prefix, tag, exit_improve_threshold, exit_min_bars)
        for sym in symbols
    ]
    all_rows = [df for df in parallel_map(tasks, _build_intrade_for_symbol, n_jobs=n_jobs) if not df.empty]
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main(args):
    symbols = list(args.symbols)
    interval = args.interval
    out_prefix = args.out_prefix
    tag = args.tag
    mode = getattr(args, "mode", "both")
    entry_mode = getattr(args, "entry_mode", "trades")
    exit_improve_threshold = float(getattr(args, "exit_improve_threshold", 0.0))
    exit_min_bars = int(getattr(args, "exit_min_bars", 0))
    config_range = getattr(args, "config_range", "app2/range/config.json")
    entry_horizon_bars = int(getattr(args, "entry_horizon_bars", 6))
    entry_return_threshold = float(getattr(args, "entry_return_threshold", 0.001))
    entry_label_mode = str(getattr(args, "entry_label_mode", "ret"))
    entry_mfe_threshold = float(getattr(args, "entry_mfe_threshold", 0.0))
    entry_mae_threshold = float(getattr(args, "entry_mae_threshold", 0.0))
    entry_quantile = float(getattr(args, "entry_quantile", 0.3))
    entry_quantile_drop_middle = bool(getattr(args, "entry_quantile_drop_middle", False))
    n_jobs = getattr(args, "n_jobs", None)

    range_params = _load_range_params(config_range)
    entry_zone_alpha = float(range_params.get("entry_zone_alpha", 0.2))
    shadow_pct = float(range_params.get("shadow_pct", 0.005))

    entry_df = pd.DataFrame()
    entry_candidates_df = pd.DataFrame()
    intrade_df = pd.DataFrame()
    if mode in ("entry", "both"):
        if entry_mode in ("trades", "both"):
            entry_df = build_entry_dataset(symbols, interval, out_prefix, tag, n_jobs=n_jobs)
        if entry_mode in ("candidates", "both"):
            entry_candidates_df = build_entry_candidates_dataset(
                symbols,
                interval,
                out_prefix,
                tag,
                entry_zone_alpha=entry_zone_alpha,
                shadow_pct=shadow_pct,
                horizon_bars=entry_horizon_bars,
                return_threshold=entry_return_threshold,
                label_mode=entry_label_mode,
                mfe_threshold=entry_mfe_threshold,
                mae_threshold=entry_mae_threshold,
                entry_quantile=entry_quantile,
                quantile_drop_middle=entry_quantile_drop_middle,
                n_jobs=n_jobs,
            )
    if mode in ("intrade", "both"):
        intrade_df = build_intrade_dataset(
            symbols,
            interval,
            out_prefix,
            tag,
            exit_improve_threshold=exit_improve_threshold,
            exit_min_bars=exit_min_bars,
            n_jobs=n_jobs,
        )
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    if mode in ("entry", "both") and entry_mode in ("trades", "both"):
        entry_path = f"{out_prefix}_entry_snapshots.csv"
        entry_df.to_csv(entry_path, index=False)

        # Compose a richer manifest for the entry snapshots dataset.  This
        # includes the dataset kind (entry), truth policy (trades) and
        # optionally the original range config path.  Adding these fields
        # clarifies the intended use of the dataset and supports downstream
        # validation in training/inference pipelines.
        meta = {
            "dataset_kind": "entry",
            "truth_policy": "trades",
            "symbols": symbols,
            "interval": interval,
            "out_prefix": out_prefix,
            "tag": tag,
            "entry_rows": int(len(entry_df)),
            "entry_mode": "trades",
            "config_path": config_range,
        }
        meta_path = f"{out_prefix}_entry_snapshots_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[range-v3-make-datasets] saved entry dataset to {entry_path}")
        print(f"[range-v3-make-datasets] saved meta to {meta_path}")

    if mode in ("entry", "both") and entry_mode in ("candidates", "both"):
        entry_path = f"{out_prefix}_entry_candidates.csv"
        entry_candidates_df.to_csv(entry_path, index=False)

        # Compose a richer manifest for the entry candidates dataset.  The
        # "truth_policy" field identifies this as a research dataset.  The
        # additional fields capture the labeling configuration used during
        # generation (horizon, threshold, label_mode, quantiles, etc.).
        meta = {
            "dataset_kind": "entry",
            "truth_policy": "candidates",
            "symbols": symbols,
            "interval": interval,
            "out_prefix": out_prefix,
            "tag": tag,
            "entry_rows": int(len(entry_candidates_df)),
            "entry_mode": "candidates",
            "entry_horizon_bars": entry_horizon_bars,
            "entry_return_threshold": entry_return_threshold,
            "entry_label_mode": entry_label_mode,
            "entry_mfe_threshold": entry_mfe_threshold,
            "entry_mae_threshold": entry_mae_threshold,
            "entry_quantile": entry_quantile,
            "entry_quantile_drop_middle": entry_quantile_drop_middle,
            "entry_zone_alpha": entry_zone_alpha,
            "shadow_pct": shadow_pct,
            "config_path": config_range,
        }
        meta_path = f"{out_prefix}_entry_candidates_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[range-v3-make-datasets] saved entry candidates to {entry_path}")
        print(f"[range-v3-make-datasets] saved meta to {meta_path}")

    if mode in ("intrade", "both"):
        intrade_path = f"{out_prefix}_intrade_timeseries.csv"
        intrade_df.to_csv(intrade_path, index=False)

        # Manifest for intrade timeseries includes dataset kind and truth policy.
        meta = {
            "dataset_kind": "intrade",
            "truth_policy": "trades",
            "symbols": symbols,
            "interval": interval,
            "out_prefix": out_prefix,
            "tag": tag,
            "intrade_rows": int(len(intrade_df)),
            "exit_improve_threshold": exit_improve_threshold,
            "exit_min_bars": exit_min_bars,
            "config_path": config_range,
        }
        meta_path = f"{out_prefix}_intrade_timeseries_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[range-v3-make-datasets] saved intrade dataset to {intrade_path}")
        print(f"[range-v3-make-datasets] saved meta to {meta_path}")

    return {"entry": entry_df, "intrade": intrade_df}
