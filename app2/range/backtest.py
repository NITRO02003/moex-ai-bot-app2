from __future__ import annotations

import os
import json
from dataclasses import fields
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import load_config
from ..parallel import parallel_map
from ..rule_backtest import _load_prices_for_backtest
from ..rule_core import RuleBtParams, run_rule_symbol
from ..utils import load_symbols, save_json

from .strategy import RangeParams, generate_range_signals
from .features import add_basic_range_features
from .dataset import trades_to_frame, make_entry_snapshots


def _build_range_params(cfg: Optional[Dict[str, Any]]) -> RangeParams:
    cfg = cfg or {}
    allowed = {f.name for f in fields(RangeParams)}
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    return RangeParams(**filtered)


def _build_bt_params(defaults: Dict[str, Any]) -> RuleBtParams:
    bt_cfg = (defaults.get("RuleBtParams") or {}) if defaults else {}
    allowed = {f.name for f in fields(RuleBtParams)}
    filtered = {k: v for k, v in bt_cfg.items() if k in allowed}
    return RuleBtParams(**filtered)


def _count_entries(sig: Optional[pd.Series]) -> int:
    if sig is None:
        return 0
    s = sig.fillna(0)
    prev = s.shift(1).fillna(0)
    entries = (prev == 0) & (s != 0)
    return int(entries.sum())


def _run_range_symbol_task(
    args: tuple[
        str,
        str,
        float,
        RangeParams,
        RuleBtParams,
        str | None,
        str | None,
        str | None,
        bool,
        bool,
    ],
) -> tuple[str, Dict[str, Any] | None, pd.DataFrame, pd.DataFrame]:
    (
        sym,
        interval,
        equity0,
        range_params,
        bt_params,
        regime_segments_path,
        regime_filter,
        profile_name,
        save_trades,
        save_snapshots,
    ) = args

    df = _load_prices_for_backtest(sym, interval=interval)
    if df is None or df.empty:
        print(f"[range-backtest] {sym}: no data for interval={interval}, skip")
        return sym, None, pd.DataFrame(), pd.DataFrame()

    if "datetime" not in df.columns:
        print(f"[range-backtest] {sym}: no datetime column after load, skip")
        return sym, None, pd.DataFrame(), pd.DataFrame()

    df_sig = generate_range_signals(df, params=range_params, regime_mask=None)

    bars_total = int(len(df_sig))
    entries_raw = _count_entries(df_sig.get("signal")) if "signal" in df_sig.columns else 0

    if regime_segments_path and regime_filter:
        bars_in_regime = 0
        entries_after_regime_filter = entries_raw
        try:
            seg_df = pd.read_csv(regime_segments_path)
        except Exception as e:
            print(
                f"[range-backtest] {sym}: failed to read regime segments "
                f"{regime_segments_path}: {e}"
            )
        else:
            seg_sym = seg_df
            if "symbol" in seg_sym.columns:
                seg_sym = seg_sym[seg_sym["symbol"] == sym]
            seg_sym = seg_sym[seg_sym["regime"] == regime_filter]

            if not seg_sym.empty and "datetime" in df_sig.columns:
                seg_sym = seg_sym.copy()
                seg_sym["start_dt"] = pd.to_datetime(seg_sym["start_dt"], utc=True)
                seg_sym["end_dt"] = pd.to_datetime(seg_sym["end_dt"], utc=True)

                dt = pd.to_datetime(df_sig["datetime"], utc=True)
                mask = pd.Series(False, index=df_sig.index)
                for _, row in seg_sym.iterrows():
                    mask |= (dt >= row["start_dt"]) & (dt <= row["end_dt"])

                bars_in_regime = int(mask.sum())

                if "signal" in df_sig.columns:
                    df_sig.loc[~mask, "signal"] = 0
                    entries_after_regime_filter = _count_entries(df_sig["signal"])
            else:
                bars_in_regime = 0
                entries_after_regime_filter = 0
    else:
        bars_in_regime = bars_total
        entries_after_regime_filter = entries_raw
        regime_segments_path = None
        regime_filter = None

    bt_result = run_rule_symbol(
        df_sig,
        bt_params,
        equity0=equity0,
        collect_bar_stats=False,
        collect_trades=True,
    )

    metrics = bt_result.get("metrics", {}) or {}
    metrics["bars_total"] = bars_total
    metrics["bars_in_regime"] = bars_in_regime
    metrics["entries_raw"] = entries_raw
    metrics["entries_after_regime_filter"] = entries_after_regime_filter
    metrics["regime_filter"] = regime_filter
    metrics["regime_segments_path"] = regime_segments_path
    metrics["profile"] = profile_name

    trades_out = pd.DataFrame()
    snaps_out = pd.DataFrame()
    trades_df = bt_result.get("trades")
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and save_trades:
        trades_out = trades_to_frame(trades_df.to_dict("records"), symbol=sym)
        if profile_name:
            trades_out["profile"] = profile_name

        if save_snapshots:
            feats = add_basic_range_features(
                df_sig,
                ma_len=range_params.ma_len,
                band_mult=range_params.band_mult,
                atr_len=range_params.atr_len,
            )
            snaps = make_entry_snapshots(
                features_df=feats,
                trades_df=trades_out,
                feature_cols=None,
                label_cols=None,
            )
            if not snaps.empty:
                snaps_out = snaps
                if profile_name:
                    snaps_out["profile"] = profile_name

    return sym, metrics, trades_out, snaps_out


def main(args) -> Dict[str, Any]:
    """Range regime backtest (long-only).

    Логика по режимному фильтру максимально копирует rule_backtest:
    сначала считаем сигналы, затем (если переданы regime_segments/regime_filter)
    обнуляем сигналы вне нужных сегментов и пересчитываем entries_after_regime_filter.
    """
    symbols = load_symbols(args.symbols)
    interval = args.interval
    equity0 = float(getattr(args, "equity0", 1_000_000.0))

    regime_segments_path = getattr(args, "regime_segments", None)
    regime_filter = getattr(args, "regime_filter", None)

    config_range_path = getattr(args, "config_range", "app2/range/config.json")
    out_prefix = getattr(args, "out_prefix", None)

    # --- range-specific config ---
    range_cfg: Dict[str, Any] = {}
    try:
        with open(config_range_path, "r", encoding="utf-8") as f:
            range_cfg = json.load(f)
        print(f"[range-backtest] loaded range config from {config_range_path}")
    except FileNotFoundError:
        print(f"[range-backtest] {config_range_path} not found, using code defaults")
    except Exception as e:
        print(f"[range-backtest] failed to read {config_range_path}: {e}")

    defaults_cfg = (range_cfg.get("defaults") or {})
    range_params_cfg = (defaults_cfg.get("RangeParams") or {})

    # optional profile from config['profiles']
    profile_name = getattr(args, "profile", None)
    profiles_cfg = (range_cfg.get("profiles") or {})
    profile_cfg = profiles_cfg.get(profile_name) if profile_name else None
    if profile_name and profile_cfg is None:
        print(f"[range-backtest] profile '{profile_name}' not found in config; using defaults")
        profile_name = None
        profile_cfg = None
    if profile_cfg:
        merged_cfg = dict(range_params_cfg)
        merged_cfg.update(profile_cfg)
        range_params_cfg = merged_cfg
        print(f"[range-backtest] applying profile '{profile_name}' overrides")

    range_params = _build_range_params(range_params_cfg)

    logging_cfg = (range_cfg.get("logging") or {})
    save_trades = bool(logging_cfg.get("save_trades", True))
    save_snapshots = bool(logging_cfg.get("save_snapshots", True))

    paths_cfg = (range_cfg.get("paths") or {})
    cfg_out_prefix = paths_cfg.get("out_prefix") or None
    if out_prefix is None:
        out_prefix = cfg_out_prefix

    write_files = bool(out_prefix)

    # --- global config for RuleBtParams ---
    global_cfg = load_config()
    global_defaults = (global_cfg.get("defaults") or {}) if isinstance(global_cfg, dict) else {}
    bt_params = _build_bt_params(global_defaults)

    all_results: Dict[str, Any] = {}
    all_trades: List[pd.DataFrame] = []
    all_snapshots: List[pd.DataFrame] = []

    n_jobs = getattr(args, "n_jobs", None)
    tasks = [
        (
            sym,
            interval,
            equity0,
            range_params,
            bt_params,
            regime_segments_path,
            regime_filter,
            profile_name,
            save_trades,
            save_snapshots,
        )
        for sym in symbols
    ]
    for sym, metrics, trades_df, snaps_df in parallel_map(tasks, _run_range_symbol_task, n_jobs=n_jobs):
        if metrics is None:
            continue
        all_results[sym] = metrics
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
            all_trades.append(trades_df)
        if isinstance(snaps_df, pd.DataFrame) and not snaps_df.empty:
            all_snapshots.append(snaps_df)

    # --- сохранение результатов ---
    if write_files and out_prefix:
        stats_path = f"{out_prefix}_stats.json"
        trades_path = f"{out_prefix}_trades.csv"
        snaps_path = f"{out_prefix}_snapshots.csv"

        if all_results:
            save_json(all_results, stats_path)
            print(f"[range-backtest] saved stats to {stats_path}")

        if save_trades and all_trades:
            trades_cat = pd.concat(all_trades, ignore_index=True)
            out_dir = os.path.dirname(trades_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            trades_cat.to_csv(trades_path, index=False)
            print(f"[range-backtest] saved trades to {trades_path}")

        if save_snapshots and all_snapshots:
            snaps_cat = pd.concat(all_snapshots, ignore_index=True)
            out_dir = os.path.dirname(snaps_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            snaps_cat.to_csv(snaps_path, index=False)
            print(f"[range-backtest] saved entry snapshots to {snaps_path}")
    else:
        # режим "только stdout", без файлов
        print(json.dumps(all_results, ensure_ascii=False, indent=2))

    return all_results
