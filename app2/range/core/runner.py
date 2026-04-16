import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..range_v3 import RangeV3Params
from .contracts import TradeRecord as Trade
from ...parallel import parallel_map

from .data_io import (
    _load_range_config,
    _list_available_symbols,
    _load_ohlcv,
)
from .engine import run_core_for_symbol
from .entry_gating import apply_entry_ai_filter
from .execution import run_trades_from_signals
from .reporting import (
    save_symbol_outputs,
    save_portfolio_outputs,
    trades_to_df,
)


def _parse_list(value: str | None) -> List[str]:
    """Utility to parse comma-separated CLI values into a list of strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def run_symbol_task(
    args: Tuple[
        str,
        str,
        float,
        RangeV3Params,
        str,
        str,
        str,
        str,
        float,
        float,
        str,
        str,
        float,
        float,
        float,
        bool,
        List[str],
    ],
) -> Tuple[str, Optional[Dict[str, Any]], pd.DataFrame]:
    """Run backtest for a single symbol and persist per-symbol artifacts.

    This function encapsulates the per-symbol workflow: loading data,
    generating signals, optional AI gating, executing trades, computing
    metrics and persisting outputs. It is designed to be used with
    ``parallel_map`` to process multiple symbols in parallel.

    Returns a tuple of (symbol, metrics dict or None, trades DataFrame).
    If data for the symbol is not found, metrics will be None and the
    trades DataFrame will be empty.
    """
    (
        symbol,
        interval,
        equity0,
        params,
        out_prefix,
        tag,
        entry_model_path,
        entry_model_mode,
        entry_model_threshold,
        entry_model_top_pct,
        entry_model_trend_path,
        entry_model_trend_mode,
        entry_model_trend_threshold,
        entry_model_trend_top_pct,
        entry_trend_slope_k,
        no_hold_weekend,
        entry_feature_cols,
    ) = args
    try:
        df = _load_ohlcv(symbol, interval)
    except FileNotFoundError:
        # No data available: skip this symbol
        return symbol, None, pd.DataFrame()

    # Run the core engine to get signals and debug information
    sig_df, debug_info = run_core_for_symbol(df, params)
    # Optional AI-based entry gating
    entry_allow_mask, entry_ai_stats = apply_entry_ai_filter(
        sig_df,
        params,
        entry_model_path,
        entry_model_mode,
        entry_model_threshold,
        entry_model_top_pct,
        entry_model_trend_path,
        entry_model_trend_mode,
        entry_model_trend_threshold,
        entry_model_trend_top_pct,
        entry_trend_slope_k,
        entry_feature_cols,
    )
    # Execute trades and compute per-symbol metrics
    trades, metrics = run_trades_from_signals(
        symbol,
        sig_df,
        params,
        equity0,
        entry_allow_mask=entry_allow_mask,
        no_hold_weekend=no_hold_weekend,
    )
    # Merge AI gating statistics into metrics, if any
    if entry_ai_stats:
        metrics.update(entry_ai_stats)
    # Persist per-symbol outputs: stats, trades, snapshots and debug information
    save_symbol_outputs(
        out_prefix=out_prefix,
        symbol=symbol,
        interval=interval,
        tag=tag,
        metrics=metrics,
        trades=trades,
        debug_info=debug_info,
        sig_df=sig_df,
    )
    trades_df = trades_to_df(trades)

    print(
        f"[range-v3] {symbol}: trades={len(trades)} "
        f"pf={metrics['pf']:.3f} win_rate={metrics['win_rate']:.3f}"
    )
    return symbol, metrics, trades_df


def run_range_backtest(args) -> None:
    """Main orchestrator for running range-v3 backtests across symbols.

    This function replicates the orchestration previously implemented in
    ``backtest.py``. It builds a list of per-symbol tasks, executes them
    (optionally in parallel), aggregates per-symbol metrics and trades,
    and finally persists portfolio-level outputs.
    """
    # Unpack CLI arguments
    symbols: List[str] = list(args.symbols)
    interval: str = args.interval
    equity0: float = float(args.equity0)
    cfg_path: str = args.config_range
    out_prefix: str = args.out_prefix
    tag: str = getattr(args, "tag", "rangeV3")
    n_jobs = getattr(args, "n_jobs", None)
    entry_model_path = str(getattr(args, "entry_model_path", "") or "")
    entry_model_mode = str(getattr(args, "entry_model_mode", "off"))
    entry_model_threshold = float(getattr(args, "entry_model_threshold", 0.5))
    entry_model_top_pct = float(getattr(args, "entry_model_top_pct", 0.3))
    entry_model_trend_path = str(getattr(args, "entry_model_trend_path", "") or "")
    entry_model_trend_mode = str(getattr(args, "entry_model_trend_mode", "off"))
    entry_model_trend_threshold = float(getattr(args, "entry_model_trend_threshold", 0.5))
    entry_model_trend_top_pct = float(getattr(args, "entry_model_trend_top_pct", 0.3))
    entry_trend_slope_k = float(getattr(args, "entry_trend_slope_k", 0.0))
    no_hold_weekend = bool(getattr(args, "no_hold_weekend", False))
    entry_feature_cols = _parse_list(getattr(args, "entry_feature_include", ""))
    if not entry_feature_cols:
        from ..baseline_ml import COMPACT_FEATURES as _DEFAULT_FEATURES
        entry_feature_cols = _DEFAULT_FEATURES

    # If the entry model path points to an artifact (JSON), validate and load it.
    # This ensures train/inference parity by enforcing feature schema, truth policy,
    # dataset kind and config fingerprint matching.
    if entry_model_path and entry_model_path.lower().endswith(".json"):
        try:
            # Defer heavy imports until needed
            from .artifact_validator import validate_inference_artifact
            from .inference_artifacts import InferenceArtifact
            import json

            # Validate the artifact against expected dataset kind and config.
            # We treat the current feature set as the expected set.
            validate_inference_artifact(
                artifact_path=entry_model_path,
                expected_dataset_kind="entry",
                expected_features=entry_feature_cols,
                expected_config_path=cfg_path,
            )
            # Load the artifact JSON and construct an instance
            with open(entry_model_path, "r", encoding="utf-8") as f:
                art_dict = json.load(f)
            art = InferenceArtifact(**art_dict)
            # Override model-related parameters with values from the artifact
            entry_model_path = art.model_path
            # Only override threshold if present; otherwise keep CLI-specified value
            if art.threshold is not None:
                entry_model_threshold = float(art.threshold)
            # Replace feature columns with artifact features; this ensures order
            if art.features:
                entry_feature_cols = list(art.features)
        except Exception as e:
            # Propagate validation errors with context
            raise RuntimeError(f"Invalid inference artifact: {e}")

    # Load range configuration and instantiate parameters
    params_cfg = _load_range_config(cfg_path)
    params = RangeV3Params(params_cfg)

    # Ensure output directory exists
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Expand 'all' symbols to the full list available for the interval
    if any(str(s).lower() == "all" for s in symbols):
        symbols = _list_available_symbols(interval)

    all_symbol_metrics: List[Dict[str, Any]] = []
    all_trades_df_list: List[pd.DataFrame] = []

    # Derive fallback slope_k for trend if not provided
    if entry_trend_slope_k <= 0:
        entry_trend_slope_k = float(getattr(params, "slope_k", 0.0) or 0.0) * 0.5

    # Construct per-symbol tasks
    tasks = [
        (
            symbol,
            interval,
            equity0,
            params,
            out_prefix,
            tag,
            entry_model_path,
            entry_model_mode,
            entry_model_threshold,
            entry_model_top_pct,
            entry_model_trend_path,
            entry_model_trend_mode,
            entry_model_trend_threshold,
            entry_model_trend_top_pct,
            entry_trend_slope_k,
            no_hold_weekend,
            entry_feature_cols,
        )
        for symbol in symbols
    ]

    # Execute tasks in parallel (or sequentially if n_jobs=1)
    for symbol, metrics, trades_df in parallel_map(tasks, run_symbol_task, n_jobs=n_jobs):
        if metrics is None:
            continue
        all_symbol_metrics.append(metrics)
        if not trades_df.empty:
            all_trades_df_list.append(trades_df)

    # Persist aggregated portfolio-level outputs
    if all_trades_df_list:
        all_trades_df = pd.concat(all_trades_df_list, ignore_index=True)
        save_portfolio_outputs(
            out_prefix=out_prefix,
            interval=interval,
            tag=tag,
            equity0=equity0,
            all_symbol_metrics=all_symbol_metrics,
            all_trades_df=all_trades_df,
        )

    return None