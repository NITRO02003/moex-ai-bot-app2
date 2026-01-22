import os
import json

import pandas as pd

from dataclasses import fields

from .rule_strategies import (
    generate_trend_signals,
    generate_meanrev_signals,
    generate_breakout_signals,
    generate_meanrev_v2_signals,
    TrendParams,
)
from .rule_core import run_rule_symbol, RuleBtParams
from .parallel import parallel_map
from .utils import load_symbols, save_json
from .config import load_config


def _load_prices_for_backtest(sym: str, interval: str = "30min"):
    """Унифицированная загрузка данных для rule-backtest.

    Приоритет:
      1) processed/{sym}_{interval}.csv
      2) data/{sym}_{interval}.csv
      3) data/{sym}.csv
    """
    candidates = [
        os.path.join("processed", f"{sym}_{interval}.csv"),
        os.path.join("data", f"{sym}_{interval}.csv"),
        os.path.join("data", f"{sym}.csv"),
    ]

    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        print(f"[rule-backtest] {sym}: no data file found, skip")
        return None

    df = pd.read_csv(path)

    # выбрасываем бары без цены, чтобы не ломать equity/PnL
    if "close" in df.columns:
        df = df[~df["close"].isna()].copy()

    # определяем временную колонку
    dt_col = None
    if "datetime" in df.columns:
        dt_col = "datetime"
    elif "begin" in df.columns:
        dt_col = "begin"

    if dt_col is None:
        print(f"[rule-backtest] {sym}: no datetime/begin column in {path}, skip")
        return None

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    # приводим к единому имени datetime
    if dt_col != "datetime":
        df["datetime"] = df[dt_col]

    return df


def _count_entries(sig: pd.Series) -> int:
    if sig is None:
        return 0
    s = sig.fillna(0)
    prev = s.shift(1).fillna(0)
    entries = (prev == 0) & (s != 0)
    return int(entries.sum())


def _run_rule_symbol_task(
    args: tuple[
        str,
        str,
        str,
        float,
        dict,
        dict,
        dict,
        dict,
        dict,
        dict,
        str | None,
        str | None,
    ],
) -> tuple[str, dict[str, object] | None]:
    (
        sym,
        strategy,
        interval,
        equity0,
        trend_cfg,
        meanrev_cfg,
        meanrev2_cfg,
        breakout_cfg,
        regime_cfg,
        bt_cfg,
        regime_segments_path,
        regime_filter,
    ) = args
    df = _load_prices_for_backtest(sym, interval=interval)
    if df is None:
        return sym, None

    # --- generate raw strategy signals ---
    if strategy == "trend":
        trend_params = TrendParams(**trend_cfg)
        trend_df = generate_trend_signals(df, trend_params)
        df["signal"] = trend_df["signal"]
    elif strategy == "meanrev":
        df["signal"] = generate_meanrev_signals(df, **meanrev_cfg)
    elif strategy == "meanrev_v2":
        side, z_score, regime = generate_meanrev_v2_signals(
            df,
            regime_params=regime_cfg,
            **meanrev2_cfg,
        )
        df["signal"] = side
        df["z_score"] = z_score
        df["regime"] = regime
    elif strategy == "breakout":
        df["signal"] = generate_breakout_signals(df, **breakout_cfg)
    else:
        raise ValueError("Invalid strategy")

    # --- diagnostics before regime filter ---
    bars_total = int(len(df))
    entries_raw = _count_entries(df.get("signal")) if "signal" in df.columns else 0

    # --- optional regime-based filtering of signals ---
    regime_mask = None
    if regime_segments_path and regime_filter:
        bars_in_regime = 0
        entries_after_regime_filter = entries_raw
        try:
            seg_df = pd.read_csv(regime_segments_path)
        except Exception as e:
            print(
                f"[rule-backtest] {sym}: failed to read regime segments "
                f"{regime_segments_path}: {e}"
            )
        else:
            seg_sym = seg_df
            if "symbol" in seg_sym.columns:
                seg_sym = seg_sym[seg_sym["symbol"] == sym]
            seg_sym = seg_sym[seg_sym["regime"] == regime_filter]

            if not seg_sym.empty and "datetime" in df.columns:
                seg_sym = seg_sym.copy()
                seg_sym["start_dt"] = pd.to_datetime(seg_sym["start_dt"], utc=True)
                seg_sym["end_dt"] = pd.to_datetime(seg_sym["end_dt"], utc=True)

                dt = pd.to_datetime(df["datetime"], utc=True)
                mask = pd.Series(False, index=df.index)
                for _, row in seg_sym.iterrows():
                    mask |= (dt >= row["start_dt"]) & (dt <= row["end_dt"])

                bars_in_regime = int(mask.sum())
                regime_mask = mask

                if "signal" in df.columns:
                    df.loc[~mask, "signal"] = 0
                    entries_after_regime_filter = _count_entries(df["signal"])
            else:
                bars_in_regime = 0
                entries_after_regime_filter = 0
    else:
        bars_in_regime = bars_total
        entries_after_regime_filter = entries_raw
        regime_segments_path = None
        regime_filter = None

    allowed_keys = {f.name for f in fields(RuleBtParams)}
    bt_cfg_filtered = {k: v for k, v in bt_cfg.items() if k in allowed_keys}
    params = RuleBtParams(**bt_cfg_filtered)

    result = run_rule_symbol(df, params, equity0, regime_mask=regime_mask)
    metrics = result.get("metrics", {})
    metrics["bars_total"] = bars_total
    metrics["bars_in_regime"] = bars_in_regime
    metrics["entries_raw"] = entries_raw
    metrics["entries_after_regime_filter"] = entries_after_regime_filter
    metrics["regime_filter"] = regime_filter
    metrics["regime_segments_path"] = regime_segments_path
    return sym, metrics


def main(args):
    symbols = load_symbols(args.symbols)
    strategy = args.strategy
    interval = args.interval
    config = load_config()
    out_path = args.out
    n_jobs = getattr(args, "n_jobs", None)

    defaults = config.get("defaults", {}) or {}
    trend_cfg = defaults.get("TrendParams", {}) or {}
    meanrev_cfg = defaults.get("MeanRevParams", {}) or {}
    meanrev2_cfg = defaults.get("MeanRevV2Params", {}) or {}
    breakout_cfg = defaults.get("BreakoutParams", {}) or {}
    regime_cfg = defaults.get("RegimeParams", {}) or {}
    bt_cfg = defaults.get("RuleBtParams", {}) or {}
    regime_segments_path = getattr(args, "regime_segments", None)
    regime_filter = getattr(args, "regime_filter", None)

    tasks = [
        (
            sym,
            strategy,
            interval,
            float(args.equity0),
            trend_cfg,
            meanrev_cfg,
            meanrev2_cfg,
            breakout_cfg,
            regime_cfg,
            bt_cfg,
            regime_segments_path,
            regime_filter,
        )
        for sym in symbols
    ]
    results = {}
    for sym, metrics in parallel_map(tasks, _run_rule_symbol_task, n_jobs=n_jobs):
        if metrics is not None:
            results[sym] = metrics

    if out_path:
        save_json(results, out_path)
    else:
        print(json.dumps(results, indent=2))
