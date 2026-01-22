
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import pandas as pd

from ..parallel import parallel_map

def _feature_group(name: str) -> str:
    if name in {"dist_from_ma", "band_pos", "z_ma", "edge_proximity"}:
        return "location"
    if name in {"band_width_pct"}:
        return "range_width"
    if name in {"atr_14_pct", "range_vs_atr"}:
        return "volatility"
    if name in {"bar_range_pct", "bar_body_pct", "body_vs_range"}:
        return "bar_shape"
    return "other"


def load_snapshots(glob_pattern: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No snapshot files matched pattern: {glob_pattern!r}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "symbol" not in df.columns:
            base = os.path.basename(p)
            sym = base.split("_", 1)[0]
            df["symbol"] = sym
        if symbols is not None:
            df = df[df["symbol"].isin(symbols)]
            if df.empty:
                continue
        frames.append(df)

    if not frames:
        raise ValueError("No snapshot rows loaded after filtering by symbols")

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Время (entry_dt или bar datetime)
    dt = None
    if "entry_dt" in df.columns:
        dt = pd.to_datetime(df["entry_dt"], errors="coerce", utc=True)
        df["entry_dt"] = dt
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    if dt is not None:
        if "hour" not in df.columns:
            df["hour"] = dt.dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = dt.dt.dayofweek

    # Производные фичи
    ma_close = df.get("ma_close")
    std_close = df.get("std_close")
    band_lower = df.get("band_lower")
    band_upper = df.get("band_upper")
    atr_14_pct = df.get("atr_14_pct")
    bar_range_pct = df.get("bar_range_pct")
    bar_body_pct = df.get("bar_body_pct")

    # z-score от MA
    if ma_close is not None and std_close is not None and "dist_from_ma" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["z_ma"] = df["dist_from_ma"] / std_close.replace(0, np.nan)

    # ширина полос в процентах
    if (
        ma_close is not None
        and band_lower is not None
        and band_upper is not None
    ):
        band_width = band_upper - band_lower
        with np.errstate(divide="ignore", invalid="ignore"):
            df["band_width_pct"] = band_width / ma_close.replace(0, np.nan)

    # близость к границе полос
    if "band_pos" in df.columns:
        bp = df["band_pos"]
        df["edge_proximity"] = np.minimum(bp, 1 - bp)

    # отношение диапазона бара к ATR
    if bar_range_pct is not None and atr_14_pct is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["range_vs_atr"] = bar_range_pct / atr_14_pct.replace(0, np.nan)

    # отношение тела к диапазону
    if bar_body_pct is not None and bar_range_pct is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["body_vs_range"] = bar_body_pct / bar_range_pct.replace(0, np.nan)

    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    n = len(df)
    out["n_trades"] = int(n)
    if n == 0:
        out["pf"] = np.nan
        out["win_rate"] = np.nan
        out["mean_pnl_rel"] = np.nan
        out["pnl_std"] = np.nan
        out["pnl_sum"] = 0.0
        out["mae_mean"] = np.nan
        out["mae_p95"] = np.nan
        out["bars_in_trade_mean"] = np.nan
        return out

    if "pnl_rel" not in df.columns:
        raise KeyError("Column 'pnl_rel' is required in snapshots for metrics computation")

    pnl = pd.to_numeric(df["pnl_rel"], errors="coerce").dropna()
    if pnl.empty:
        out["pf"] = np.nan
        out["win_rate"] = np.nan
        out["mean_pnl_rel"] = np.nan
        out["pnl_std"] = np.nan
        out["pnl_sum"] = 0.0
    else:
        pos = pnl[pnl > 0].sum()
        neg = pnl[pnl < 0].sum()
        if neg == 0:
            out["pf"] = float("inf") if pos > 0 else 0.0
        else:
            out["pf"] = float(pos / abs(neg))
        out["win_rate"] = float((pnl > 0).mean())
        out["mean_pnl_rel"] = float(pnl.mean())
        out["pnl_std"] = float(pnl.std(ddof=0))
        out["pnl_sum"] = float(pnl.sum())

    if "max_adverse_excursion" in df.columns:
        mae = pd.to_numeric(df["max_adverse_excursion"], errors="coerce").dropna()
        out["mae_mean"] = float(mae.mean()) if not mae.empty else np.nan
        out["mae_p95"] = float(mae.quantile(0.95)) if not mae.empty else np.nan
    else:
        out["mae_mean"] = np.nan
        out["mae_p95"] = np.nan

    if "bars_in_trade" in df.columns:
        bit = pd.to_numeric(df["bars_in_trade"], errors="coerce").dropna()
        out["bars_in_trade_mean"] = float(bit.mean()) if not bit.empty else np.nan
    else:
        out["bars_in_trade_mean"] = np.nan

    return out


def _univariate_group_task(
    args: tuple[str, pd.DataFrame, List[str], List[float], int, str],
) -> pd.DataFrame:
    symbol, g, features, quantiles, min_trades_bin, scope = args
    rows: List[Dict[str, Any]] = []
    g = g.copy()
    for feat in features:
        if feat not in g.columns:
            continue

        series = pd.to_numeric(g[feat], errors="coerce").dropna()
        if series.empty:
            continue

        q_vals = sorted(set(quantiles))
        q_vals = [q for q in q_vals if 0.0 <= q <= 1.0]
        if len(q_vals) < 2:
            continue

        quantile_map = {q: float(series.quantile(q)) for q in q_vals}

        bin_index = 0
        for i in range(len(q_vals) - 1):
            q_low = q_vals[i]
            q_high = q_vals[i + 1]
            v_low = quantile_map[q_low]
            v_high = quantile_map[q_high]

            if not np.isfinite(v_low) or not np.isfinite(v_high):
                continue
            if v_high < v_low:
                continue

            mask = (g[feat] >= v_low) & (g[feat] <= v_high)
            bin_df = g[mask]
            if len(bin_df) < min_trades_bin:
                continue

            metrics = compute_metrics(bin_df)
            row: Dict[str, Any] = {
                "scope": scope,
                "symbol": symbol,
                "feature_group": _feature_group(feat),
                "feature": feat,
                "bin_index": bin_index,
                "q_low": q_low,
                "q_high": q_high,
                "value_low": v_low,
                "value_high": v_high,
            }
            row.update(metrics)
            rows.append(row)
            bin_index += 1

        metrics_full = compute_metrics(g)
        row_full: Dict[str, Any] = {
            "scope": scope,
            "symbol": symbol,
            "feature_group": _feature_group(feat),
            "feature": feat,
            "bin_index": -1,
            "q_low": 0.0,
            "q_high": 1.0,
            "value_low": float(series.min()),
            "value_high": float(series.max()),
        }
        row_full.update(metrics_full)
        rows.append(row_full)

    return pd.DataFrame(rows)


def univariate_sweep(
    df_all: pd.DataFrame,
    features: List[str],
    quantiles: List[float],
    min_trades_bin: int,
    mode: str,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if mode not in {"pooled", "per-symbol"}:
        raise ValueError(f"Unknown mode: {mode!r}")

    if mode == "pooled":
        groups = [("ALL", df_all)]
        scope = "pooled"
    else:
        groups = list(df_all.groupby("symbol"))
        scope = "per_symbol"

    if mode == "per-symbol":
        tasks = [(symbol, g, features, quantiles, min_trades_bin, scope) for symbol, g in groups]
        for df_part in parallel_map(tasks, _univariate_group_task, n_jobs=n_jobs):
            if not df_part.empty:
                rows.extend(df_part.to_dict(orient="records"))
    else:
        for symbol, g in groups:
            df_part = _univariate_group_task((symbol, g, features, quantiles, min_trades_bin, scope))
            if not df_part.empty:
                rows.extend(df_part.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    preferred = [
        "scope",
        "symbol",
        "feature_group",
        "feature",
        "bin_index",
        "q_low",
        "q_high",
        "value_low",
        "value_high",
        "n_trades",
        "pf",
        "win_rate",
        "mean_pnl_rel",
        "pnl_std",
        "pnl_sum",
        "mae_mean",
        "mae_p95",
        "bars_in_trade_mean",
    ]
    cols_first = [c for c in preferred if c in df_out.columns]
    other_cols = [c for c in df_out.columns if c not in cols_first]
    df_out = df_out[cols_first + other_cols]
    return df_out


def time_stats_hour(df_all: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df_all.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for hour, g in df_all.groupby("hour"):
        metrics = compute_metrics(g)
        row = {"hour": int(hour)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hour")


def time_stats_dow(df_all: pd.DataFrame) -> pd.DataFrame:
    if "day_of_week" not in df_all.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for dow, g in df_all.groupby("day_of_week"):
        metrics = compute_metrics(g)
        row = {"day_of_week": int(dow)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("day_of_week")


def main(args):
    snapshots_glob: str = args.snapshots_glob
    features: List[str] = list(args.features or [])
    quantiles = [float(q) for q in str(args.quantiles).split(",") if q.strip()]
    min_trades_bin: int = args.min_trades_bin
    mode: str = args.mode
    symbols = args.symbols
    out_prefix: str = args.out_prefix
    no_time_stats: bool = bool(getattr(args, "no_time_stats", False))
    n_jobs = getattr(args, "n_jobs", None)

    df_all = load_snapshots(snapshots_glob, symbols=symbols)
    df_all = add_derived_features(df_all)

    available_cols = set(df_all.columns)
    features = [f for f in features if (f in available_cols)]
    if not features:
        raise ValueError("None of the requested features are present in snapshots")

    df_uni = univariate_sweep(df_all, features, quantiles, min_trades_bin, mode, n_jobs=n_jobs)

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    uni_path = f"{out_prefix}_univariate.csv"
    df_uni.to_csv(uni_path, index=False)
    print(f"[range-feature-sweep] written univariate sweep to {uni_path} (rows={len(df_uni)})")

    if not no_time_stats:
        df_hour = time_stats_hour(df_all)
        df_dow = time_stats_dow(df_all)

        if not df_hour.empty:
            hour_path = f"{out_prefix}_time_hour.csv"
            df_hour.to_csv(hour_path, index=False)
            print(f"[range-feature-sweep] written hour stats to {hour_path} (rows={len(df_hour)})")

        if not df_dow.empty:
            dow_path = f"{out_prefix}_time_dow.csv"
            df_dow.to_csv(dow_path, index=False)
            print(f"[range-feature-sweep] written day_of_week stats to {dow_path} (rows={len(df_dow)})")
