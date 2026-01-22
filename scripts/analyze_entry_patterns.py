import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


DEFAULT_FEATURES = [
    "band_width_pct",
    "range_vs_atr",
    "atr_14_pct",
    "dist_from_ma",
    "band_pos",
    "edge_proximity",
    "z_ma",
    "hour",
    "day_of_week",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_mean_20",
    "ret_vol_20",
    "bar_range_pct",
    "bar_body_pct",
    "body_vs_range",
]


def _parse_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_naive_dt(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if hasattr(dt.dt, "tz_convert"):
        dt = dt.dt.tz_convert(None)
    return dt


def _select_bad(df: pd.DataFrame, quantile: float, min_rows: int) -> pd.Series:
    pnl = pd.to_numeric(df.get("pnl_rel"), errors="coerce")
    pnl = pnl.replace([np.inf, -np.inf], np.nan)
    pnl = pnl.fillna(0.0)
    q = float(quantile)
    q = min(max(q, 0.0), 0.5)
    cutoff = pnl.quantile(q)
    mask = pnl <= cutoff
    if mask.sum() < min_rows:
        mask = pnl.nsmallest(min_rows).index
        mask = df.index.isin(mask)
    return mask


def _numeric_bins(df: pd.DataFrame, feat: str, quantiles: List[float]) -> pd.DataFrame:
    series = pd.to_numeric(df[feat], errors="coerce")
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.DataFrame()
    q_vals = sorted(set(quantiles))
    q_vals = [q for q in q_vals if 0.0 <= q <= 1.0]
    if len(q_vals) < 2:
        return pd.DataFrame()
    try:
        bins = pd.qcut(valid, q=q_vals, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    temp = df.loc[valid.index, ["is_bad"]].copy()
    temp["bin"] = bins
    g = temp.groupby("bin", observed=True)["is_bad"]
    out = g.agg(["count", "sum"]).reset_index().rename(columns={"sum": "bad_count"})
    out["bad_rate"] = out["bad_count"] / out["count"].replace(0, np.nan)
    out["feature"] = feat
    return out


def _numeric_summary(df: pd.DataFrame, feat: str) -> dict:
    series = pd.to_numeric(df[feat], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    all_vals = series.dropna()
    bad_vals = series[df["is_bad"]].dropna()
    if all_vals.empty:
        return {}
    return {
        "feature": feat,
        "all_mean": float(all_vals.mean()),
        "all_median": float(all_vals.median()),
        "bad_mean": float(bad_vals.mean()) if not bad_vals.empty else np.nan,
        "bad_median": float(bad_vals.median()) if not bad_vals.empty else np.nan,
        "bad_rate": float(df["is_bad"].mean()),
        "all_p10": float(all_vals.quantile(0.10)),
        "all_p90": float(all_vals.quantile(0.90)),
        "bad_p10": float(bad_vals.quantile(0.10)) if not bad_vals.empty else np.nan,
        "bad_p90": float(bad_vals.quantile(0.90)) if not bad_vals.empty else np.nan,
    }


def _categorical_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(col, dropna=False)["is_bad"]
    out = g.agg(["count", "sum"]).reset_index().rename(columns={"sum": "bad_count"})
    out["bad_rate"] = out["bad_count"] / out["count"].replace(0, np.nan)
    out["feature"] = col
    return out.sort_values("bad_rate", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze bad-entry patterns using entry candidates.")
    parser.add_argument("--trades", required=True, help="Path to trades.csv for the run")
    parser.add_argument("--entry-candidates", required=True, help="Path to entry candidates CSV")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for reports")
    parser.add_argument("--features", type=str, default="", help="Comma-separated feature list")
    parser.add_argument("--quantiles", type=str, default="0,0.2,0.4,0.6,0.8,1", help="Quantiles for bins")
    parser.add_argument("--bad-quantile", type=float, default=0.01, help="Bad trades quantile")
    parser.add_argument("--bad-min", type=int, default=50, help="Minimum bad trades rows")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    entry_path = Path(args.entry_candidates)
    if not trades_path.exists():
        raise SystemExit(f"Missing trades file: {trades_path}")
    if not entry_path.exists():
        raise SystemExit(f"Missing entry candidates file: {entry_path}")

    trades = pd.read_csv(trades_path)
    entries = pd.read_csv(entry_path)
    if "symbol" not in trades.columns or "entry_time" not in trades.columns:
        raise SystemExit("Trades file missing symbol/entry_time.")
    if "symbol" not in entries.columns or "entry_dt" not in entries.columns:
        raise SystemExit("Entry candidates missing symbol/entry_dt.")

    trades["entry_dt"] = _to_naive_dt(trades["entry_time"])
    entries["entry_dt"] = _to_naive_dt(entries["entry_dt"])
    entries = entries.drop_duplicates(subset=["symbol", "entry_dt"])

    merged = trades.merge(entries, on=["symbol", "entry_dt"], how="left", suffixes=("", "_entry"))
    merged["is_bad"] = _select_bad(merged, args.bad_quantile, args.bad_min)

    # ensure time features
    if "hour" not in merged.columns:
        merged["hour"] = merged["entry_dt"].dt.hour
    if "day_of_week" not in merged.columns:
        merged["day_of_week"] = merged["entry_dt"].dt.dayofweek

    features = _parse_list(args.features) or list(DEFAULT_FEATURES)
    features = [f for f in features if f in merged.columns]
    quantiles = [float(q) for q in str(args.quantiles).split(",") if q.strip()]

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # numeric summaries
    summary_rows = []
    bins_rows = []
    for feat in features:
        summary = _numeric_summary(merged, feat)
        if summary:
            summary_rows.append(summary)
        bins = _numeric_bins(merged, feat, quantiles)
        if not bins.empty:
            bins_rows.append(bins)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_prefix.with_name(out_prefix.name + "_entry_feature_summary.csv"), index=False)
    if bins_rows:
        pd.concat(bins_rows, ignore_index=True).to_csv(
            out_prefix.with_name(out_prefix.name + "_entry_feature_bins.csv"), index=False
        )

    # categorical patterns
    cat_frames = []
    for col in ["side", "entry_reason", "entry_geo_class", "trend", "hour", "day_of_week"]:
        cat = _categorical_rates(merged, col)
        if not cat.empty:
            cat_frames.append(cat)
    if cat_frames:
        pd.concat(cat_frames, ignore_index=True).to_csv(
            out_prefix.with_name(out_prefix.name + "_entry_feature_categorical.csv"), index=False
        )

    print(f"[entry_patterns] rows={len(merged)} bad={int(merged['is_bad'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
