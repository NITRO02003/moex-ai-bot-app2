import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app2.metrics import summary_from_pnl


def _trade_stats(df: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(df.get("pnl"), errors="coerce").fillna(0.0)
    stats = summary_from_pnl(pnl, equity0=1_000_000.0)
    return {
        "trades": int(len(pnl)),
        "pf": float(stats.get("profit_factor", 0.0)),
        "win_rate": float(stats.get("win_rate", 0.0)),
        "calmar": float(stats.get("calmar", 0.0)),
        "total_return": float(stats.get("total_return", 0.0)),
        "max_drawdown": float(stats.get("max_drawdown", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze trend vs non-trend trade stats.")
    parser.add_argument("--trades", required=True, help="Path to trades.csv")
    parser.add_argument("--trend-k", type=float, default=0.00025, help="Trend slope threshold")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for reports")
    parser.add_argument("--bad-quantile", type=float, default=0.01, help="Bad trades quantile")
    parser.add_argument("--bad-min", type=int, default=50, help="Minimum bad trades rows")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        raise SystemExit(f"Missing trades file: {trades_path}")

    df = pd.read_csv(trades_path)
    if "entry_slope_pct_per_bar" not in df.columns:
        raise SystemExit("Missing entry_slope_pct_per_bar in trades file.")

    for col in [
        "pnl",
        "pnl_rel",
        "entry_slope_pct_per_bar",
        "entry_atr_pct",
        "entry_geo_h_pct",
        "entry_dist_L_pct",
        "entry_dist_U_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    trend_mask = df["entry_slope_pct_per_bar"].abs() >= float(args.trend_k)

    summary = {
        "trend_k": float(args.trend_k),
        "overall": _trade_stats(df),
        "trend": _trade_stats(df[trend_mask]),
        "nontrend": _trade_stats(df[~trend_mask]),
    }
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_summary = out_prefix.with_name(out_prefix.name + "_trend_split_summary.json")
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for symbol, group in df.groupby("symbol"):
        rows.append({"symbol": symbol, "segment": "trend", **_trade_stats(group[trend_mask.loc[group.index]])})
        rows.append({"symbol": symbol, "segment": "nontrend", **_trade_stats(group[~trend_mask.loc[group.index]])})
    per_symbol = pd.DataFrame(rows)
    per_symbol_path = out_prefix.with_name(out_prefix.name + "_trend_per_symbol_stats.csv")
    per_symbol.to_csv(per_symbol_path, index=False)

    if "pnl_rel" in df.columns:
        q = float(args.bad_quantile)
        q = min(max(q, 0.0), 0.5)
        cutoff = df["pnl_rel"].quantile(q)
        bad = df[df["pnl_rel"] <= cutoff].copy()
        if len(bad) < args.bad_min:
            bad = df.nsmallest(args.bad_min, "pnl_rel").copy()
    else:
        bad = df.copy()

    bad["trend"] = trend_mask.loc[bad.index].astype(bool)
    keep_cols = [
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "pnl",
        "pnl_rel",
        "bars_held",
        "entry_reason",
        "exit_reason",
        "entry_atr_pct",
        "entry_slope_pct_per_bar",
        "entry_geo_class",
        "entry_geo_h_pct",
        "entry_dist_L_pct",
        "entry_dist_U_pct",
        "trend",
    ]
    keep_cols = [c for c in keep_cols if c in bad.columns]
    bad_path = out_prefix.with_name(out_prefix.name + "_bad_entries.csv")
    bad[keep_cols].to_csv(bad_path, index=False)

    reason_all = df["exit_reason"].value_counts(dropna=False)
    reason_bad = bad["exit_reason"].value_counts(dropna=False)
    reason_df = pd.DataFrame({"overall": reason_all, "bad": reason_bad}).fillna(0).astype(int)
    reason_path = out_prefix.with_name(out_prefix.name + "_bad_exit_reasons.csv")
    reason_df.to_csv(reason_path)

    print(f"[trend_split] summary -> {out_summary}")
    print(f"[trend_split] per-symbol -> {per_symbol_path}")
    print(f"[bad_entries] -> {bad_path}")
    print(f"[bad_entries] reasons -> {reason_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
