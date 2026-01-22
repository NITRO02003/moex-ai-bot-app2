import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app2.metrics import summary_from_pnl


def _per_symbol_stats(df: pd.DataFrame, equity0: float, min_trades: int) -> pd.DataFrame:
    rows = []
    for symbol, group in df.groupby("symbol"):
        pnl = pd.to_numeric(group.get("pnl"), errors="coerce").fillna(0.0)
        stats = summary_from_pnl(pnl, equity0=equity0)
        rows.append(
            {
                "symbol": symbol,
                "trades": int(len(pnl)),
                "pf": float(stats.get("profit_factor", 0.0)),
                "win_rate": float(stats.get("win_rate", 0.0)),
                "calmar": float(stats.get("calmar", 0.0)),
                "total_return": float(stats.get("total_return", 0.0)),
                "max_drawdown": float(stats.get("max_drawdown", 0.0)),
                "total_pnl": float(pnl.sum()),
                "avg_trade": float(stats.get("avg_trade", 0.0)),
                "low_trades": bool(min_trades > 0 and len(pnl) < min_trades),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_pf"] = out["pf"].rank(ascending=False, method="min").astype(int)
    out["rank_calmar"] = out["calmar"].rank(ascending=False, method="min").astype(int)
    out["rank_avg"] = ((out["rank_pf"] + out["rank_calmar"]) / 2.0).round(2)
    return out.sort_values(["rank_avg", "symbol"]).reset_index(drop=True)


def _add_deltas(base: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    base = base.add_suffix("_base")
    curr = curr.add_suffix("_curr")
    merged = base.merge(curr, left_on="symbol_base", right_on="symbol_curr", how="inner")
    merged = merged.rename(columns={"symbol_base": "symbol"}).drop(columns=["symbol_curr"])
    for col in [
        "trades",
        "pf",
        "win_rate",
        "calmar",
        "total_return",
        "max_drawdown",
        "total_pnl",
        "avg_trade",
    ]:
        merged[f"{col}_delta"] = merged[f"{col}_curr"] - merged[f"{col}_base"]
    merged["rank_pf_delta"] = merged["pf_delta"].rank(ascending=False, method="min").astype(int)
    merged["rank_calmar_delta"] = merged["calmar_delta"].rank(ascending=False, method="min").astype(int)
    return merged.sort_values(["rank_calmar_delta", "rank_pf_delta", "symbol"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ticker suitability by PF/Calmar.")
    parser.add_argument("--trades", required=True, help="Path to trades.csv for current run")
    parser.add_argument("--baseline-trades", default="", help="Optional baseline trades.csv")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for reports")
    parser.add_argument("--equity0", type=float, default=1_000_000.0, help="Equity baseline")
    parser.add_argument("--min-trades", type=int, default=0, help="Flag low trades threshold")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        raise SystemExit(f"Missing trades file: {trades_path}")

    df = pd.read_csv(trades_path)
    if "symbol" not in df.columns:
        raise SystemExit("Missing symbol column in trades file.")
    if "pnl" not in df.columns:
        raise SystemExit("Missing pnl column in trades file.")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    curr_stats = _per_symbol_stats(df, args.equity0, args.min_trades)
    stats_path = out_prefix.with_name(out_prefix.name + "_ticker_stats.csv")
    curr_stats.to_csv(stats_path, index=False)

    impact_path = None
    if args.baseline_trades:
        baseline_path = Path(args.baseline_trades)
        if not baseline_path.exists():
            raise SystemExit(f"Missing baseline trades file: {baseline_path}")
        base_df = pd.read_csv(baseline_path)
        base_stats = _per_symbol_stats(base_df, args.equity0, args.min_trades)
        impact = _add_deltas(base_stats, curr_stats)
        impact_path = out_prefix.with_name(out_prefix.name + "_ticker_impact_vs_baseline.csv")
        impact.to_csv(impact_path, index=False)

    print(f"[ticker_suitability] stats -> {stats_path}")
    if impact_path:
        print(f"[ticker_suitability] impact -> {impact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
