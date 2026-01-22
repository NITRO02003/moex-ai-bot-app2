from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=float)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _feature_stats(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        mask = series.notna()
        n = int(len(series))
        n_valid = int(mask.sum())
        missing_rate = float(1.0 - n_valid / n) if n else 0.0
        unique = int(series[mask].nunique()) if n_valid else 0
        mean = float(series[mask].mean()) if n_valid else None
        std = float(series[mask].std(ddof=0)) if n_valid else None

        auc = None
        corr = None
        if n_valid >= 5 and target[mask].nunique() >= 2:
            auc = _auc_rank(target[mask].to_numpy(dtype=int), series[mask].to_numpy(dtype=float))
            if unique > 1:
                corr = float(np.corrcoef(series[mask].to_numpy(dtype=float), target[mask].to_numpy(dtype=float))[0, 1])

        auc_abs = None
        if auc is not None:
            auc_abs = float(max(auc, 1.0 - auc))

        rows.append(
            {
                "name": col,
                "n": n,
                "n_valid": n_valid,
                "missing_rate": missing_rate,
                "unique": unique,
                "mean": mean,
                "std": std,
                "auc": auc,
                "auc_abs": auc_abs,
                "corr": corr,
            }
        )
    return pd.DataFrame(rows).sort_values(["auc_abs", "missing_rate"], ascending=[False, True])


def _prepare_numeric_features(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    work = df.replace([np.inf, -np.inf], np.nan)
    num = work.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in exclude if c in num.columns]
    if drop_cols:
        num.drop(columns=drop_cols, inplace=True)
    return num


def _report_entry(entry_path: Path) -> Dict[str, object]:
    df = pd.read_csv(entry_path)
    df = df[df["pnl_rel"].notna()].copy()
    df["y_profit"] = (df["pnl_rel"] > 0).astype(int)

    exclude = ["pnl_rel", "pnl_abs", "bars_in_trade", "max_adverse_excursion", "y_profit"]
    features = _prepare_numeric_features(df, exclude)
    stats_df = _feature_stats(features, df["y_profit"])

    return {
        "rows": int(len(df)),
        "target": "pnl_rel > 0",
        "target_rate": float(df["y_profit"].mean()) if len(df) else 0.0,
        "features": stats_df.to_dict(orient="records"),
    }


def _report_intrade(intrade_path: Path) -> Dict[str, object]:
    df = pd.read_csv(intrade_path)
    df = df[df["y_exit"].notna()].copy()
    df["y_exit"] = pd.to_numeric(df["y_exit"], errors="coerce").fillna(0).astype(int)

    exclude = [
        "y_exit",
        "trade_id",
        "trade_pnl_abs",
        "trade_pnl_rel",
        "trade_bars_in_trade",
        "exit_price",
    ]
    features = _prepare_numeric_features(df, exclude)
    stats_df = _feature_stats(features, df["y_exit"])

    return {
        "rows": int(len(df)),
        "target": "y_exit",
        "target_rate": float(df["y_exit"].mean()) if len(df) else 0.0,
        "features": stats_df.to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature usefulness report for Dataset A/B.")
    parser.add_argument(
        "--entry-path",
        type=str,
        default="out/range_v3/ALL_30m_BASE_entry_snapshots.csv",
        help="Path to entry snapshots dataset (Dataset A).",
    )
    parser.add_argument(
        "--intrade-path",
        type=str,
        default="out/range_v3/ALL_30m_BASE_intrade_timeseries.csv",
        help="Path to intrade timeseries dataset (Dataset B).",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="out/range_v3/ALL_30m_BASE_feature_report.json",
        help="Path to output JSON report.",
    )
    args = parser.parse_args()

    entry_path = Path(args.entry_path)
    intrade_path = Path(args.intrade_path)
    out_path = Path(args.out_path)

    report = {
        "entry": _report_entry(entry_path),
        "intrade": _report_intrade(intrade_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Also write CSVs for quick viewing
    entry_csv = out_path.with_name(out_path.stem + "_entry.csv")
    intrade_csv = out_path.with_name(out_path.stem + "_intrade.csv")
    pd.DataFrame(report["entry"]["features"]).to_csv(entry_csv, index=False)
    pd.DataFrame(report["intrade"]["features"]).to_csv(intrade_csv, index=False)

    print(f"[feature-report] saved report to {out_path}")
    print(f"[feature-report] entry CSV: {entry_csv}")
    print(f"[feature-report] intrade CSV: {intrade_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
