from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import baseline_ml

try:
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover - import guard for missing catboost
    raise SystemExit(
        "CatBoost is not available in this environment. "
        "Run with your venv1 Python where catboost is installed."
    ) from exc


COMPACT_FEATURES = [
    "v3_signal",
    "dist_from_ma",
    "band_pos",
    "band_width_pct",
    "edge_proximity",
    "z_ma",
    "bar_range_pct",
    "bar_body_pct",
    "body_vs_range",
    "range_vs_atr",
    "atr_14_pct",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_mean_20",
    "ret_vol_20",
]


def _select_features(df: pd.DataFrame, include: List[str] | None, exclude: List[str]) -> pd.DataFrame:
    work = df.replace([np.inf, -np.inf], np.nan)
    num = work.select_dtypes(include=[np.number]).copy()
    if include:
        present = [c for c in include if c in num.columns]
        return num[present]
    for col in exclude:
        if col in num.columns:
            num.drop(columns=[col], inplace=True)
    return num


def _train_catboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    params: Dict[str, object],
) -> CatBoostClassifier:
    model = CatBoostClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=(x_test, y_test),
        verbose=False,
    )
    return model


def _metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float | None]:
    out = baseline_ml._classification_metrics(y_true, y_prob, threshold=threshold)
    out["threshold"] = threshold
    return out


def _per_symbol_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Dict[str, float | None]]:
    if "symbol" not in df.columns:
        return {}
    work = df.reset_index(drop=True)
    y_series = pd.Series(y_true)
    p_series = pd.Series(y_prob)
    out: Dict[str, Dict[str, float | None]] = {}
    for symbol, idx in work.groupby("symbol").indices.items():
        y_sym = y_series.iloc[idx].to_numpy(dtype=int)
        p_sym = p_series.iloc[idx].to_numpy(dtype=float)
        metrics = baseline_ml._classification_metrics(y_sym, p_sym, threshold=threshold)
        metrics["n"] = int(len(y_sym))
        metrics["target_rate"] = float(y_sym.mean()) if len(y_sym) else 0.0
        metrics["threshold"] = threshold
        out[str(symbol)] = metrics
    return out


def _run_entry(
    entry_path: Path,
    test_size: float,
    feature_set: str,
    params: Dict[str, object],
    model_path: Path,
) -> Dict[str, object]:
    df = pd.read_csv(entry_path)
    df = df[df["pnl_rel"].notna()].copy()
    df["y_profit"] = (df["pnl_rel"] > 0).astype(int)
    train_df, test_df = baseline_ml._split_by_time(df, "entry_dt", test_size)

    exclude = ["pnl_rel", "pnl_abs", "bars_in_trade", "max_adverse_excursion", "y_profit"]
    include = COMPACT_FEATURES if feature_set == "compact" else None

    x_train = _select_features(train_df, include, exclude)
    x_test = _select_features(test_df, include, exclude)
    x_all = _select_features(df, include, exclude)
    y_train = train_df["y_profit"].to_numpy(dtype=int)
    y_test = test_df["y_profit"].to_numpy(dtype=int)
    y_all = df["y_profit"].to_numpy(dtype=int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = float(neg / pos) if pos > 0 else 1.0

    params = dict(params)
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "AUC")
    params.setdefault("class_weights", [1.0, pos_weight])

    model = _train_catboost(x_train, y_train, x_test, y_test, params)
    prob_train = model.predict_proba(x_train)[:, 1]
    prob_test = model.predict_proba(x_test)[:, 1]
    prob_all = model.predict_proba(x_all)[:, 1]
    best_thr = baseline_ml._best_threshold(y_train, prob_train)

    report = {
        "rows": int(len(df)),
        "target": "pnl_rel > 0",
        "target_rate": float(df["y_profit"].mean()) if len(df) else 0.0,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": list(x_train.columns),
        "pos_weight": pos_weight,
        "train_metrics": _metrics(y_train, prob_train, threshold=best_thr),
        "test_metrics": _metrics(y_test, prob_test, threshold=best_thr),
        "per_symbol_test": _per_symbol_metrics(test_df, y_test, prob_test, threshold=best_thr),
        "per_symbol_full": _per_symbol_metrics(df, y_all, prob_all, threshold=best_thr),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    return report


def _run_intrade(
    intrade_path: Path,
    test_size: float,
    feature_set: str,
    params: Dict[str, object],
    model_path: Path,
) -> Dict[str, object]:
    df = pd.read_csv(intrade_path)
    df = df[df["y_exit"].notna()].copy()
    df["y_exit"] = pd.to_numeric(df["y_exit"], errors="coerce").fillna(0).astype(int)
    train_df, test_df = baseline_ml._split_by_trade(df, "trade_uid", "entry_dt", test_size)

    exclude = [
        "y_exit",
        "trade_id",
        "trade_pnl_abs",
        "trade_pnl_rel",
        "trade_bars_in_trade",
        "exit_price",
    ]
    include = COMPACT_FEATURES if feature_set == "compact" else None

    x_train = _select_features(train_df, include, exclude)
    x_test = _select_features(test_df, include, exclude)
    x_all = _select_features(df, include, exclude)
    y_train = train_df["y_exit"].to_numpy(dtype=int)
    y_test = test_df["y_exit"].to_numpy(dtype=int)
    y_all = df["y_exit"].to_numpy(dtype=int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = float(neg / pos) if pos > 0 else 1.0

    params = dict(params)
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "AUC")
    params.setdefault("class_weights", [1.0, pos_weight])

    model = _train_catboost(x_train, y_train, x_test, y_test, params)
    prob_train = model.predict_proba(x_train)[:, 1]
    prob_test = model.predict_proba(x_test)[:, 1]
    prob_all = model.predict_proba(x_all)[:, 1]
    best_thr = baseline_ml._best_threshold(y_train, prob_train)

    report = {
        "rows": int(len(df)),
        "target": "y_exit",
        "target_rate": float(df["y_exit"].mean()) if len(df) else 0.0,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": list(x_train.columns),
        "pos_weight": pos_weight,
        "train_metrics": _metrics(y_train, prob_train, threshold=best_thr),
        "test_metrics": _metrics(y_test, prob_test, threshold=best_thr),
        "per_symbol_test": _per_symbol_metrics(test_df, y_test, prob_test, threshold=best_thr),
        "per_symbol_full": _per_symbol_metrics(df, y_all, prob_all, threshold=best_thr),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CatBoost on Range datasets A/B.")
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
        "--mode",
        type=str,
        choices=["entry", "intrade", "both"],
        default="both",
        help="Which dataset to train on.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["compact", "all"],
        default="compact",
        help="Feature set selection.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--iterations", type=int, default=500, help="CatBoost iterations.")
    parser.add_argument("--depth", type=int, default=5, help="CatBoost depth.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="CatBoost learning rate.")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="CatBoost L2 reg.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="out/range_v3/ALL_30m_BASE_catboost_report.json",
        help="Path to output JSON report.",
    )
    args = parser.parse_args()

    params = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.seed,
    }

    model_dir = Path(args.model_dir)
    report_path = Path(args.report_out)

    report: Dict[str, object] = {
        "params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "seed": args.seed,
            "feature_set": args.feature_set,
            "test_size": args.test_size,
        }
    }

    if args.mode in ("entry", "both"):
        report["entry"] = _run_entry(
            Path(args.entry_path),
            args.test_size,
            args.feature_set,
            params,
            model_dir / "range_catboost_entry.cbm",
        )
    if args.mode in ("intrade", "both"):
        report["intrade"] = _run_intrade(
            Path(args.intrade_path),
            args.test_size,
            args.feature_set,
            params,
            model_dir / "range_catboost_intrade.cbm",
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[catboost] saved report to {report_path}")
    if "entry" in report:
        print(f"[catboost] entry test F1: {report['entry']['test_metrics']['f1']}")
    if "intrade" in report:
        print(f"[catboost] intrade test F1: {report['intrade']['test_metrics']['f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
