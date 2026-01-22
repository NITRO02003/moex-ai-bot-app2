from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

PROXY_INTRRADE_FEATURES = [
    "volume",
    "bars_held",
    "time_since_entry_min",
    "entry_price",
    "qty",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _train_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
    sample_weight: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    n, d = x_train.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=float)
    weight_sum = float(sample_weight.sum()) if n > 0 else 1.0
    for _ in range(epochs):
        z = x_train @ w + b
        p = _sigmoid(z)
        error = (p - y_train) * sample_weight
        grad_w = (x_train.T @ error) / weight_sum + l2 * w
        grad_b = float(error.sum() / weight_sum)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _predict_logreg(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _sigmoid(x @ w + b)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.arange(1, len(y_score) + 1)
    rank_sum_pos = int(ranks[order][y_true[order] == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float | None]:
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = float((y_pred == y_true).mean())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    auc = _roc_auc(y_true, y_prob)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        metrics = _classification_metrics(y_true, y_prob, threshold=float(thr))
        f1 = metrics["f1"] or 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _prepare_features(
    df: pd.DataFrame,
    exclude: List[str],
) -> pd.DataFrame:
    work = df.replace([np.inf, -np.inf], np.nan)
    num = work.select_dtypes(include=[np.number]).copy()
    for col in exclude:
        if col in num.columns:
            num.drop(columns=[col], inplace=True)
    return num


def _parse_exclude(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _standardize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    means = train_df.mean(skipna=True)
    valid_cols = means.index[~means.isna()].tolist()
    train_df = train_df[valid_cols].copy()
    test_df = test_df[valid_cols].copy()
    means = means.loc[valid_cols]
    train_df = train_df.fillna(means)
    test_df = test_df.fillna(means)
    stds = train_df.std(ddof=0).replace(0.0, 1.0).loc[valid_cols]
    train_df = (train_df - means) / stds
    test_df = (test_df - means) / stds
    return train_df.to_numpy(dtype=float), test_df.to_numpy(dtype=float), valid_cols


def _split_by_time(df: pd.DataFrame, time_col: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    order = df.copy()
    order[time_col] = pd.to_datetime(order[time_col], errors="coerce")
    order = order.sort_values(time_col)
    split_idx = int(len(order) * (1.0 - test_size))
    return order.iloc[:split_idx], order.iloc[split_idx:]


def _split_by_trade(
    df: pd.DataFrame,
    trade_col: str,
    time_col: str,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = df[[trade_col, time_col]].drop_duplicates()
    trades[time_col] = pd.to_datetime(trades[time_col], errors="coerce")
    trades = trades.sort_values(time_col)
    split_idx = int(len(trades) * (1.0 - test_size))
    train_trades = set(trades.iloc[:split_idx][trade_col].tolist())
    train_df = df[df[trade_col].isin(train_trades)].copy()
    test_df = df[~df[trade_col].isin(train_trades)].copy()
    return train_df, test_df


def _run_entry_baseline(
    entry_path: Path,
    test_size: float,
    lr: float,
    epochs: int,
    l2: float,
    extra_exclude: List[str] | None = None,
) -> Dict[str, object]:
    df = pd.read_csv(entry_path)
    df = df[df["pnl_rel"].notna()].copy()
    df["y_profit"] = (df["pnl_rel"] > 0).astype(int)
    train_df, test_df = _split_by_time(df, "entry_dt", test_size)

    exclude = ["pnl_rel", "pnl_abs", "bars_in_trade", "max_adverse_excursion", "y_profit"]
    if extra_exclude:
        exclude.extend(extra_exclude)
    x_train_df = _prepare_features(train_df, exclude)
    x_test_df = _prepare_features(test_df, exclude)
    x_train, x_test, features = _standardize(x_train_df, x_test_df)

    y_train = train_df["y_profit"].to_numpy(dtype=int)
    y_test = test_df["y_profit"].to_numpy(dtype=int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = float(neg / pos) if pos > 0 else 1.0
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)
    w, b = _train_logreg(x_train, y_train, lr=lr, epochs=epochs, l2=l2, sample_weight=sample_weight)
    prob_train = _predict_logreg(x_train, w, b)
    prob_test = _predict_logreg(x_test, w, b)

    best_thr = _best_threshold(y_train, prob_train)
    train_metrics = _classification_metrics(y_train, prob_train, threshold=best_thr)
    test_metrics = _classification_metrics(y_test, prob_test, threshold=best_thr)
    base_rate = float(y_test.mean()) if len(y_test) else 0.0
    baseline_acc = max(base_rate, 1.0 - base_rate) if len(y_test) else 0.0

    return {
        "rows": int(len(df)),
        "target": "pnl_rel > 0",
        "target_rate": float(df["y_profit"].mean()) if len(df) else 0.0,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": features,
        "excluded_features": sorted(set(exclude)),
        "threshold": best_thr,
        "pos_weight": pos_weight,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline": {
            "majority_class": int(base_rate >= 0.5) if len(y_test) else 0,
            "accuracy": float(baseline_acc),
        },
    }


def _run_intrade_baseline(
    intrade_path: Path,
    test_size: float,
    lr: float,
    epochs: int,
    l2: float,
    extra_exclude: List[str] | None = None,
) -> Dict[str, object]:
    df = pd.read_csv(intrade_path)
    df = df[df["y_exit"].notna()].copy()
    df["y_exit"] = pd.to_numeric(df["y_exit"], errors="coerce").fillna(0).astype(int)
    train_df, test_df = _split_by_trade(df, "trade_uid", "entry_dt", test_size)

    exclude = [
        "y_exit",
        "trade_id",
        "trade_pnl_abs",
        "trade_pnl_rel",
        "trade_bars_in_trade",
        "exit_price",
    ]
    if extra_exclude:
        exclude.extend(extra_exclude)
    x_train_df = _prepare_features(train_df, exclude)
    x_test_df = _prepare_features(test_df, exclude)
    x_train, x_test, features = _standardize(x_train_df, x_test_df)

    y_train = train_df["y_exit"].to_numpy(dtype=int)
    y_test = test_df["y_exit"].to_numpy(dtype=int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = float(neg / pos) if pos > 0 else 1.0
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)
    w, b = _train_logreg(x_train, y_train, lr=lr, epochs=epochs, l2=l2, sample_weight=sample_weight)
    prob_train = _predict_logreg(x_train, w, b)
    prob_test = _predict_logreg(x_test, w, b)

    best_thr = _best_threshold(y_train, prob_train)
    train_metrics = _classification_metrics(y_train, prob_train, threshold=best_thr)
    test_metrics = _classification_metrics(y_test, prob_test, threshold=best_thr)
    base_rate = float(y_test.mean()) if len(y_test) else 0.0
    baseline_acc = max(base_rate, 1.0 - base_rate) if len(y_test) else 0.0

    return {
        "rows": int(len(df)),
        "target": "y_exit",
        "target_rate": float(df["y_exit"].mean()) if len(df) else 0.0,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": features,
        "excluded_features": sorted(set(exclude)),
        "threshold": best_thr,
        "pos_weight": pos_weight,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline": {
            "majority_class": int(base_rate >= 0.5) if len(y_test) else 0,
            "accuracy": float(baseline_acc),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ML baselines for Range datasets A/B.")
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
        default="out/range_v3/ALL_30m_BASE_ml_baseline.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--exclude-entry",
        type=str,
        default="",
        help="Comma-separated feature names to drop from Dataset A.",
    )
    parser.add_argument(
        "--exclude-intrade",
        type=str,
        default="",
        help="Comma-separated feature names to drop from Dataset B.",
    )
    parser.add_argument(
        "--exclude-proxy",
        action="store_true",
        help="Drop proxy features in Dataset B (volume, bars_held, time_since_entry_min, entry_price, qty).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization.")
    args = parser.parse_args()

    entry_path = Path(args.entry_path)
    intrade_path = Path(args.intrade_path)
    out_path = Path(args.out_path)

    exclude_entry = _parse_exclude(args.exclude_entry)
    exclude_intrade = _parse_exclude(args.exclude_intrade)
    if args.exclude_proxy:
        exclude_intrade.extend(PROXY_INTRRADE_FEATURES)
    exclude_entry = sorted(set(exclude_entry))
    exclude_intrade = sorted(set(exclude_intrade))

    report = {
        "entry": _run_entry_baseline(
            entry_path, args.test_size, args.lr, args.epochs, args.l2, extra_exclude=exclude_entry
        ),
        "intrade": _run_intrade_baseline(
            intrade_path, args.test_size, args.lr, args.epochs, args.l2, extra_exclude=exclude_intrade
        ),
        "params": {
            "test_size": args.test_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "l2": args.l2,
            "exclude_entry": exclude_entry,
            "exclude_intrade": exclude_intrade,
            "exclude_proxy": bool(args.exclude_proxy),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[baseline-ml] saved report to {out_path}")
    print(f"[baseline-ml] entry test F1: {report['entry']['test_metrics']['f1']}")
    print(f"[baseline-ml] intrade test F1: {report['intrade']['test_metrics']['f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
