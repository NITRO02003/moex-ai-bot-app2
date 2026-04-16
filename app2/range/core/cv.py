"""
Time series cross‑validation utilities for entry/intrade datasets.

This module implements a simple time‑ordered cross‑validation routine
for datasets used in the range‑core project.  It splits the data
chronologically into ``n_splits`` folds using scikit‑learn's
``TimeSeriesSplit``, trains a CatBoost classifier on each training
portion and evaluates it on the corresponding test portion.  The
resulting metrics can be used to assess the temporal stability of a
model and detect potential overfitting.

The CLI wrapper allows running cross‑validation from the command line
and writing a CSV or JSON summary of the per‑fold metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


def _guess_time_column(df: pd.DataFrame) -> str:
    """Try to guess the name of the timestamp column for sorting."""
    for col in [
        "datetime",
        "timestamp",
        "entry_time",
        "exit_time",
        "ts",
    ]:
        if col in df.columns:
            return col
    # fallback: first object column that can be parsed as datetime
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            try:
                pd.to_datetime(df[c].dropna().iloc[0])
                return c
            except Exception:
                continue
    raise ValueError("Could not infer a time column; please include a datetime column.")


def _determine_label_column(df: pd.DataFrame) -> str:
    """
    Determine the target label column for classification.

    Priority order:
      - 'y_entry'
      - 'y_profit_binary'
      - 'y_intrade'
      - 'pnl_rel' (converted to binary: pnl > 0)

    A ValueError is raised if no suitable label is found.
    """
    if "y_entry" in df.columns:
        return "y_entry"
    if "y_profit_binary" in df.columns:
        return "y_profit_binary"
    if "y_intrade" in df.columns:
        return "y_intrade"
    if "pnl_rel" in df.columns:
        return "pnl_rel"
    raise ValueError(
        "Dataset must contain one of 'y_entry', 'y_profit_binary', 'y_intrade' or 'pnl_rel' as the target column"
    )


def run_time_series_cv(
    dataset_path: str,
    n_splits: int,
    catboost_params: Optional[Dict[str, object]] = None,
    out_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """
    Perform time series cross‑validation on a dataset using CatBoost.

    Parameters
    ----------
    dataset_path:
        Path to the CSV dataset.  The dataset should contain a
        datetime column (see ``_guess_time_column``) and a binary
        target column (see ``_determine_label_column``).

    n_splits:
        Number of time series splits.  Must be at least 2.

    catboost_params:
        Optional dictionary of parameters to pass to the
        ``CatBoostClassifier``.  If None, sensible defaults will be
        used.  Users may override keys such as ``iterations``,
        ``depth``, ``learning_rate``, etc.

    out_path:
        Optional path to write the per‑fold metrics as JSON or CSV.
        If the file extension is ``.json`` the output will be JSON;
        otherwise a CSV will be written.  If None, no file will
        be written.

    Returns
    -------
    List[Dict[str, float]]
        List of dictionaries containing metrics for each fold.
    """
    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError(f"Dataset {dataset_path} is empty")
    time_col = _guess_time_column(df)
    # parse datetime
    df[time_col] = pd.to_datetime(df[time_col], utc=False)
    df = df.sort_values(time_col).reset_index(drop=True)
    label_col = _determine_label_column(df)
    # Create binary labels if using pnl_rel
    if label_col == "pnl_rel":
        labels = (df[label_col].astype(float) > 0).astype(int)
    else:
        labels = df[label_col].astype(int)
    # Determine feature columns (numeric, excluding labels, time, symbol)
    ignore_cols = set([label_col, time_col])
    ignore_cols.update([col for col in df.columns if "symbol" in col.lower()])
    feat_cols: List[str] = []
    for col in df.columns:
        if col in ignore_cols:
            continue
        # include numeric and boolean
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            feat_cols.append(col)
    if not feat_cols:
        raise ValueError("No numeric feature columns found in the dataset")
    X = df[feat_cols].fillna(0)
    # Setup cross-validation
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for time series cross-validation")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    params = catboost_params.copy() if catboost_params else {}
    # Provide sensible defaults if not specified
    params.setdefault("loss_function", "Logloss")
    params.setdefault("iterations", 200)
    params.setdefault("depth", 6)
    params.setdefault("learning_rate", 0.1)
    params.setdefault("verbose", False)
    metrics_list: List[Dict[str, float]] = []
    fold = 1
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        model = CatBoostClassifier(**params)
        # Fit model; use eval_set for early stopping if desired
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = None
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # Log loss and AUC if possible
        ll = log_loss(y_test, y_prob) if y_prob is not None else float("nan")
        auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and len(set(y_test)) > 1) else float("nan")
        metrics_list.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "logloss": ll,
                "auc": auc,
            }
        )
        fold += 1
    # Optionally write out
    if out_path:
        out_path_str = str(out_path)
        if out_path_str.lower().endswith(".json"):
            Path(out_path_str).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path_str, "w", encoding="utf-8") as f:
                json.dump(metrics_list, f, indent=2)
        else:
            out_df = pd.DataFrame(metrics_list)
            Path(out_path_str).parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path_str, index=False)
    return metrics_list


def _parse_catboost_params(params_str: Optional[str]) -> Optional[Dict[str, object]]:
    """Parse catboost parameters from a comma‑separated key=value string."""
    if not params_str:
        return None
    result: Dict[str, object] = {}
    for item in params_str.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        # attempt to convert to int or float
        if val.isdigit():
            result[key] = int(val)
        else:
            try:
                result[key] = float(val)
            except ValueError:
                # leave as string
                result[key] = val
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run time series cross‑validation on an entry/intrade dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset CSV (entry or intrade).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for time series cross‑validation. Defaults to 5.",
    )
    parser.add_argument(
        "--catboost-params",
        type=str,
        default=None,
        help=(
            "Optional comma‑separated list of CatBoost parameters, e.g. "
            "'iterations=200,depth=6,learning_rate=0.1'."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output file for the metrics (JSON if .json extension otherwise CSV).",
    )
    args = parser.parse_args()
    params = _parse_catboost_params(args.catboost_params)
    metrics = run_time_series_cv(
        dataset_path=args.dataset,
        n_splits=args.n_splits,
        catboost_params=params,
        out_path=args.out,
    )
    print(
        f"[cv] Completed time series CV on {args.dataset} with {args.n_splits} splits."
        f" Results saved to {args.out if args.out else 'not saved'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())