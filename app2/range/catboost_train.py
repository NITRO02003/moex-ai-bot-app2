from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import baseline_ml
from ..metrics import summary_from_pnl
from ..parallel import default_n_jobs

try:
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover - import guard for missing catboost
    raise SystemExit(
        "CatBoost is not available in this environment. "
        "Run with your venv1 Python where catboost is installed."
    ) from exc


COMPACT_FEATURES = baseline_ml.COMPACT_FEATURES


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_features(
    df: pd.DataFrame,
    include: List[str] | None,
    exclude: List[str],
) -> pd.DataFrame:
    work = df.replace([np.inf, -np.inf], np.nan)
    num = work.select_dtypes(include=[np.number]).copy()
    if include:
        present = [c for c in include if c in num.columns]
        num = num[present].copy()
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


def _trade_stats_from_returns(
    returns: pd.Series,
    equity0: float = 1.0,
) -> Dict[str, float | int]:
    ret = pd.to_numeric(returns, errors="coerce").dropna()
    if ret.empty:
        return {
            "trades": 0,
            "pf": 0.0,
            "win_rate": 0.0,
            "calmar": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
        }
    stats = summary_from_pnl(ret, equity0=equity0)
    return {
        "trades": int(len(ret)),
        "pf": stats.get("profit_factor"),
        "win_rate": stats.get("win_rate"),
        "calmar": stats.get("calmar"),
        "total_return": stats.get("total_return"),
        "max_drawdown": stats.get("max_drawdown"),
    }


def _per_symbol_trade_stats_from_returns(
    df: pd.DataFrame,
    returns_col: str,
    mask: pd.Series | None = None,
) -> Dict[str, Dict[str, float | int]]:
    if "symbol" not in df.columns:
        return {}
    work = df.copy()
    if mask is not None:
        work = work.loc[mask]
    if work.empty:
        return {}
    out: Dict[str, Dict[str, float | int]] = {}
    for symbol, group in work.groupby("symbol"):
        out[str(symbol)] = _trade_stats_from_returns(group[returns_col])
    return out


def _top_pct_trade_metrics(
    df: pd.DataFrame,
    prob: pd.Series,
    returns_col: str,
    pcts: List[float],
) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    if df.empty or returns_col not in df.columns:
        return out
    series = prob.reindex(df.index)
    for pct in pcts:
        if pct <= 0 or pct > 1:
            continue
        thr = float(series.quantile(1.0 - pct))
        mask = series >= thr
        out[str(pct)] = _trade_stats_from_returns(df.loc[mask, returns_col])
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


def _trade_level_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "trade_uid" in df.columns:
        return df.drop_duplicates(subset=["trade_uid"])
    if "trade_id" in df.columns and "symbol" in df.columns:
        return df.drop_duplicates(subset=["symbol", "trade_id"])
    if "trade_id" in df.columns:
        return df.drop_duplicates(subset=["trade_id"])
    return df.drop_duplicates()


def _pick_trade_pnl(df: pd.DataFrame) -> tuple[str | None, float]:
    candidates = [
        ("trade_pnl_abs", 1_000_000.0),
        ("pnl_abs", 1_000_000.0),
        ("trade_pnl_rel", 1.0),
        ("pnl_rel", 1.0),
    ]
    for col, equity0 in candidates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            return col, equity0
    return None, 1_000_000.0


def _per_symbol_trade_stats(
    df: pd.DataFrame,
    pnl_col: str | None,
    equity0: float,
) -> Dict[str, Dict[str, float | int | None]]:
    if df.empty or pnl_col is None or "symbol" not in df.columns:
        return {}
    work = df.copy()
    if "entry_dt" in work.columns:
        work["entry_dt"] = pd.to_datetime(work["entry_dt"], errors="coerce")

    out: Dict[str, Dict[str, float | int | None]] = {}
    for symbol, group in work.groupby("symbol"):
        if "entry_dt" in group.columns:
            group = group.sort_values("entry_dt")
        pnl = pd.to_numeric(group[pnl_col], errors="coerce").dropna()
        if pnl.empty:
            continue
        stats = summary_from_pnl(pnl, equity0=equity0)
        out[str(symbol)] = {
            "trades": int(len(pnl)),
            "pf": stats.get("profit_factor"),
            "win_rate": stats.get("win_rate"),
            "calmar": stats.get("calmar"),
            "total_return": stats.get("total_return"),
            "max_drawdown": stats.get("max_drawdown"),
        }
    return out


def _run_entry(
    entry_path: Path,
    test_size: float,
    feature_set: str,
    split_mode: str,
    include_override: List[str],
    exclude: List[str],
    params: Dict[str, object],
    model_path: Path,
) -> Dict[str, object]:
    df = pd.read_csv(entry_path)
    # Determine dataset type (trades vs candidates) from the accompanying meta
    # file if available.  This prevents accidental mixing of research and policy
    # datasets.  Fallback to heuristics based on column presence when meta is
    # missing.
    entry_mode: str = "trades"
    meta_path = None
    try:
        p = Path(entry_path)
        # Replace .csv suffix with _meta.json; robust if file names contain
        # multiple dots
        meta_path = p.with_name(p.stem + "_meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # truth_policy field overrides entry_mode; fallback to old key
            entry_mode = str(meta.get("truth_policy", meta.get("entry_mode", "trades")))
    except Exception:
        # meta not available or invalid; will rely on heuristics below
        entry_mode = "trades"
    # Column-based heuristics: preserve legacy behaviour if meta is absent or ambiguous
    label_col: str = "pnl_rel"
    if entry_mode == "candidates" or "y_entry" in df.columns:
        # Candidate datasets are expected to have ``y_entry`` and entry_ret columns
        if "y_entry" in df.columns:
            label_col = "y_entry"
            entry_mode = "candidates"
            df = df[df[label_col].notna()].copy()
            df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        else:
            # If candidate mode but label missing, interpret binary target from entry_ret
            entry_mode = "candidates"
            if "entry_ret" in df.columns:
                label_col = "y_entry"
                df[label_col] = (pd.to_numeric(df["entry_ret"], errors="coerce") > 0).astype(int)
            else:
                # fallback to trades behaviour
                entry_mode = "trades"
    if entry_mode != "candidates":
        # Trades datasets: derive y_profit target from pnl_rel
        df = df[df["pnl_rel"].notna()].copy()
        df["y_profit"] = (df["pnl_rel"] > 0).astype(int)
        label_col = "y_profit"
    # Perform time‑based split for training/test.  If splitting by symbol,
    # ensure that each symbol's training segment ends strictly before the test segment begins.
    if split_mode == "per_symbol":
        time_col = "entry_dt" if "entry_dt" in df.columns else "signal_dt"
        train_df, test_df = baseline_ml._split_by_symbol_time(df, "symbol", time_col, test_size)
        # Verify temporal split per symbol (no lookahead)
        if not train_df.empty and not test_df.empty:
            # ensure that for each symbol the max train time is < min test time
            work_train = train_df[["symbol", time_col]].copy()
            work_train[time_col] = pd.to_datetime(work_train[time_col], errors="coerce")
            work_test = test_df[["symbol", time_col]].copy()
            work_test[time_col] = pd.to_datetime(work_test[time_col], errors="coerce")
            for sym in work_train["symbol"].unique():
                train_times = work_train.loc[work_train["symbol"] == sym, time_col].dropna()
                test_times = work_test.loc[work_test["symbol"] == sym, time_col].dropna()
                if not train_times.empty and not test_times.empty:
                    max_train = train_times.max()
                    min_test = test_times.min()
                    if max_train >= min_test:
                        raise ValueError(
                            f"Temporal split violation for symbol {sym}: max train {max_train} >= min test {min_test}"  # noqa: E501
                        )
    else:
        time_col = "entry_dt" if "entry_dt" in df.columns else "signal_dt"
        train_df, test_df = baseline_ml._split_by_time(df, time_col, test_size)
        # Verify temporal split globally
        if not train_df.empty and not test_df.empty:
            train_dt = pd.to_datetime(train_df[time_col], errors="coerce")
            test_dt = pd.to_datetime(test_df[time_col], errors="coerce")
            if train_dt.notna().any() and test_dt.notna().any():
                max_train = train_dt.max()
                min_test = test_dt.min()
                if max_train >= min_test:
                    raise ValueError(
                        f"Temporal split violation: max train {max_train} >= min test {min_test}"  # noqa: E501
                    )

    drop_cols = [
        "pnl_rel",
        "pnl_abs",
        "bars_in_trade",
        "max_adverse_excursion",
        "y_profit",
        "y_entry",
        "entry_ret",
        "entry_mfe",
        "entry_mae",
        "entry_price",
        "signal_dt",
        "entry_dt",
    ]
    include = include_override or (COMPACT_FEATURES if feature_set == "compact" else None)
    exclude = drop_cols + exclude

    x_train = _select_features(train_df, include, exclude)
    x_test = _select_features(test_df, include, exclude)
    x_all = _select_features(df, include, exclude)
    y_train = train_df[label_col].to_numpy(dtype=int)
    y_test = test_df[label_col].to_numpy(dtype=int)
    y_all = df[label_col].to_numpy(dtype=int)

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
    # Select threshold on a validation subset of the training data to avoid
    # optimistic bias.  We take the last 20% of the training data in
    # time‑sorted order (for per‑symbol mode we split each symbol).  If
    # the training set is too small, fallback to using the entire training.
    def _select_threshold_from_validation(train_df: pd.DataFrame, prob: np.ndarray) -> float:
        if len(train_df) < 10:
            return baseline_ml._best_threshold(y_train, prob)
        # Determine time column
        time_col_inner = "entry_dt" if "entry_dt" in train_df.columns else "signal_dt"
        if split_mode == "per_symbol":
            val_probs: List[float] = []
            val_labels: List[int] = []
            for sym, group_idx in train_df.groupby("symbol").indices.items():
                group_df = train_df.iloc[group_idx]
                group_prob = prob[group_idx]
                # sort by time
                group_df = group_df.reset_index(drop=True)
                group_time = pd.to_datetime(group_df[time_col_inner], errors="coerce")
                order = group_time.argsort()
                group_df_sorted = group_df.iloc[order]
                group_prob_sorted = group_prob[order]
                n = len(group_df_sorted)
                split_idx = int(n * 0.8)
                # validation portion = last 20%
                val_probs.extend(group_prob_sorted[split_idx:].tolist())
                val_labels.extend(group_df_sorted[label_col].iloc[split_idx:].to_numpy(dtype=int).tolist())
            if len(val_labels) < 1:
                return baseline_ml._best_threshold(y_train, prob)
            return baseline_ml._best_threshold(np.array(val_labels, dtype=int), np.array(val_probs, dtype=float))
        else:
            # global splitting
            train_copy = train_df.copy()
            train_copy[time_col_inner] = pd.to_datetime(train_copy[time_col_inner], errors="coerce")
            train_copy = train_copy.sort_values(time_col_inner)
            n = len(train_copy)
            split_idx = int(n * 0.8)
            val_labels = train_copy[label_col].iloc[split_idx:].to_numpy(dtype=int)
            val_probs = prob[split_idx:]
            if len(val_labels) < 1:
                return baseline_ml._best_threshold(y_train, prob)
            return baseline_ml._best_threshold(val_labels, val_probs)

    best_thr = _select_threshold_from_validation(train_df, prob_train)

    per_symbol_trade_full: Dict[str, Dict[str, float | int | None]] = {}
    per_symbol_trade_test: Dict[str, Dict[str, float | int | None]] = {}
    trade_metrics_source: Dict[str, object] = {}
    entry_trade_metrics_test: Dict[str, object] = {}
    entry_trade_metrics_full: Dict[str, object] = {}
    if entry_mode == "trades":
        trade_full = _trade_level_rows(df)
        trade_test = _trade_level_rows(test_df)
        trade_pnl_col, trade_equity0 = _pick_trade_pnl(trade_full)
        per_symbol_trade_full = _per_symbol_trade_stats(trade_full, trade_pnl_col, trade_equity0)
        per_symbol_trade_test = _per_symbol_trade_stats(trade_test, trade_pnl_col, trade_equity0)
        trade_metrics_source = {"pnl_col": trade_pnl_col, "equity0": trade_equity0}
    elif entry_mode == "candidates" and "entry_ret" in df.columns:
        prob_test_series = pd.Series(prob_test, index=test_df.index)
        prob_all_series = pd.Series(prob_all, index=df.index)
        test_mask = prob_test_series >= best_thr
        full_mask = prob_all_series >= best_thr
        top_pcts = [0.1, 0.2, 0.3, 0.5]
        entry_trade_metrics_test = {
            "baseline_all": _trade_stats_from_returns(test_df["entry_ret"]),
            "filtered": _trade_stats_from_returns(test_df.loc[test_mask, "entry_ret"]),
            "per_symbol_all": _per_symbol_trade_stats_from_returns(test_df, "entry_ret"),
            "per_symbol_filtered": _per_symbol_trade_stats_from_returns(test_df, "entry_ret", mask=test_mask),
            "top_pct": _top_pct_trade_metrics(test_df, prob_test_series, "entry_ret", top_pcts),
        }
        entry_trade_metrics_full = {
            "baseline_all": _trade_stats_from_returns(df["entry_ret"]),
            "filtered": _trade_stats_from_returns(df.loc[full_mask, "entry_ret"]),
            "per_symbol_all": _per_symbol_trade_stats_from_returns(df, "entry_ret"),
            "per_symbol_filtered": _per_symbol_trade_stats_from_returns(df, "entry_ret", mask=full_mask),
            "top_pct": _top_pct_trade_metrics(df, prob_all_series, "entry_ret", top_pcts),
        }
        trade_metrics_source = {"returns_col": "entry_ret", "equity0": 1.0}

    report = {
        "rows": int(len(df)),
        "target": label_col,
        "target_rate": float(pd.Series(y_all).mean()) if len(df) else 0.0,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": list(x_train.columns),
        "pos_weight": pos_weight,
        "train_metrics": _metrics(y_train, prob_train, threshold=best_thr),
        "test_metrics": _metrics(y_test, prob_test, threshold=best_thr),
        "per_symbol_test": _per_symbol_metrics(test_df, y_test, prob_test, threshold=best_thr),
        "per_symbol_full": _per_symbol_metrics(df, y_all, prob_all, threshold=best_thr),
        "entry_mode": entry_mode,
        "per_symbol_trade_test": per_symbol_trade_test,
        "per_symbol_trade_full": per_symbol_trade_full,
        "entry_trade_metrics_test": entry_trade_metrics_test,
        "entry_trade_metrics_full": entry_trade_metrics_full,
        "trade_metrics_source": trade_metrics_source,
        "feature_config": {
            "feature_set": feature_set,
            "include": include_override,
            "exclude": exclude,
        },
        "threshold_selection": "val_split" if len(train_df) >= 10 else "train",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    return report


def _run_intrade(
    intrade_path: Path,
    test_size: float,
    feature_set: str,
    split_mode: str,
    include_override: List[str],
    exclude: List[str],
    params: Dict[str, object],
    model_path: Path,
) -> Dict[str, object]:
    df = pd.read_csv(intrade_path)
    df = df[df["y_exit"].notna()].copy()
    df["y_exit"] = pd.to_numeric(df["y_exit"], errors="coerce").fillna(0).astype(int)
    if split_mode == "per_symbol":
        # per-symbol split by trade ensures temporal ordering per symbol
        train_df, test_df = baseline_ml._split_by_symbol_trade(df, "symbol", "trade_uid", "entry_dt", test_size)
        # Temporal split guard: for each symbol ensure no overlap between train and test
        if not train_df.empty and not test_df.empty:
            # Build per-symbol entry_dt times for trades
            work_train = train_df[["symbol", "entry_dt"]].copy()
            work_train["entry_dt"] = pd.to_datetime(work_train["entry_dt"], errors="coerce")
            work_test = test_df[["symbol", "entry_dt"]].copy()
            work_test["entry_dt"] = pd.to_datetime(work_test["entry_dt"], errors="coerce")
            for sym in work_train["symbol"].unique():
                train_times = work_train.loc[work_train["symbol"] == sym, "entry_dt"].dropna()
                test_times = work_test.loc[work_test["symbol"] == sym, "entry_dt"].dropna()
                if not train_times.empty and not test_times.empty:
                    max_train = train_times.max()
                    min_test = test_times.min()
                    if max_train >= min_test:
                        raise ValueError(
                            f"Temporal split violation for symbol {sym}: max train {max_train} >= min test {min_test}"  # noqa: E501
                        )
    else:
        # global split by trade id
        train_df, test_df = baseline_ml._split_by_trade(df, "trade_uid", "entry_dt", test_size)
        if not train_df.empty and not test_df.empty:
            train_dt = pd.to_datetime(train_df["entry_dt"], errors="coerce")
            test_dt = pd.to_datetime(test_df["entry_dt"], errors="coerce")
            if train_dt.notna().any() and test_dt.notna().any():
                max_train = train_dt.max()
                min_test = test_dt.min()
                if max_train >= min_test:
                    raise ValueError(
                        f"Temporal split violation: max train {max_train} >= min test {min_test}"  # noqa: E501
                    )

    drop_cols = [
        "y_exit",
        "trade_id",
        "trade_pnl_abs",
        "trade_pnl_rel",
        "trade_bars_in_trade",
        "exit_price",
    ]
    include = include_override or (COMPACT_FEATURES if feature_set == "compact" else None)
    exclude = drop_cols + exclude

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
    # Select threshold via validation subset of training data (20% of training trades)
    def _select_threshold_from_validation_intrade(train_df: pd.DataFrame, prob: np.ndarray) -> float:
        if len(train_df) < 10:
            return baseline_ml._best_threshold(y_train, prob)
        if split_mode == "per_symbol":
            val_probs: List[float] = []
            val_labels: List[int] = []
            # group by symbol, ensure ordering by entry_dt
            for sym, idx in train_df.groupby("symbol").indices.items():
                group_df = train_df.iloc[idx]
                group_prob = prob[idx]
                group_time = pd.to_datetime(group_df["entry_dt"], errors="coerce")
                order = group_time.argsort()
                group_df_sorted = group_df.iloc[order]
                group_prob_sorted = group_prob[order]
                n = len(group_df_sorted)
                split_idx = int(n * 0.8)
                val_probs.extend(group_prob_sorted[split_idx:].tolist())
                val_labels.extend(group_df_sorted["y_exit"].iloc[split_idx:].to_numpy(dtype=int).tolist())
            if len(val_labels) < 1:
                return baseline_ml._best_threshold(y_train, prob)
            return baseline_ml._best_threshold(np.array(val_labels, dtype=int), np.array(val_probs, dtype=float))
        else:
            # global split by time
            train_copy = train_df.copy()
            train_copy["entry_dt"] = pd.to_datetime(train_copy["entry_dt"], errors="coerce")
            train_copy = train_copy.sort_values("entry_dt")
            n = len(train_copy)
            split_idx = int(n * 0.8)
            val_labels = train_copy["y_exit"].iloc[split_idx:].to_numpy(dtype=int)
            val_probs = prob[split_idx:]
            if len(val_labels) < 1:
                return baseline_ml._best_threshold(y_train, prob)
            return baseline_ml._best_threshold(val_labels, val_probs)

    best_thr = _select_threshold_from_validation_intrade(train_df, prob_train)

    trade_full = _trade_level_rows(df)
    trade_test = _trade_level_rows(test_df)
    trade_pnl_col, trade_equity0 = _pick_trade_pnl(trade_full)
    per_symbol_trade_full = _per_symbol_trade_stats(trade_full, trade_pnl_col, trade_equity0)
    per_symbol_trade_test = _per_symbol_trade_stats(trade_test, trade_pnl_col, trade_equity0)

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
        "per_symbol_trade_test": per_symbol_trade_test,
        "per_symbol_trade_full": per_symbol_trade_full,
        "trade_metrics_source": {
            "pnl_col": trade_pnl_col,
            "equity0": trade_equity0,
        },
        "feature_config": {
            "feature_set": feature_set,
            "include": include_override,
            "exclude": exclude,
        },
        "threshold_selection": "val_split" if len(train_df) >= 10 else "train",
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
        "--split-mode",
        type=str,
        choices=["global", "per_symbol"],
        default="per_symbol",
        help="Split mode: global time/trade split or per-symbol split.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["compact", "all"],
        default="compact",
        help="Feature set selection.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated feature list to include (overrides feature-set).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated feature list to exclude.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--iterations", type=int, default=500, help="CatBoost iterations.")
    parser.add_argument("--depth", type=int, default=5, help="CatBoost depth.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="CatBoost learning rate.")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="CatBoost L2 reg.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--thread-count",
        type=int,
        default=0,
        help="Threads for CatBoost (0/<=0 = auto)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["CPU", "GPU"],
        default="CPU",
        help="CatBoost task type.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="GPU devices list for CatBoost, e.g. '0' or '0,1'.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="out/range_v3/ALL_30m_BASE_catboost_report.json",
        help="Path to output JSON report.",
    )
    args = parser.parse_args()

    include_override = _parse_list(args.include)
    exclude = _parse_list(args.exclude)
    thread_count = default_n_jobs(args.thread_count)

    params = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.seed,
    }
    if args.task_type == "GPU":
        params["task_type"] = "GPU"
        if args.devices:
            params["devices"] = args.devices
    else:
        params["thread_count"] = thread_count

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
            "split_mode": args.split_mode,
            "include": include_override,
            "exclude": exclude,
            "task_type": args.task_type,
            "devices": args.devices,
        }
    }

    if args.mode in ("entry", "both"):
        report["entry"] = _run_entry(
            Path(args.entry_path),
            args.test_size,
            args.feature_set,
            args.split_mode,
            include_override,
            exclude,
            params,
            model_dir / "range_catboost_entry.cbm",
        )
    if args.mode in ("intrade", "both"):
        report["intrade"] = _run_intrade(
            Path(args.intrade_path),
            args.test_size,
            args.feature_set,
            args.split_mode,
            include_override,
            exclude,
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
