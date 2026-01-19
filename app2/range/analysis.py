from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class EDAConfig:
    features_univariate: List[str]
    feature_pairs_bivariate: List[Tuple[str, str]]
    min_trades_bin: int = 10


def _load_csv(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("CSV path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df


def _merge_trades_snapshots(trades: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Merge trades and snapshots on symbol+trade_id or symbol+entry_dt.

    Если trade_id присутствует в обоих, используем его.
    Иначе пытаемся мёржить по (symbol, entry_dt).
    Если merge неудачен, возвращаем snapshots как основной источник, но с предупреждением.
    """
    df = snaps.copy()

    # нормализуем даты
    for col in ("entry_dt", "exit_dt"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("entry_dt", "exit_dt"):
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], errors="coerce")

    if "symbol" in df.columns and "symbol" in trades.columns:
        if "trade_id" in df.columns and "trade_id" in trades.columns:
            merged = pd.merge(
                df,
                trades[["symbol", "trade_id", "pnl_rel", "pnl_abs", "bars_in_trade", "max_adverse_excursion"]],
                on=["symbol", "trade_id"],
                how="left",
                suffixes=("", "_trades"),
            )
            return merged
        elif "entry_dt" in df.columns and "entry_dt" in trades.columns:
            merged = pd.merge(
                df,
                trades[["symbol", "entry_dt", "pnl_rel", "pnl_abs", "bars_in_trade", "max_adverse_excursion"]],
                on=["symbol", "entry_dt"],
                how="left",
                suffixes=("", "_trades"),
            )
            return merged

    # fallback: если не смогли смёржить, но в снапшотах уже есть pnl_rel — используем их как есть
    return df


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pnl_rel" not in df.columns:
        df["pnl_rel"] = 0.0
    df["label_good_simple"] = df["pnl_rel"] > 0
    df["label_good_strict"] = df["pnl_rel"] > 0.001
    df["label_big_loss"] = df["pnl_rel"] < -0.01
    return df


def _compute_pf(pnl: pd.Series) -> float:
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def _univariate_stats(df: pd.DataFrame, feature: str, n_bins: int = 10, min_trades_bin: int = 10) -> pd.DataFrame:
    if feature not in df.columns:
        return pd.DataFrame()

    data = df[[feature, "pnl_rel"]].dropna()
    if data.empty:
        return pd.DataFrame()

    # защищаемся от вырожденных распределений
    try:
        bins = pd.qcut(data[feature], q=n_bins, duplicates="drop")
    except Exception:
        # fallback: равные интервалы
        bins = pd.cut(data[feature], bins=n_bins, duplicates="drop")

    grouped = data.groupby(bins, observed=True)
    rows = []
    for interval, g in grouped:
        count = len(g)
        if count < min_trades_bin:
            continue
        pnl = g["pnl_rel"]
        win_rate = float((pnl > 0).mean())
        mean_pnl = float(pnl.mean())
        pf = _compute_pf(pnl)
        rows.append(
            {
                "bin": str(interval),
                "bin_left": getattr(interval, "left", None),
                "bin_right": getattr(interval, "right", None),
                "count_trades": count,
                "win_rate": win_rate,
                "mean_pnl_rel": mean_pnl,
                "pf": pf,
            }
        )

    return pd.DataFrame(rows)


def _bivariate_stats(
    df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    n_bins_x: int = 5,
    n_bins_y: int = 5,
    min_trades_bin: int = 10,
) -> pd.DataFrame:
    if feature_x not in df.columns or feature_y not in df.columns:
        return pd.DataFrame()

    data = df[[feature_x, feature_y, "pnl_rel"]].dropna()
    if data.empty:
        return pd.DataFrame()

    try:
        bins_x = pd.qcut(data[feature_x], q=n_bins_x, duplicates="drop")
    except Exception:
        bins_x = pd.cut(data[feature_x], bins=n_bins_x, duplicates="drop")

    try:
        bins_y = pd.qcut(data[feature_y], q=n_bins_y, duplicates="drop")
    except Exception:
        bins_y = pd.cut(data[feature_y], bins=n_bins_y, duplicates="drop")

    df_bins = data.copy()
    df_bins["bin_x"] = bins_x
    df_bins["bin_y"] = bins_y

    grouped = df_bins.groupby(["bin_x", "bin_y"], observed=True)
    rows = []
    for (bx, by), g in grouped:
        count = len(g)
        if count < min_trades_bin:
            continue
        pnl = g["pnl_rel"]
        win_rate = float((pnl > 0).mean())
        mean_pnl = float(pnl.mean())
        pf = _compute_pf(pnl)
        rows.append(
            {
                "bin_x": str(bx),
                "bin_y": str(by),
                "count_trades": count,
                "win_rate": win_rate,
                "mean_pnl_rel": mean_pnl,
                "pf": pf,
            }
        )

    return pd.DataFrame(rows)


def _time_stats_year_month(df: pd.DataFrame) -> pd.DataFrame:
    if "entry_dt" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["entry_dt"] = pd.to_datetime(d["entry_dt"], errors="coerce")
    d = d.dropna(subset=["entry_dt"])
    d["year"] = d["entry_dt"].dt.year
    d["month"] = d["entry_dt"].dt.month
    grouped = d.groupby(["year", "month"], observed=True)
    rows = []
    for (year, month), g in grouped:
        pnl = g["pnl_rel"]
        rows.append(
            {
                "year": int(year),
                "month": int(month),
                "count_trades": len(g),
                "win_rate": float((pnl > 0).mean()),
                "mean_pnl_rel": float(pnl.mean()),
                "pf": _compute_pf(pnl),
            }
        )
    return pd.DataFrame(rows)


def _time_stats_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if "entry_dt" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["entry_dt"] = pd.to_datetime(d["entry_dt"], errors="coerce")
    d = d.dropna(subset=["entry_dt"])
    d["hour"] = d["entry_dt"].dt.hour
    grouped = d.groupby("hour", observed=True)
    rows = []
    for hour, g in grouped:
        pnl = g["pnl_rel"]
        rows.append(
            {
                "hour": int(hour),
                "count_trades": len(g),
                "win_rate": float((pnl > 0).mean()),
                "mean_pnl_rel": float(pnl.mean()),
                "pf": _compute_pf(pnl),
            }
        )
    return pd.DataFrame(rows)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _save_report(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        return
    _ensure_dir(path)
    df.to_csv(path, index=False)


def main(args):
    snapshots_path: str = args.snapshots
    trades_path: str = args.trades
    out_prefix: str = args.out_prefix

    snaps = _load_csv(snapshots_path)
    trades = _load_csv(trades_path)

    df = _merge_trades_snapshots(trades, snaps)
    df = _add_labels(df)

    cfg = EDAConfig(
        features_univariate=[
            "band_pos",
            "dist_from_ma",
            "atr_14_pct",
            "bar_range_pct",
            "bar_body_pct",
        ],
        feature_pairs_bivariate=[
            ("band_pos", "atr_14_pct"),
            ("band_pos", "bar_range_pct"),
            ("dist_from_ma", "atr_14_pct"),
        ],
        min_trades_bin=10,
    )

    # Univariate reports
    for feat in cfg.features_univariate:
        uni_df = _univariate_stats(df, feat, n_bins=10, min_trades_bin=cfg.min_trades_bin)
        out_path = f"{out_prefix}_univariate_{feat}.csv"
        _save_report(uni_df, out_path)

    # Bivariate reports
    for fx, fy in cfg.feature_pairs_bivariate:
        bi_df = _bivariate_stats(df, fx, fy, n_bins_x=5, n_bins_y=5, min_trades_bin=cfg.min_trades_bin)
        out_path = f"{out_prefix}_bivariate_{fx}_{fy}.csv"
        _save_report(bi_df, out_path)

    # Time-based reports
    ym_df = _time_stats_year_month(df)
    _save_report(ym_df, f"{out_prefix}_time_year_month.csv")

    intraday_df = _time_stats_intraday(df)
    _save_report(intraday_df, f"{out_prefix}_time_intraday.csv")
