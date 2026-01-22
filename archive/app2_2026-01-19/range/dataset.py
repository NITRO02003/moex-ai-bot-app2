from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


TRADE_REQUIRED_KEYS = [
    "trade_id",
    "entry_dt",
    "exit_dt",
    "entry_price",
    "exit_price",
    "direction",
    "qty",
    "pnl_abs",
    "pnl_rel",
    "bars_in_trade",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "exit_reason",
]

V3_TRADE_REQUIRED_KEYS = [
    "trade_id",
    "entry_dt",
    "exit_dt",
    "entry_price",
    "exit_price",
    "direction",
    "qty",
    "pnl_abs",
    "pnl_rel",
    "bars_in_trade",
    "exit_reason",
]


def trades_to_frame(trades: Sequence[Dict[str, Any]], symbol: str) -> pd.DataFrame:
    """Преобразует список сделок в DataFrame с единым форматом.

    Ожидается структура сделок, аналогичная output из rule_core.run_rule_symbol.
    """
    if not trades:
        return pd.DataFrame(columns=["symbol"] + TRADE_REQUIRED_KEYS)

    df = pd.DataFrame(trades).copy()
    # гарантируем наличие всех ключей (отсутствующие → NaN)
    for k in TRADE_REQUIRED_KEYS:
        if k not in df.columns:
            df[k] = np.nan

    df["symbol"] = symbol
    # конвертируем даты
    for col in ("entry_dt", "exit_dt"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df[["symbol"] + TRADE_REQUIRED_KEYS]


def v3_trades_to_frame(trades_df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Normalize Range V3 trades (legacy/core) into a common schema for datasets."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["symbol"] + V3_TRADE_REQUIRED_KEYS)

    df = trades_df.copy()
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    elif symbol is not None:
        df["symbol"] = symbol

    df["trade_id"] = df.index.astype(int)
    df["entry_dt"] = pd.to_datetime(df.get("entry_time"), errors="coerce")
    df["exit_dt"] = pd.to_datetime(df.get("exit_time"), errors="coerce")
    df["direction"] = df.get("side")
    df["pnl_abs"] = df.get("pnl")
    df["bars_in_trade"] = df.get("bars_held")
    df["exit_reason"] = df.get("exit_reason")

    required = ["entry_price", "exit_price", "qty", "pnl_rel"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    base_cols = ["symbol"] + V3_TRADE_REQUIRED_KEYS
    # Keep selected extra columns (entry_* analytics, circuit breaker flags, etc.)
    extra_cols = []
    for col in df.columns:
        if col in base_cols:
            continue
        if col in ("entry_time", "exit_time", "side", "pnl", "bars_held"):
            continue
        if col.startswith(("entry_", "exit_", "post_", "geo_")):
            extra_cols.append(col)

    cols = base_cols + extra_cols
    return df[cols]


def snapshots_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure snapshots have 'datetime' column for dataset joins."""
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        for cand in ("begin", "time", "date", "timestamp", "ts"):
            if cand in out.columns:
                out["datetime"] = pd.to_datetime(out[cand], errors="coerce")
                break
        if "datetime" not in out.columns and len(out.columns) > 0:
            first_col = out.columns[0]
            if str(first_col).lower().startswith("unnamed"):
                parsed = pd.to_datetime(out[first_col], errors="coerce")
                if parsed.notna().mean() >= 0.5:
                    out["datetime"] = parsed
        if "datetime" not in out.columns and out.index is not None:
            try:
                out["datetime"] = pd.to_datetime(out.index, errors="coerce")
            except Exception:
                pass
    if "datetime" in out.columns and getattr(out["datetime"].dt, "tz", None) is not None:
        out["datetime"] = out["datetime"].dt.tz_localize(None)
    return out


def make_entry_snapshots(
    features_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    label_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Строит датасет «снимков» фич на момент входа в сделку.

    - features_df: DataFrame с колонкой `datetime` и набором feature-колонок.
    - trades_df: результат trades_to_frame() или аналогичный формат.
    - feature_cols: явный список фич; если None, берутся все колонки кроме
      служебных (open/high/low/close/volume/datetime/signal и т.п.).
    - label_cols: какие метрики сделки включить как таргеты (по умолчанию
      разумный набор вроде pnl_rel, pnl_abs, bars_in_trade).

    Возвращает DataFrame, где одна строка = одна сделка, колонки:
    - символ, даты входа/выхода,
    - выбранные фичи,
    - выбранные label-метрики.
    """
    if features_df.empty or trades_df.empty:
        return pd.DataFrame()

    fdf = features_df.copy()
    if "datetime" not in fdf.columns:
        raise ValueError("features_df must have 'datetime' column")

    fdf["datetime"] = pd.to_datetime(fdf["datetime"])
    if getattr(fdf["datetime"].dt, "tz", None) is not None:
        fdf["datetime"] = fdf["datetime"].dt.tz_localize(None)
    tdf = trades_df.copy()
    tdf["entry_dt"] = pd.to_datetime(tdf["entry_dt"])
    if getattr(tdf["entry_dt"].dt, "tz", None) is not None:
        tdf["entry_dt"] = tdf["entry_dt"].dt.tz_localize(None)

    # Выбор фич по умолчанию: всё, кроме явных price/tech колонок
    if feature_cols is None:
        exclude = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "datetime",
            "signal",
            "symbol",
        }
        feature_cols = [c for c in fdf.columns if c not in exclude]

    feature_cols = list(feature_cols)

    # Таргеты по умолчанию
    if label_cols is None:
        label_cols = ["pnl_rel", "pnl_abs", "bars_in_trade", "max_adverse_excursion"]
    label_cols = list(label_cols)

    # джойним по entry_dt → datetime
    merged = tdf.merge(
        fdf[["datetime"] + feature_cols],
        left_on="entry_dt",
        right_on="datetime",
        how="left",
        suffixes=("", "_feat"),
    )

    # если каких-то фич нет (NaN) — это повод для диагностики, но тут просто оставляем NaN
    cols = (
        ["symbol", "entry_dt", "exit_dt", "direction"]
        + list(feature_cols)
        + list(label_cols)
    )
    # гарантируем наличие label-колонок
    for c in label_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    return merged[cols]


def make_intrade_timeseries(
    features_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    interval: str,
    feature_cols: Optional[Iterable[str]] = None,
    exit_improve_threshold: float = 0.0,
    exit_min_bars: int = 0,
) -> pd.DataFrame:
    """Build bar-level in-trade dataset (Dataset B).

    Each row corresponds to one bar while a trade is open.
    """
    if features_df.empty or trades_df.empty:
        return pd.DataFrame()

    fdf = snapshots_to_features(features_df)
    if "datetime" not in fdf.columns:
        raise ValueError("features_df must have 'datetime' column")

    fdf = fdf.copy()
    fdf["datetime"] = pd.to_datetime(fdf["datetime"], errors="coerce")
    if getattr(fdf["datetime"].dt, "tz", None) is not None:
        fdf["datetime"] = fdf["datetime"].dt.tz_localize(None)
    fdf = fdf.sort_values("datetime").reset_index(drop=True)

    tdf = trades_df.copy()
    if "entry_dt" not in tdf.columns and "entry_time" in tdf.columns:
        tdf["entry_dt"] = pd.to_datetime(tdf["entry_time"], errors="coerce")
    if "exit_dt" not in tdf.columns and "exit_time" in tdf.columns:
        tdf["exit_dt"] = pd.to_datetime(tdf["exit_time"], errors="coerce")
    if "entry_dt" in tdf.columns and getattr(tdf["entry_dt"].dt, "tz", None) is not None:
        tdf["entry_dt"] = tdf["entry_dt"].dt.tz_localize(None)
    if "exit_dt" in tdf.columns and getattr(tdf["exit_dt"].dt, "tz", None) is not None:
        tdf["exit_dt"] = tdf["exit_dt"].dt.tz_localize(None)
    if "trade_id" not in tdf.columns:
        tdf["trade_id"] = np.arange(len(tdf), dtype=int)

    # Feature columns by default: everything except datetime/symbol
    if feature_cols is None:
        exclude = {"datetime", "symbol"}
        feature_cols = [c for c in fdf.columns if c not in exclude]
    feature_cols = list(feature_cols)

    base_trade_cols = {
        "symbol",
        "trade_id",
        "entry_dt",
        "exit_dt",
        "direction",
        "entry_price",
        "exit_price",
        "qty",
        "pnl_abs",
        "pnl_rel",
        "bars_in_trade",
        "exit_reason",
    }
    redundant = {"entry_time", "exit_time", "side", "pnl", "bars_held"}
    extra_trade_cols = [c for c in tdf.columns if c not in base_trade_cols and c not in redundant]

    rows: List[pd.DataFrame] = []
    for _, trade in tdf.iterrows():
        entry_dt = trade.get("entry_dt")
        exit_dt = trade.get("exit_dt")
        if pd.isna(entry_dt) or pd.isna(exit_dt):
            continue

        mask = (fdf["datetime"] >= entry_dt) & (fdf["datetime"] <= exit_dt)
        trade_bars = fdf.loc[mask, ["datetime"] + feature_cols].copy()
        if trade_bars.empty:
            continue

        symbol = trade.get("symbol")
        trade_id = int(trade.get("trade_id", 0))
        trade_uid = f"{symbol}:{trade_id}" if symbol is not None else str(trade_id)

        direction = trade.get("direction")
        if pd.isna(direction):
            direction = 1
        direction = int(direction)

        entry_price = trade.get("entry_price")
        try:
            entry_price = float(entry_price)
        except Exception:
            entry_price = np.nan
        if not np.isfinite(entry_price):
            if "open" in trade_bars.columns:
                entry_price = float(trade_bars["open"].iloc[0])
            else:
                entry_price = float(trade_bars["close"].iloc[0])

        exit_price = trade.get("exit_price")
        try:
            exit_price = float(exit_price)
        except Exception:
            exit_price = np.nan
        if not np.isfinite(exit_price):
            if "open" in trade_bars.columns:
                exit_price = float(trade_bars["open"].iloc[-1])
            else:
                exit_price = float(trade_bars["close"].iloc[-1])

        qty = trade.get("qty")
        try:
            qty = float(qty)
        except Exception:
            qty = np.nan
        if not np.isfinite(qty) or qty <= 0:
            qty = np.nan

        price_col = "close" if "close" in trade_bars.columns else ("open" if "open" in trade_bars.columns else None)
        if price_col is None:
            raise ValueError("features_df must contain at least 'close' or 'open' column")
        price_series = trade_bars[price_col].astype(float)

        if not np.isfinite(entry_price) or entry_price == 0:
            continue
        if not np.isfinite(exit_price):
            continue

        price_diff = (price_series - float(entry_price)) * direction
        pnl_rel = price_diff / float(entry_price)
        pnl_abs = price_diff * qty if np.isfinite(qty) else np.nan

        trade_bars["dt"] = trade_bars["datetime"]
        trade_bars["bars_held"] = np.arange(len(trade_bars), dtype=int)
        trade_bars["time_since_entry_min"] = (
            (trade_bars["dt"] - entry_dt).dt.total_seconds() / 60.0
        )

        trade_bars["pnl_abs"] = pnl_abs
        trade_bars["pnl_rel"] = pnl_rel
        trade_bars["mae_abs"] = pnl_abs.cummin()
        trade_bars["mfe_abs"] = pnl_abs.cummax()
        trade_bars["mae_rel"] = pnl_rel.cummin()
        trade_bars["mfe_rel"] = pnl_rel.cummax()

        trade_exit_rel = (
            (float(exit_price) - float(entry_price)) * direction / float(entry_price)
            if np.isfinite(entry_price) and entry_price != 0
            else np.nan
        )
        pnl_rel_next = pnl_rel.shift(-1)
        y_exit = (
            pnl_rel_next.notna()
            & (pnl_rel_next >= trade_exit_rel + float(exit_improve_threshold))
            & (trade_bars["bars_held"] >= int(exit_min_bars))
        )
        trade_bars["y_exit"] = y_exit.astype(int)

        # Trade-level metadata (constant per bar)
        trade_bars["symbol"] = symbol
        trade_bars["interval"] = interval
        trade_bars["trade_id"] = trade_id
        trade_bars["trade_uid"] = trade_uid
        trade_bars["entry_dt"] = entry_dt
        trade_bars["exit_dt"] = exit_dt
        trade_bars["direction"] = direction
        trade_bars["entry_price"] = entry_price
        trade_bars["exit_price"] = exit_price
        trade_bars["qty"] = qty
        trade_bars["trade_pnl_abs"] = trade.get("pnl_abs")
        trade_bars["trade_pnl_rel"] = trade.get("pnl_rel")
        trade_bars["trade_bars_in_trade"] = trade.get("bars_in_trade")
        trade_bars["trade_exit_reason"] = trade.get("exit_reason")

        for col in extra_trade_cols:
            trade_bars[col] = trade.get(col)

        rows.append(trade_bars)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    # keep consistent column order: meta -> per-bar -> features
    meta_cols = [
        "symbol",
        "interval",
        "trade_id",
        "trade_uid",
        "dt",
        "entry_dt",
        "exit_dt",
        "direction",
        "entry_price",
        "exit_price",
        "qty",
        "trade_pnl_abs",
        "trade_pnl_rel",
        "trade_bars_in_trade",
        "trade_exit_reason",
    ]
    bar_cols = [
        "bars_held",
        "time_since_entry_min",
        "pnl_abs",
        "pnl_rel",
        "mae_abs",
        "mfe_abs",
        "mae_rel",
        "mfe_rel",
        "y_exit",
    ]
    extra_cols = [c for c in extra_trade_cols if c in out.columns]
    feature_cols = [c for c in feature_cols if c in out.columns]
    ordered = [c for c in meta_cols + extra_cols + bar_cols + feature_cols if c in out.columns]
    return out[ordered]
