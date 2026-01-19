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
    tdf = trades_df.copy()
    tdf["entry_dt"] = pd.to_datetime(tdf["entry_dt"])

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
