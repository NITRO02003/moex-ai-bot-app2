from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RuleBtParams:
    """
    Параметры риск-менеджмента для rule-based бэктеста.
    """

    commission: float = 0.0005        # комиссия (на круг) от объёма сделки
    slippage_bps: float = 1.0         # проскальзывание в б.п. от объёма
    risk_per_trade: float = 0.0015    # доля капитала под риск на сделку (0.0015 = 0.15%)
    atr_len: int = 14                 # длина ATR
    sl_atr_mult: float = 1.5          # стоп в ATR
    tp_mult: float = 3.0              # тейк-профит в ATR
    max_leverage: float = 3.0         # максимально допустимое плечо по позиции


def _compute_atr(df: pd.DataFrame, atr_len: int) -> pd.Series:
    """
    Рассчитывает ATR по классической формуле True Range.
    Ожидает колонки: high, low, close.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_len, min_periods=1).mean()
    return atr


def _compute_metrics(equity: pd.Series, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Считает базовые метрики по эквити и списку сделок.
    """
    metrics: Dict[str, Any] = {}

    if equity.empty:
        metrics.update(
            total_return=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            volatility_ann=0.0,
            sharpe_ann=0.0,
            trade_count=0,
            win_rate=0.0,
            avg_trade=0.0,
            pnl_sum=0.0,
            pnl_mean=0.0,
            pnl_std=0.0,
        )
        return metrics

    equity0 = float(equity.iloc[0])
    equity_end = float(equity.iloc[-1])
    total_return = equity_end / equity0 - 1.0 if equity0 > 0 else 0.0

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0

    # доходности по барам
    rets = equity.pct_change(fill_method=None).fillna(0.0)
    vol = float(rets.std()) * np.sqrt(252.0) if len(rets) > 1 else 0.0
    mean_ret = float(rets.mean()) if len(rets) > 0 else 0.0
    sharpe = (mean_ret / rets.std() * np.sqrt(252.0)) if rets.std() > 0 else 0.0

    if max_dd < 0:
        calmar = total_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
    else:
        calmar = 0.0

    # по сделкам
    trade_count = len(trades)
    pnl_list = [t["pnl_abs"] for t in trades] if trades else []
    pnl_sum = float(sum(pnl_list)) if pnl_list else 0.0
    pnl_mean = float(np.mean(pnl_list)) if pnl_list else 0.0
    pnl_std = float(np.std(pnl_list)) if pnl_list else 0.0

    wins = [p for p in pnl_list if p > 0]
    win_rate = len(wins) / trade_count if trade_count > 0 else 0.0
    avg_trade = pnl_mean

    metrics.update(
        total_return=total_return,
        max_drawdown=max_dd,
        calmar=calmar,
        volatility_ann=vol,
        sharpe_ann=sharpe,
        trade_count=trade_count,
        win_rate=win_rate,
        avg_trade=avg_trade,
        pnl_sum=pnl_sum,
        pnl_mean=pnl_mean,
        pnl_std=pnl_std,
    )

    return metrics


def run_rule_symbol(
    df: pd.DataFrame,
    params: RuleBtParams,
    equity0: float = 1_000_000.0,
    collect_bar_stats: bool = False,
    collect_trades: bool = False,
) -> Dict[str, Any]:
    """
    Базовый бэктест по одному символу.

    Ожидаемый input DataFrame:
      - datetime или begin (временная колонка)
      - open, high, low, close
      - signal (-1, 0, +1) — target-позиция стратегии на баре

    Параметры:
      - params: RuleBtParams с настройками риск-менеджмента
      - equity0: стартовый капитал
      - collect_bar_stats: если True, возвращает bar_stats DataFrame
      - collect_trades: если True, возвращает trades DataFrame

    Возвращает dict:
      - "equity_curve": pd.Series
      - "metrics": dict
      - опционально "bar_stats": pd.DataFrame
      - опционально "trades": pd.DataFrame
    """
    df = df.copy()

    # нормализуем временную колонку
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        dt = df["datetime"]
    elif "begin" in df.columns:
        df["begin"] = pd.to_datetime(df["begin"])
        df = df.sort_values("begin").reset_index(drop=True)
        dt = df["begin"]
    else:
        # fallback — используем индекс
        dt = pd.to_datetime(df.index)

    # базовая проверка обязательных колонок
    required_cols = {"close", "signal"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"run_rule_symbol: missing columns: {missing}")

    # ATR (если есть ohlc, иначе ATR не используется для стопов)
    if {"high", "low", "close"}.issubset(df.columns):
        atr = _compute_atr(df, params.atr_len)
    else:
        atr = pd.Series(np.nan, index=df.index)

    close = df["close"].astype(float)
    signal = df["signal"].fillna(0).astype(float)

    equity_base = float(equity0)   # эквити без учёта текущей нереализованной сделки
    equity_curve: List[float] = []
    equity_times: List[pd.Timestamp] = []

    peak_equity = equity_base
    drawdowns: List[float] = []

    # состояние позиции
    position_qty = 0.0
    position_dir = 0   # 1 = long, -1 = short, 0 = нет позиции
    entry_price = 0.0
    entry_equity = equity_base
    entry_dt: Optional[pd.Timestamp] = None
    trade_id = 0

    # MAE/MFE
    max_favorable = 0.0
    max_adverse = 0.0
    bars_in_trade = 0

    trades: List[Dict[str, Any]] = []

    # bar-level лог
    bar_rows: List[Dict[str, Any]] = []

    for i in range(len(df)):
        t = dt.iloc[i]
        price = float(close.iloc[i])
        sig = int(np.sign(signal.iloc[i]))  # -1/0/+1
        atr_i = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else None

        # ----- обработка выхода из позиции -----
        if position_dir != 0:
            # проверяем стоп/тейк по ATR, если ATR есть
            exit_reason = None
            if atr_i is not None and atr_i > 0:
                move = (price - entry_price) * position_dir  # >0 = прибыль, <0 = убыток
                if move <= -params.sl_atr_mult * atr_i:
                    exit_reason = "stop_atr"
                elif move >= params.tp_mult * atr_i:
                    exit_reason = "take_atr"

            # проверяем смену сигнала
            if exit_reason is None:
                if sig == 0 or sig == -position_dir:
                    exit_reason = "signal_change"

            if exit_reason is not None:
                # считаем PnL по сделке
                qty = abs(position_qty)
                direction = position_dir
                price_diff = (price - entry_price) * direction
                pnl_abs = price_diff * qty

                # комиссия и slippage
                trade_volume = qty * price
                fee_commission = trade_volume * params.commission * 2.0
                fee_slippage = trade_volume * (params.slippage_bps / 10000.0)
                fees = fee_commission + fee_slippage

                pnl_abs -= fees
                pnl_rel = pnl_abs / entry_equity if entry_equity != 0 else 0.0

                trade_row = {
                    "trade_id": trade_id,
                    "entry_dt": entry_dt,
                    "exit_dt": t,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "direction": direction,
                    "qty": qty,
                    "pnl_abs": pnl_abs,
                    "pnl_rel": pnl_rel,
                    "bars_in_trade": bars_in_trade + 1,
                    "max_favorable_excursion": max_favorable,
                    "max_adverse_excursion": max_adverse,
                    "exit_reason": exit_reason,
                }
                trades.append(trade_row)

                equity_base += pnl_abs  # реализованный результат сделки
                position_qty = 0.0
                position_dir = 0
                entry_price = 0.0
                entry_equity = equity_base
                entry_dt = None
                max_favorable = 0.0
                max_adverse = 0.0
                bars_in_trade = 0

        # ----- открытие новой позиции -----
        if position_dir == 0 and sig != 0:
            # открываем новую сделку
            if atr_i is not None and atr_i > 0 and params.sl_atr_mult > 0:
                risk_capital = equity_base * params.risk_per_trade
                # риск на одну акцию ~ sl_atr_mult * ATR
                qty = risk_capital / (params.sl_atr_mult * atr_i)
                # ограничим плечо
                position_value = qty * price
                max_position_value = equity_base * params.max_leverage
                if position_value > max_position_value and position_value > 0:
                    scale = max_position_value / position_value
                    qty *= scale
            else:
                qty = 0.0

            if qty > 0:
                trade_id += 1
                position_qty = qty
                position_dir = sig
                entry_price = price
                entry_equity = equity_base
                entry_dt = t
                max_favorable = 0.0
                max_adverse = 0.0
                bars_in_trade = 0

        # ----- mark-to-market текущего бара -----
        if position_dir != 0:
            # нереализованный PnL
            qty = abs(position_qty)
            direction = position_dir
            price_diff = (price - entry_price) * direction
            unrealized = price_diff * qty
            equity_bar = equity_base + unrealized

            # обновляем MAE/MFE
            if price_diff > max_favorable:
                max_favorable = price_diff
            if price_diff < max_adverse:
                max_adverse = price_diff

            bars_in_trade += 1
        else:
            equity_bar = equity_base

        peak_equity = max(peak_equity, equity_bar)
        dd = equity_bar / peak_equity - 1.0 if peak_equity > 0 else 0.0

        equity_curve.append(equity_bar)
        equity_times.append(t)
        drawdowns.append(dd)

        if collect_bar_stats:
            bar_rows.append(
                {
                    "datetime": t,
                    "close": price,
                    "signal": sig,
                    "position": position_dir * abs(position_qty),
                    "equity": equity_bar,
                    "drawdown": dd,
                    "trade_id": trade_id if position_dir != 0 else np.nan,
                }
            )

    equity_series = pd.Series(equity_curve, index=pd.to_datetime(equity_times))

    metrics = _compute_metrics(equity_series, trades)

    result: Dict[str, Any] = {
        "equity_curve": equity_series,
        "metrics": metrics,
    }

    if collect_bar_stats:
        bar_df = pd.DataFrame(bar_rows)
        result["bar_stats"] = bar_df

    if collect_trades:
        trades_df = pd.DataFrame(trades)
        result["trades"] = trades_df

    return result
