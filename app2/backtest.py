# backtest.py - ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ
from __future__ import annotations
import json, numpy as np, pandas as pd
from dataclasses import dataclass
from .paths import REPORTS_DIR
from . import strategy as S, metrics as MX
from .correct_metrics import calculate_correct_metrics  # ИМПОРТ ИСПРАВЛЕННЫХ МЕТРИК


@dataclass
class BtParams:
    commission: float = 0.0005
    slippage_bps: float = 1.0
    horizon: int = 1


def run_symbol(prices: pd.DataFrame, model_bundle, rp, bt: BtParams, equity0: float = 1_000_000.0,
               threshold: float = 0.5):
    from . import risk as R

    sig = S.signal_and_size(prices, model_bundle, rp, equity0, threshold=threshold)
    close = prices['close'].astype(float)

    # КОРРЕКТНЫЙ РАСЧЕТ ПОЗИЦИИ И PnL
    shares = (sig['size'].abs() / close).fillna(0.0) * sig['side']
    position_value = shares * close
    position_value_prev = shares.shift(1) * close.shift(1)
    gross = (position_value - position_value_prev).fillna(0.0)

    # КОМИССИИ И ПРОСКАЛЬЗЫВАНИЕ
    d_sh = shares.diff().abs().fillna(0.0)
    cost = (d_sh * close) * bt.commission + (d_sh * close) * (bt.slippage_bps / 10000.0)

    pnl = gross - cost

    # РАСЧЕТ КАПИТАЛА С ЗАЩИТОЙ
    equity = equity0 + pnl.cumsum()

    # ЗАЩИТА ОТ БАНКРОТСТВА
    bankruptcy = (equity < equity0 * 0.2)
    if bankruptcy.any():
        first_bankrupt = bankruptcy[bankruptcy].index[0]
        pnl.loc[first_bankrupt:] = 0
        equity = equity0 + pnl.cumsum()

    # ПРИМЕНЕНИЕ РИСК-МЕНЕДЖМЕНТА
    pnl = R.apply_daily_risk_cap(pnl, equity0, rp)
    equity = equity0 + pnl.cumsum()

    # ДЕТАЛЬНАЯ ДИАГНОСТИКА
    trades = extract_trades_from_signals(sig, pnl, close, bt)
    corrected_metrics = calculate_correct_metrics(trades, equity0)

    # ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ
    if validate_backtest_results(equity, pnl, corrected_metrics):
        print("✅ Бэктест прошел валидацию")
    else:
        print("⚠️  Предупреждение: возможны аномалии в расчетах")

    return {
        'signals': sig,
        'equity': equity,
        'pnl': pnl,
        'metrics': corrected_metrics,  # ИСПОЛЬЗУЕМ ИСПРАВЛЕННЫЕ МЕТРИКИ
        'trades': trades
    }


def extract_trades_from_signals(signals, pnl, close, bt: BtParams) -> list:
    """Извлечение информации о сделках для метрик"""
    trades = []
    position_changes = signals['side'].diff().fillna(0)
    entry_points = position_changes[position_changes != 0]

    current_trade = None

    for idx, change in entry_points.items():
        if current_trade is None and change != 0:
            # Начало новой сделки
            current_trade = {
                'entry_time': idx,
                'side': signals.loc[idx, 'side'],
                'entry_price': close.loc[idx],
                'size': signals.loc[idx, 'size']
            }
        elif current_trade is not None and change != 0:
            # Закрытие текущей сделки и открытие новой
            current_trade['exit_time'] = idx
            current_trade['exit_price'] = close.loc[idx]
            current_trade['gross_pnl'] = calculate_trade_pnl(current_trade)
            current_trade['commission'] = calculate_trade_commission(current_trade, bt)
            current_trade['net_pnl'] = current_trade['gross_pnl'] - current_trade['commission']

            trades.append(current_trade)

            # Новая сделка
            current_trade = {
                'entry_time': idx,
                'side': signals.loc[idx, 'side'],
                'entry_price': close.loc[idx],
                'size': signals.loc[idx, 'size']
            }

    return trades


def calculate_trade_pnl(trade: dict) -> float:
    """Расчет PnL для одной сделки"""
    if trade['side'] == 1:  # Long
        return (trade['exit_price'] - trade['entry_price']) * (abs(trade['size']) / trade['entry_price'])
    elif trade['side'] == -1:  # Short
        return (trade['entry_price'] - trade['exit_price']) * (abs(trade['size']) / trade['entry_price'])
    return 0.0


def calculate_trade_commission(trade: dict, bt: BtParams) -> float:
    """Расчет комиссий для сделки"""
    trade_value = abs(trade['size'])
    return trade_value * bt.commission


def validate_backtest_results(equity, pnl, metrics) -> bool:
    """Валидация результатов бэктеста"""
    checks = [
        equity.iloc[-1] >= 0,  # Капитал не отрицательный
        metrics['total_trades'] >= 0,  # Корректное количество сделок
        abs(metrics['net_pnl'] - (equity.iloc[-1] - equity.iloc[0])) < 1,  # Сходимость PnL
    ]
    return all(checks)


def save_report(name: str, res: dict):
    path = REPORTS_DIR / f"{name}_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res['metrics'], f, ensure_ascii=False, indent=2)
    return str(path)