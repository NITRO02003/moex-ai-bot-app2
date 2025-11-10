# backtest.py - ИСПРАВЛЕННАЯ ВЕРСИЯ С ПРАВИЛЬНЫМ ИМПОРТОМ
import os
import sys
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "app"

from .core import TradeCosts, calculate_performance_metrics  # ИМПОРТИРУЕМ ПЕРЕИМЕНОВАННУЮ ФУНКЦИЮ
from .strategy import FeatureStrategy, ParallelStrategyExecutor
from .ai_models import AIStrategyModel, AIRiskModel
from .config import config
from .utils import load_all, slice_by_date, ensure_dtindex_utc
from .features import FINAL_FEATURE_SET

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    positions: Dict
    metrics: Dict


def run_simple_backtest() -> BacktestResult:
    print("[start] simple backtest starting...")

    # Загружаем данные
    data_dir = "data"
    symbols = config.symbols_cfg.symbols

    data = load_all(data_dir, symbols)
    for symbol in list(data.keys()):
        data[symbol] = slice_by_date(data[symbol], config.bt_cfg.start_date, config.bt_cfg.end_date)
        data[symbol] = ensure_dtindex_utc(data[symbol])
        print(f"Loaded {symbol}: {len(data[symbol])} rows")

    # Создаем фичи
    print("[features] computing...")
    strat = FeatureStrategy()
    executor = ParallelStrategyExecutor(strat)
    feats = executor.compute_features(data)

    # AI модели
    ai_strategy = AIStrategyModel()
    ai_risk = AIRiskModel()

    # Получаем общий индекс времени
    all_indices = [feats[s].index for s in symbols if s in feats and not feats[s].empty]
    if not all_indices:
        raise ValueError("No features computed")

    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.union(idx)
    common_index = common_index.sort_values()

    # Инициализация симуляции
    initial_equity = config.bt_cfg.initial_equity
    cash = initial_equity
    positions = {s: 0.0 for s in symbols}
    trade_log = []
    equity_curve = []

    print(f"[backtest] running simulation with {len(common_index)} bars...")

    for i, timestamp in enumerate(common_index):
        if i % 100 == 0 and i > 0:
            print(f"Progress: {i}/{len(common_index)}")

        total_portfolio_value = cash

        # Рассчитываем общую стоимость портфеля
        for symbol in symbols:
            if symbol in data and timestamp in data[symbol].index:
                price = data[symbol].loc[timestamp, 'close']
                total_portfolio_value += positions[symbol] * price

        # Торговая логика для каждого символа
        for symbol in symbols:
            if (symbol not in feats or symbol not in data or
                    timestamp not in feats[symbol].index or timestamp not in data[symbol].index):
                continue

            # Получаем фичи и цену
            features_row = feats[symbol].loc[timestamp]
            price = data[symbol].loc[timestamp, 'open']

            # Генерируем AI сигнал
            if ai_strategy.available():
                feature_cols = [col for col in features_row.index if col in FINAL_FEATURE_SET]
                if feature_cols:
                    X = pd.DataFrame([features_row[feature_cols]], columns=feature_cols)
                    signal = ai_strategy.predict_series(X).iloc[0]
                else:
                    signal = 0.0
            else:
                # Fallback на тренд
                signal = np.sign(features_row.get('trend_strength', 0))

            # Получаем риск-скор
            risk_score = ai_risk.get_series(pd.DatetimeIndex([timestamp])).iloc[0]

            # Рассчитываем целевую позицию (упрощенная логика)
            position_size = 0.02  # 2% от портфеля на сигнал
            target_value = signal * position_size * total_portfolio_value * risk_score
            target_quantity = target_value / price if price > 0 else 0

            # Исполняем торговлю если нужно
            current_qty = positions.get(symbol, 0)
            quantity_delta = target_quantity - current_qty

            # Минимальный порог для торговли
            min_trade_value = 1000  # Минимальная сумма сделки
            if abs(quantity_delta * price) > min_trade_value:
                # Расчет комиссий и проскальзывания
                costs = TradeCosts(
                    commission_per_side=config.costs_cfg.commission_per_side,
                    tick_size=config.symbols_cfg.tick_size,
                    typical_spread_ticks=config.symbols_cfg.typical_spread_ticks
                )

                # Цена исполнения с учетом проскальзывания
                spread_ticks = config.symbols_cfg.typical_spread_ticks.get(symbol, 2)
                slip_ticks = spread_ticks + 1  # дополнительное проскальзывание
                slip_price = slip_ticks * config.symbols_cfg.tick_size.get(symbol, 0.01)

                execution_price = costs.round_price(
                    symbol,
                    price + (slip_price if quantity_delta > 0 else -slip_price)
                )

                # Комиссия
                commission = abs(execution_price * quantity_delta) * costs.commission_per_side

                # Обновляем кэш и позицию
                cash -= execution_price * quantity_delta + commission
                positions[symbol] = target_quantity

                # Логируем сделку
                trade_log.append({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "BUY" if quantity_delta > 0 else "SELL",
                    "quantity": abs(quantity_delta),
                    "price": execution_price,
                    "commission": commission,
                    "signal": signal,
                    "risk_score": risk_score
                })

        # Рассчитываем equity для этого timestamp
        current_equity = cash
        for symbol in symbols:
            if symbol in data and timestamp in data[symbol].index:
                current_equity += positions[symbol] * data[symbol].loc[timestamp, 'close']

        equity_curve.append((timestamp, current_equity))

    # Создаем результаты
    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    # Расчет метрик
    metrics = {}
    if not equity_df.empty:
        equity_series = equity_df["equity"]
        metrics = calculate_performance_metrics(equity_series)  # ИСПОЛЬЗУЕМ ПЕРЕИМЕНОВАННУЮ ФУНКЦИЮ

    print(
        f"[done] backtest finished. Trades: {len(trades_df)}, Final equity: {equity_df['equity'].iloc[-1] if not equity_df.empty else initial_equity:.2f}")

    # Сохраняем результаты
    equity_df.to_csv(os.path.join(OUT_DIR, "equity.csv"))
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return BacktestResult(
        equity=equity_df,
        trades=trades_df,
        positions=positions,
        metrics=metrics
    )


if __name__ == "__main__":
    result = run_simple_backtest()
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
    print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {result.metrics.get('win_rate', 0):.2%}")
    print(
        f"Final Equity: {result.equity['equity'].iloc[-1] if not result.equity.empty else config.bt_cfg.initial_equity:.2f}")
    print(f"Total Trades: {len(result.trades)}")