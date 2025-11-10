# app/fixed_backtest.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
import os
import sys
import json
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "app"

from .stationary_features import calculate_stationary_features
from .ai_models import AIStrategyModel, AIRiskModel
from .config import config
from .utils import load_all, slice_by_date, ensure_dtindex_utc
from .core import TradeCosts, calculate_performance_metrics

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    positions: Dict
    metrics: Dict


def run_fixed_backtest() -> BacktestResult:
    """Бэктест с исправленными фичами и моделью"""
    print("[start] FIXED MODEL BACKTEST")

    # Загрузка данных
    data_dir = "data"
    symbols = config.symbols_cfg.symbols

    data = load_all(data_dir, symbols)
    for symbol in list(data.keys()):
        data[symbol] = slice_by_date(data[symbol], config.bt_cfg.start_date, config.bt_cfg.end_date)
        data[symbol] = ensure_dtindex_utc(data[symbol])
        print(f"Loaded {symbol}: {len(data[symbol])} rows")

    # Создаем стационарные фичи
    print("[features] computing stationary features...")
    feats = {}
    for symbol, df in data.items():
        feats[symbol] = calculate_stationary_features(df)
        print(f"Stationary features for {symbol}: {feats[symbol].shape}")

    # Загружаем исправленную модель
    try:
        ai_strategy = AIStrategyModel("models/ai_strategy_fixed.pkl")
        print("[ai] Using fixed model")
    except:
        print("[ai] ❌ Fixed model not found, training...")
        from .emergency_retrain_fixed import emergency_retrain_fixed
        emergency_retrain_fixed()
        ai_strategy = AIStrategyModel("models/ai_strategy_fixed.pkl")

    ai_risk = AIRiskModel()

    # Получаем общий индекс времени
    all_indices = [feats[s].index for s in symbols if s in feats and not feats[s].empty]
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
        if i % 500 == 0 and i > 0:
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

            # Генерируем AI сигнал (-1, 0, 1)
            if ai_strategy.available():
                # Исключаем нечисловые колонки
                feature_cols = [col for col in features_row.index if col not in ['symbol', 'close']]
                X = pd.DataFrame([features_row[feature_cols]], columns=feature_cols)
                signal = ai_strategy.predict_series(X).iloc[0]
            else:
                signal = 0.0

            # Получаем риск-скор
            risk_score = ai_risk.get_series(pd.DatetimeIndex([timestamp])).iloc[0]

            # Улучшенная логика позиции
            if signal != 0:  # Только если сигнал не "удерживать"
                # Базовый размер с учетом риска
                base_size = 0.015
                position_size = base_size * risk_score * abs(signal)

                target_value = signal * position_size * total_portfolio_value
                target_quantity = target_value / price if price > 0 else 0
            else:
                target_quantity = 0

            # Исполняем торговлю если нужно
            current_qty = positions.get(symbol, 0)
            quantity_delta = target_quantity - current_qty

            # Минимальный порог для торговли
            min_trade_value = 5000
            if abs(quantity_delta * price) > min_trade_value:
                costs = TradeCosts(
                    commission_per_side=config.costs_cfg.commission_per_side,
                    tick_size=config.symbols_cfg.tick_size,
                    typical_spread_ticks=config.symbols_cfg.typical_spread_ticks
                )

                spread_ticks = config.symbols_cfg.typical_spread_ticks.get(symbol, 2)
                slip_price = spread_ticks * config.symbols_cfg.tick_size.get(symbol, 0.01)
                execution_price = price + (slip_price if quantity_delta > 0 else -slip_price)

                commission = abs(execution_price * quantity_delta) * costs.commission_per_side
                cash -= execution_price * quantity_delta + commission
                positions[symbol] = target_quantity

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

        # Рассчитываем equity
        current_equity = cash
        for symbol in symbols:
            if symbol in data and timestamp in data[symbol].index:
                current_equity += positions[symbol] * data[symbol].loc[timestamp, 'close']
        equity_curve.append((timestamp, current_equity))

    # Результаты
    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    metrics = calculate_performance_metrics(equity_df["equity"])

    print(f"\n=== FIXED MODEL RESULTS ===")
    print(f"Final Equity: {equity_df['equity'].iloc[-1]:.2f}")
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Total Trades: {len(trades_df)}")

    # Анализ сигналов
    if not trades_df.empty and 'signal' in trades_df.columns:
        signal_stats = trades_df['signal'].value_counts()
        print(f"Signal distribution: {signal_stats.to_dict()}")

    # Сохраняем
    equity_df.to_csv(OUT_DIR / "equity_fixed.csv")
    if not trades_df.empty:
        trades_df.to_csv(OUT_DIR / "trades_fixed.csv", index=False)

    with open(OUT_DIR / "metrics_fixed.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return BacktestResult(
        equity=equity_df,
        trades=trades_df,
        positions=positions,
        metrics=metrics
    )


if __name__ == "__main__":
    result = run_fixed_backtest()