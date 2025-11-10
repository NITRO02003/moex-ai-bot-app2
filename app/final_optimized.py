# app/final_optimized.py
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

from .features import build_features_10m
from .ai_models import AIStrategyModel, AIRiskModel
from .config import config
from .utils import load_all, slice_by_date, ensure_dtindex_utc
from .core import TradeCosts, calculate_performance_metrics
from .features import FINAL_FEATURE_SET


@dataclass
class OptimizedStrategy:
    """Финальная оптимизированная стратегия"""

    # Оптимизированные параметры на основе анализа
    min_signal_strength: float = 0.45
    min_volume_factor: float = 1.4
    max_volatility: float = 0.035
    min_trend_strength: float = 0.12
    momentum_threshold: float = 0.0025
    base_position_size: float = 0.01
    max_position_size: float = 0.05

    def apply_filters(self, features_row: pd.Series, signal: float) -> bool:
        """Строгие фильтры для качественных сигналов"""
        if abs(signal) < self.min_signal_strength:
            return False

        filters = [
            features_row.get('volume_ratio', 1) > self.min_volume_factor,
            features_row.get('realized_vol', 0) < self.max_volatility,
            abs(features_row.get('trend_strength', 0)) > self.min_trend_strength,
            abs(features_row.get('momentum_5', 0)) > self.momentum_threshold,
            features_row.get('price_efficiency', 0) > -0.6,
            features_row.get('liquidity_score', 0) > 100000,  # Фильтр ликвидности
        ]

        return all(filters)

    def calculate_position_size(self, signal: float, features_row: pd.Series,
                                risk_score: float, portfolio_value: float) -> float:
        """Консервативный расчет размера позиции"""
        # Базовый размер
        size = self.base_position_size

        # Усиление силой тренда (осторожно)
        trend_strength = abs(features_row.get('trend_strength', 0))
        trend_boost = 1.0 + min(trend_strength * 0.3, 0.3)

        # Корректировка на волатильность
        volatility = features_row.get('realized_vol', 0.02)
        vol_adjustment = 0.02 / max(volatility, 0.015)

        # Корректировка на риск
        risk_adjustment = risk_score

        final_size = size * trend_boost * vol_adjustment * risk_adjustment * abs(signal)
        return min(final_size, self.max_position_size)


def run_optimized_backtest():
    """Запуск финального оптимизированного бэктеста"""
    print("[start] FINAL OPTIMIZED STRATEGY BACKTEST")

    # Загрузка данных
    data_dir = "data"
    symbols = config.symbols_cfg.symbols

    data = load_all(data_dir, symbols)
    for symbol in list(data.keys()):
        data[symbol] = slice_by_date(data[symbol], config.bt_cfg.start_date, config.bt_cfg.end_date)
        data[symbol] = ensure_dtindex_utc(data[symbol])

    # AI модели с высоким порогом
    ai_strategy = AIStrategyModel(threshold=0.55)
    ai_risk = AIRiskModel()

    # Оптимизированная стратегия
    strategy = OptimizedStrategy()

    # Создаем фичи
    feats = {}
    for symbol, df in data.items():
        feats[symbol] = build_features_10m(df)
        # Добавляем volume_ratio
        feats[symbol]['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Общий индекс
    all_indices = [feats[s].index for s in symbols if s in feats and not feats[s].empty]
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.union(idx)
    common_index = common_index.sort_values()

    # Симуляция
    initial_equity = config.bt_cfg.initial_equity
    cash = initial_equity
    positions = {s: 0.0 for s in symbols}
    trade_log = []
    equity_curve = []

    print(f"Running optimized backtest with {len(common_index)} bars...")

    for i, timestamp in enumerate(common_index):
        if i % 100 == 0 and i > 0:
            print(f"Progress: {i}/{len(common_index)}")

        total_value = cash
        for symbol in symbols:
            if symbol in data and timestamp in data[symbol].index:
                total_value += positions[symbol] * data[symbol].loc[timestamp, 'close']

        for symbol in symbols:
            if (symbol not in feats or timestamp not in feats[symbol].index or
                    timestamp not in data[symbol].index):
                continue

            features_row = feats[symbol].loc[timestamp]
            price = data[symbol].loc[timestamp, 'open']

            # AI сигнал
            if ai_strategy.available():
                feature_cols = [col for col in features_row.index if col in FINAL_FEATURE_SET]
                if feature_cols:
                    X = pd.DataFrame([features_row[feature_cols]], columns=feature_cols)
                    signal = ai_strategy.predict_series(X).iloc[0]
                else:
                    signal = 0.0
            else:
                signal = 0.0

            risk_score = ai_risk.get_series(pd.DatetimeIndex([timestamp])).iloc[0]

            # Применяем оптимизированные фильтры
            if strategy.apply_filters(features_row, signal):
                position_size = strategy.calculate_position_size(
                    signal, features_row, risk_score, total_value
                )
                target_value = signal * position_size * total_value
                target_quantity = target_value / price if price > 0 else 0
            else:
                target_quantity = 0

            current_qty = positions.get(symbol, 0)
            quantity_delta = target_quantity - current_qty

            if abs(quantity_delta * price) > 3000:  # Минимум 3000 руб
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
                    "signal": signal,
                    "commission": commission
                })

        current_equity = cash
        for symbol in symbols:
            if symbol in data and timestamp in data[symbol].index:
                current_equity += positions[symbol] * data[symbol].loc[timestamp, 'close']
        equity_curve.append((timestamp, current_equity))

    # Результаты
    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    metrics = calculate_performance_metrics(equity_df["equity"])

    print(f"\n=== OPTIMIZED RESULTS ===")
    print(f"Final Equity: {equity_df['equity'].iloc[-1]:.2f}")
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Total Trades: {len(trades_df)}")

    # Сохраняем
    out_dir = Path("out")
    equity_df.to_csv(out_dir / "equity_optimized.csv")
    if not trades_df.empty:
        trades_df.to_csv(out_dir / "trades_optimized.csv", index=False)

    return metrics


if __name__ == "__main__":
    run_optimized_backtest()