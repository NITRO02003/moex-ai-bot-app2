# app/improved_strategy.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import os
import sys
import json

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "app"

from .features import build_features_10m
from .ai_models import AIStrategyModel, AIRiskModel
from .config import config
from .utils import load_all, slice_by_date, ensure_dtindex_utc
from .core import TradeCosts, calculate_performance_metrics
from .features import FINAL_FEATURE_SET

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    positions: Dict
    metrics: Dict


@dataclass
class ImprovedStrategy:
    """Улучшенная стратегия с фильтрами сигналов"""

    # Фильтры сигналов
    min_volume_factor: float = 1.2
    max_volatility: float = 0.05
    min_trend_strength: float = 0.1
    momentum_threshold: float = 0.002
    min_signal_strength: float = 0.3  # Минимальная сила AI сигнала

    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расширенные фичи для фильтрации"""
        feats = build_features_10m(df)

        # Дополнительные фичи
        close = df['close']
        volume = df['volume']

        # Объемные фичи
        feats['volume_ratio'] = volume / volume.rolling(20).mean()

        # Ценовые фичи
        feats['price_acceleration'] = close.pct_change().diff()

        return feats

    def apply_filters(self, features_row: pd.Series, signal: float) -> bool:
        """Применение фильтров к сигналу"""
        # Фильтр по силе сигнала
        if abs(signal) < self.min_signal_strength:
            return False

        filters = [
            features_row.get('volume_ratio', 1) > self.min_volume_factor,
            features_row.get('realized_vol', 0) < self.max_volatility,
            abs(features_row.get('trend_strength', 0)) > self.min_trend_strength,
            abs(features_row.get('momentum_5', 0)) > self.momentum_threshold,
            features_row.get('price_efficiency', 0) > -0.8,
        ]

        return all(filters)

    def calculate_position_size(self, signal: float, features_row: pd.Series,
                                risk_score: float, portfolio_value: float) -> float:
        """Динамический расчет размера позиции"""
        base_size = 0.015  # Уменьшили базовый размер

        # Усиление сигнала силой тренда
        trend_strength = abs(features_row.get('trend_strength', 0))
        signal_boost = 1.0 + min(trend_strength * 0.5, 0.5)  # Более консервативно

        # Корректировка на волатильность
        volatility = features_row.get('realized_vol', 0.02)
        vol_adjustment = 0.02 / max(volatility, 0.01)

        # Корректировка на риск
        risk_adjustment = risk_score

        final_size = base_size * signal_boost * vol_adjustment * risk_adjustment * abs(signal)
        return min(final_size, 0.08)  # Максимум 8% на позицию


def run_improved_backtest() -> BacktestResult:
    """Запуск бэктеста с улучшенной стратегией"""
    print("[start] IMPROVED STRATEGY BACKTEST")

    # Загружаем данные
    data_dir = "data"
    symbols = config.symbols_cfg.symbols

    data = load_all(data_dir, symbols)
    for symbol in list(data.keys()):
        data[symbol] = slice_by_date(data[symbol], config.bt_cfg.start_date, config.bt_cfg.end_date)
        data[symbol] = ensure_dtindex_utc(data[symbol])
        print(f"Loaded {symbol}: {len(data[symbol])} rows")

    # Создаем улучшенные фичи
    print("[features] computing advanced features...")
    improved_strat = ImprovedStrategy()

    advanced_feats = {}
    for symbol, df in data.items():
        advanced_feats[symbol] = improved_strat.calculate_advanced_features(df)
        print(f"Advanced features for {symbol}: {advanced_feats[symbol].shape}")

    # AI модели
    try:
        ai_strategy = AIStrategyModel("models/ai_strategy_optimized.pkl")
        print("[ai] Using optimized strategy model")
    except:
        ai_strategy = AIStrategyModel(threshold=0.6)  # Повышенный порог
        print("[ai] Using standard strategy model with higher threshold")

    ai_risk = AIRiskModel()

    # Получаем общий индекс времени
    all_indices = [advanced_feats[s].index for s in symbols if s in advanced_feats and not advanced_feats[s].empty]
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
            if (symbol not in advanced_feats or symbol not in data or
                    timestamp not in advanced_feats[symbol].index or timestamp not in data[symbol].index):
                continue

            # Получаем фичи и цену
            features_row = advanced_feats[symbol].loc[timestamp]
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
                signal = 0.0

            # Получаем риск-скор
            risk_score = ai_risk.get_series(pd.DatetimeIndex([timestamp])).iloc[0]

            # Применяем фильтры улучшенной стратегии
            if improved_strat.apply_filters(features_row, signal):
                # Рассчитываем размер позиции через улучшенную стратегию
                position_size = improved_strat.calculate_position_size(
                    signal, features_row, risk_score, total_portfolio_value
                )
                target_value = signal * position_size * total_portfolio_value
                target_quantity = target_value / price if price > 0 else 0
            else:
                target_quantity = 0  # Фильтры не пройдены - не торгуем

            # Исполняем торговлю если нужно
            current_qty = positions.get(symbol, 0)
            quantity_delta = target_quantity - current_qty

            # Минимальный порог для торговли
            min_trade_value = 2000  # Увеличили минимальную сумму сделки
            if abs(quantity_delta * price) > min_trade_value:
                # Расчет комиссий и проскальзывания
                costs = TradeCosts(
                    commission_per_side=config.costs_cfg.commission_per_side,
                    tick_size=config.symbols_cfg.tick_size,
                    typical_spread_ticks=config.symbols_cfg.typical_spread_ticks
                )

                # Цена исполнения с учетом проскальзывания
                spread_ticks = config.symbols_cfg.typical_spread_ticks.get(symbol, 2)
                slip_ticks = spread_ticks + 1
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
                    "risk_score": risk_score,
                    "filters_passed": True
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
        metrics = calculate_performance_metrics(equity_series)

    print(
        f"[done] backtest finished. Trades: {len(trades_df)}, Final equity: {equity_df['equity'].iloc[-1] if not equity_df.empty else initial_equity:.2f}")

    # Сохраняем результаты
    equity_df.to_csv(os.path.join(OUT_DIR, "equity_improved.csv"))
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(OUT_DIR, "trades_improved.csv"), index=False)

    with open(os.path.join(OUT_DIR, "metrics_improved.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Дополнительная статистика
    if not trades_df.empty:
        passed_trades = trades_df[trades_df.get('filters_passed', False)]
        total_trades = len(trades_df)
        passed_count = len(passed_trades)
        print(f"Trades passed filters: {passed_count}/{total_trades} ({passed_count / total_trades * 100:.1f}%)")

        # Анализ сигналов
        if 'signal' in trades_df.columns:
            strong_signals = trades_df[abs(trades_df['signal']) > 0.5]
            print(f"Strong signals (|signal| > 0.5): {len(strong_signals)}")

    return BacktestResult(
        equity=equity_df,
        trades=trades_df,
        positions=positions,
        metrics=metrics
    )


if __name__ == "__main__":
    result = run_improved_backtest()
    print("\n=== IMPROVED STRATEGY RESULTS ===")
    print(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
    print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {result.metrics.get('win_rate', 0):.2%}")
    print(
        f"Final Equity: {result.equity['equity'].iloc[-1] if not result.equity.empty else config.bt_cfg.initial_equity:.2f}")
    print(f"Total Trades: {len(result.trades)}")