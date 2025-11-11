# regime_backtest.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from datetime import datetime
from app2 import data as D, models as M, risk as R, backtest as B, features as F
from app2.backtest import run_symbol, BtParams
from app2.gb_backtest_system import GBBacktestSystem


def create_empty_signal(prices):
    return pd.DataFrame({
        'p': 0.5, 'side': 0, 'size': 0.0
    }, index=prices.index)


def simple_regime_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.5):
    if len(prices) < 100:
        return create_empty_signal(prices)

    close = prices['close'].astype(float)
    volume = prices.get('volume', pd.Series(1, index=prices.index))

    # Индикаторы
    volatility = close.pct_change().rolling(20).std()
    trend_strength = abs(close.rolling(10).mean() - close.rolling(30).mean()) / close
    volume_z = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

    # Квантили
    vol_quantile = volatility.quantile(0.7)
    trend_quantile = trend_strength.quantile(0.7)

    # Режимы - ИСПРАВЛЕНО: правильное определение
    high_vol = volatility > vol_quantile
    strong_trend = trend_strength > trend_quantile
    high_volume = volume_z > 1.0

    # Векторизованное определение режима - ИСПРАВЛЕНО
    regime_conditions = pd.Series('RANGING', index=prices.index)
    regime_conditions[high_vol] = 'VOLATILE'
    regime_conditions[strong_trend & high_volume] = 'TRENDING'
    regime_conditions[strong_trend & ~high_volume] = 'TRENDING'

    # Параметры режимов
    regime_params = {
        'VOLATILE': {'threshold': 0.55, 'multiplier': 0.5},  # ПОНИЖЕН порог
        'TRENDING': {'threshold': 0.52, 'multiplier': 1.2},  # ПОНИЖЕН порог
        'RANGING': {'threshold': 0.53, 'multiplier': 0.8}  # ПОНИЖЕН порог
    }

    # Генерация сигналов
    X = F.build(prices)
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))

    # ДИАГНОСТИКА: выводим статистику предсказаний
    print(f"   📊 Статистика предсказаний:")
    print(f"      Min: {p.min():.3f}, Max: {p.max():.3f}, Mean: {p.mean():.3f}")
    print(f"      >0.5: {(p > 0.5).sum()}, >0.55: {(p > 0.55).sum()}")

    # Применяем пороги - ИСПРАВЛЕНО: правильная инициализация
    long_cond = pd.Series(False, index=p.index, dtype=bool)
    short_cond = pd.Series(False, index=p.index, dtype=bool)

    for regime, params in regime_params.items():
        regime_mask = (regime_conditions == regime)
        regime_threshold = params['threshold']

        # ИСПРАВЛЕНО: правильное присвоение с булевыми значениями
        long_cond_regime = (p > regime_threshold) & regime_mask
        short_cond_regime = (p < (1 - regime_threshold)) & regime_mask

        long_cond = long_cond | long_cond_regime
        short_cond = short_cond | short_cond_regime

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # Размер позиции
    size_multiplier = pd.Series(1.0, index=prices.index)
    for regime, params in regime_params.items():
        regime_mask = (regime_conditions == regime)
        size_multiplier[regime_mask] = params['multiplier']

    try:
        from app2.risk import position_size
        base_size = position_size(prices['close'].astype(float), p, equity, rp)
        dynamic_size = base_size * size_multiplier
    except Exception as e:
        print(f"⚠️  Ошибка расчета размера: {e}")
        dynamic_size = side * (equity * 0.02 * size_multiplier)

    result = pd.DataFrame({
        'p': p, 'side': side, 'size': dynamic_size,
        'regime': regime_conditions
    }, index=prices.index)

    # Статистика
    regime_stats = result['regime'].value_counts()
    signals_by_regime = result[result['side'] != 0]['regime'].value_counts()

    print(f"🎯 СТАТИСТИКА РЕЖИМОВ:")
    for regime in regime_stats.index:
        total_bars = regime_stats[regime]
        signals = signals_by_regime.get(regime, 0)
        print(f"   {regime}: {total_bars} баров, {signals} сигналов")

    return result


def debug_data_quality(symbol: str, prices: pd.DataFrame):
    """Проверка качества данных и предсказаний"""
    if prices.empty:
        print(f"❌ Нет данных для {symbol}")
        return

    print(f"\n📊 ДИАГНОСТИКА ДАННЫХ {symbol}:")
    print(f"   Период: {prices.index.min()} - {prices.index.max()}")
    print(f"   Баров: {len(prices)}")
    print(f"   Columns: {prices.columns.tolist()}")

    # Проверяем предсказания
    from app2 import models as M, features as F
    bundle = M.load()
    X = F.build(prices)
    p = M.predict_proba(bundle, X, prices['close'].astype(float))

    print(f"   Предсказания: min={p.min():.3f}, max={p.max():.3f}")
    print(f"   >0.5: {(p > 0.5).sum()}, >0.6: {(p > 0.6).sum()}, >0.7: {(p > 0.7).sum()}")


def debug_simple_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.5):
    """Упрощенная стратегия для отладки"""
    from app2 import features as F, models as M

    if len(prices) < 100:
        return create_empty_signal(prices)

    X = F.build(prices)
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))

    # ПРОСТЫЕ УСЛОВИЯ
    long_cond = p > 0.55
    short_cond = p < 0.45

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # ФИКСИРОВАННЫЙ РАЗМЕР
    size = side * (equity * 0.02)

    # ДИАГНОСТИКА
    print(f"🔍 ДИАГНОСТИКА СТРАТЕГИИ:")
    print(f"   Всего баров: {len(p)}")
    print(f"   Предсказания: min={p.min():.3f}, max={p.max():.3f}, mean={p.mean():.3f}")
    print(f"   Long сигналов (p>0.55): {(p > 0.55).sum()}")
    print(f"   Short сигналов (p<0.45): {(p < 0.45).sum()}")
    print(f"   Активных сигналов: {len(side[side != 0])}")

    return pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)


def run_regime_backtest():
    """Запуск бэктеста с режимной стратегией"""

    print("🚀 ЗАПУСК РЕЖИМНОГО БЭКТЕСТА")
    print("=" * 50)

    symbols = ['SBER', 'GAZP']  # Тестируем на 2 тикерах для начала

    # Инициализация системы
    system = GBBacktestSystem()

    # Загрузка данных
    prices_data = system.load_and_prepare_data(symbols, '2023-01-01', '2024-01-01')  # Укороченный период для теста

    for symbol in symbols:
        if symbol in prices_data:
            debug_data_quality(symbol, prices_data[symbol])


    if not prices_data:
        print("❌ Нет данных для тестирования")
        return

    # Обучение модели
    if not system.train_gb_model(prices_data):
        print("❌ Ошибка обучения модели")
        return

    # Настройки бэктеста
    rp = R.from_config()
    bt = BtParams(commission=0.0005, slippage_bps=1.0, horizon=2)

    results = {}

    for symbol in symbols:
        if symbol not in prices_data:
            continue

        print(f"\n🔍 Тестируем {symbol}...")

        try:
            # Создаем совместимый bundle
            gb_bundle = {
                'model': system.gb_model,
                'feature_names': system.feature_names,
                'predict_proba': lambda X, close: system.gb_predict(X, close)
            }

            # Временная подмена стратегии
            from app2.strategy import emergency_debug_strategy
            import app2.strategy as S
            original_strategy = S.signal_and_size
            S.signal_and_size = emergency_debug_strategy

            result = run_symbol(prices_data[symbol], gb_bundle, rp, bt, 1000000.0, threshold=threshold)
            results[symbol] = result

            # Восстанавливаем стратегию
            S.signal_and_size = original_strategy

            # Вывод результатов
            metrics = result['metrics']
            print(f"   📊 РЕЗУЛЬТАТЫ:")
            print(f"      Сделок: {metrics['total_trades']}")
            print(f"      Доходность: {metrics['total_return']:.2%}")
            print(f"      Win Rate: {metrics['win_rate']:.1%}")
            print(f"      Комиссии: {metrics['total_commissions']:,.0f} руб")
            print(f"      Чистая прибыль: {metrics['net_pnl']:,.0f} руб")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


    # Сохранение результатов
    if results:
        summary = {}
        for symbol, result in results.items():
            summary[symbol] = result['metrics']
        from .paths import REPORTS_DIR

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"regime_backtest_results_{timestamp}.json"
        path = REPORTS_DIR / filename

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Результаты сохранены в {path}")


    protocol = TransferProtocol()
    protocol.update_auto_report(
        latest_results=summary,  # результаты бэктеста
        current_experiments="Testing simplified regime strategy with VOLA+TREND core",
        problems="Zero trades in backtest - debugging signal generation",
        decisions="Temporarily simplifying model to get baseline working


    return results


if __name__ == "__main__":
    run_regime_backtest()