# strategy.py - ОБНОВЛЕННАЯ ВЕРСИЯ
from __future__ import annotations
import pandas as pd
import numpy as np
from . import features as F, models as M


def filtered_signal_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.65):
    """Стратегия с агрессивной фильтрацией шума"""

    X = F.build(prices)
    if X.empty:
        return create_empty_signal(prices)

    X = X[F.final_columns(X.columns)]
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))

    # СТРОГИЕ ФИЛЬТРЫ:
    close = prices['close'].astype(float)
    volume = prices.get('volume', pd.Series(1, index=p.index))

    # 1. ФИЛЬТР ОБЪЕМА - только при повышенном объеме
    volume_ma = volume.rolling(20).mean()
    volume_ok = volume > volume_ma * 1.5

    # 2. ФИЛЬТР ВОЛАТИЛЬНОСТИ - избегать экстремальной волатильности
    volatility = close.pct_change().rolling(20).std()
    volatility_ok = (volatility > volatility.quantile(0.2)) & (volatility < volatility.quantile(0.8))

    # 3. ФИЛЬТР ВРЕМЕНИ - минимум 5 баров между сделками
    time_filter = pd.Series(True, index=p.index)
    if hasattr(filtered_signal_strategy, 'last_trade_time') and filtered_signal_strategy.last_trade_time is not None:
        time_since_last = (p.index - filtered_signal_strategy.last_trade_time).total_seconds() / 600
        time_filter = time_since_last > 50  # 5 баров по 10 минут

    # 4. ФИЛЬТР РАЗМЕРА - минимальный размер сделки
    min_trade_size = equity * 0.005  # 0.5% от капитала

    # КОМБИНИРОВАННЫЕ УСЛОВИЯ
    long_cond = (p > 0.7) & volume_ok & volatility_ok & time_filter
    short_cond = (p < 0.3) & volume_ok & volatility_ok & time_filter

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # РАЗМЕР С ФИЛЬТРОМ МИНИМАЛЬНОЙ СДЕЛКИ
    base_size = side * (equity * 0.02)  # 2% базово
    size = base_size.where(base_size.abs() >= min_trade_size, 0)

    # ОБНОВЛЯЕМ ВРЕМЯ ПОСЛЕДНЕЙ СДЕЛКИ
    if side.abs().sum() > 0:
        filtered_signal_strategy.last_trade_time = side[side != 0].index[-1]

    result = pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)

    print(f"⚡ ФИЛЬТРОВАННАЯ СТРАТЕГИЯ:")
    print(f"   Сигналов до фильтров: {(p > 0.7).sum() + (p < 0.3).sum()}")
    print(f"   Сигналов после фильтров: {len(result[result['side'] != 0])}")
    if ((p > 0.7).sum() + (p < 0.3).sum()) > 0:
        filtered_percent = ((p > 0.7).sum() + (p < 0.3).sum() - len(result[result['side'] != 0])) / (
                    (p > 0.7).sum() + (p < 0.3).sum()) * 100
        print(f"   Процент отфильтровано: {filtered_percent:.1f}%")

    return result


# Инициализация статической переменной
filtered_signal_strategy.last_trade_time = None


def signal_and_size(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.5):
    from . import risk as R

    if len(prices) < 100:
        return create_empty_signal(prices)

    X = F.build(prices)
    if X.empty:
        return create_empty_signal(prices)

    X = X[F.final_columns(X.columns)]
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))

    # УЛУЧШЕННЫЕ УСЛОВИЯ ДЛЯ СИГНАЛОВ
    min_confidence = 0.65

    # СИГНАЛЫ С УЧЕТОМ РЕЖИМА РЫНКА
    volatility = prices['close'].pct_change().rolling(20).std()
    high_vol = volatility > volatility.median()

    # АДАПТИВНЫЕ ПОРОГИ
    long_threshold = np.where(high_vol, max(threshold, 0.75), max(threshold, 0.70))  # было 0.7/0.65
    short_threshold = np.where(high_vol, min(1 - threshold, 0.25), min(1 - threshold, 0.30))  # было 0.3/0.35

    long_cond = (p >= long_threshold) & (p >= min_confidence)
    short_cond = (p <= short_threshold) & ((1 - p) >= min_confidence)

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # ФИЛЬТР ОБЪЕМА - только при повышенном объеме
    volume_ma = prices.get('volume', pd.Series(1, index=p.index)).rolling(20).mean()
    volume_ok = prices.get('volume', pd.Series(1, index=p.index)) > volume_ma * 1.5

    # ФИЛЬТР ВОЛАТИЛЬНОСТИ - избегать экстремальных значений
    volatility = prices['close'].pct_change().rolling(20).std()
    volatility_ok = volatility.between(volatility.quantile(0.3), volatility.quantile(0.7))

    # ФИЛЬТР ВРЕМЕНИ - минимум 3 бара между сделками
    time_filter = pd.Series(True, index=p.index)
    if hasattr(signal_and_size, 'last_trade_bar'):
        bars_since_last = p.index.get_indexer(p.index) - signal_and_size.last_trade_bar
        time_filter = bars_since_last > 3

    # ОБНОВЛЕННЫЕ УСЛОВИЯ С ФИЛЬТРАМИ
    long_cond = (p >= long_threshold) & (p >= min_confidence) & volume_ok & volatility_ok & time_filter
    short_cond = (p <= short_threshold) & ((1 - p) >= min_confidence) & volume_ok & volatility_ok & time_filter

    # ОБНОВЛЯЕМ ВРЕМЯ ПОСЛЕДНЕЙ СДЕЛКИ
    if side.abs().sum() > 0:
        signal_and_size.last_trade_bar = p.index.get_indexer(side[side != 0].index[-1])[0]

    # УЛУЧШЕННЫЙ РАСЧЕТ РАЗМЕРА
    try:
        # Используем улучшенный риск-менеджмент
        size = R.ai_optimized_position_size(
            prices['close'].astype(float),
            p,
            equity,
            rp,
            R.advanced_market_regime(prices['close'].astype(float))
        )
    except:
        # Фолбэк на консервативный размер
        size = side * (equity * 0.02)

    result = pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)

    # ДЕТАЛЬНАЯ СТАТИСТИКА
    active_signals = result[result['side'] != 0]
    print(f"\n=== СТАТИСТИКА СИГНАЛОВ ===")
    print(f"Всего баров: {len(result)}")
    print(f"Активных сигналов: {len(active_signals)}")
    print(f"Long: {(result['side'] == 1).sum()}, Short: {(result['side'] == -1).sum()}")
    print(f"Процент активных: {len(active_signals) / len(result) * 100:.1f}%")
    print(f"Средняя уверенность: {result[result['side'] != 0]['p'].mean():.3f}")

    return result


def create_empty_signal(prices: pd.DataFrame):
    return pd.DataFrame({
        'p': 0.5,
        'side': 0,
        'size': 0.0
    }, index=prices.index)


def ultra_conservative_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.7):
    """СВЕРХКОНСЕРВАТИВНАЯ стратегия - только лучшие сигналы"""

    X = F.build(prices)
    if X.empty or len(prices) < 100:
        return create_empty_signal(prices)

    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))
    close = prices['close'].astype(float)
    volume = prices.get('volume', pd.Series(1, index=p.index))

    # СУПЕР-СТРОГИЕ ФИЛЬТРЫ:

    # 1. ТОЛЬКО СИЛЬНЫЕ СИГНАЛЫ
    strong_long = p > 0.75  # было 0.65
    strong_short = p < 0.25  # было 0.35

    # 2. ФИЛЬТР ТРЕНДА - только по тренду
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    trend_up = sma_20 > sma_50
    trend_down = sma_20 < sma_50

    # 3. ФИЛЬТР ОБЪЕМА - объем в 2 раза выше среднего
    volume_ok = volume > volume.rolling(20).mean() * 2

    # 4. ФИЛЬТР ВОЛАТИЛЬНОСТИ - избегать экстремальных значений
    volatility = close.pct_change().rolling(20).std()
    volatility_ok = volatility.between(volatility.quantile(0.3), volatility.quantile(0.7))

    # КОМБИНИРОВАННЫЕ УСЛОВИЯ
    long_cond = strong_long & trend_up & volume_ok & volatility_ok
    short_cond = strong_short & trend_down & volume_ok & volatility_ok

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # МАЛЕНЬКИЙ ФИКСИРОВАННЫЙ РАЗМЕР
    size = side * (equity * 0.01)  # 1% капитала

    result = pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)

    active_signals = result[result['side'] != 0]
    print(f"🎯 СВЕРХКОНСЕРВАТИВНАЯ СТРАТЕГИЯ:")
    print(f"   Всего сигналов: {len(active_signals)}")
    print(f"   Long: {(result['side'] == 1).sum()}, Short: {(result['side'] == -1).sum()}")
    print(f"   Средняя уверенность: {result[result['side'] != 0]['p'].mean():.3f}")

    return result


def conservative_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float):
    """Сверхконсервативная стратегия с улучшенными фильтрами"""
    from . import risk as R

    X = F.build(prices)
    if X.empty:
        return create_empty_signal(prices)

    X = X[F.final_columns(X.columns)]
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))

    # СТРОГИЕ УСЛОВИЯ С ФИЛЬТРАМИ
    volatility_filter = prices['close'].pct_change().rolling(20).std() < 0.02
    volume_filter = prices.get('volume', pd.Series(1, index=p.index)) > 1000

    strong_long = (p > 0.65) & volatility_filter & volume_filter
    strong_short = (p < 0.35) & volatility_filter & volume_filter

    side = pd.Series(0, index=p.index)
    side[strong_long] = 1
    side[strong_short] = -1

    # КОНСЕРВАТИВНЫЙ РАЗМЕР
    size = side * (equity * 0.01)

    result = pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)
    print(f"Консервативная стратегия: {len(result[result['side'] != 0])} сигналов")

    return result


def emergency_debug_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.51):
    """ЭКСТРЕННАЯ стратегия для получения первых сделок"""
    from . import features as F, models as M

    if len(prices) < 100:
        return create_empty_signal(prices)

    X = F.build(prices)
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))

    # СУПЕР-НИЗКИЕ ПОРОГИ для тестирования
    long_cond = p > 0.51
    short_cond = p < 0.49

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # ФИКСИРОВАННЫЙ РАЗМЕР
    size = side * (equity * 0.01)

    # ДЕТАЛЬНАЯ ДИАГНОСТИКА
    print(f"🚨 EMERGENCY STRATEGY DIAGNOSTICS:")
    print(f"   Всего баров: {len(p)}")
    print(f"   Предсказания: min={p.min():.3f}, max={p.max():.3f}, mean={p.mean():.3f}")
    print(f"   Статистика порогов:")
    for th in [0.45, 0.49, 0.51, 0.55]:
        long_count = (p > th).sum()
        short_count = (p < (1 - th)).sum()
        print(f"   >{th}: {long_count}, <{1 - th}: {short_count}")
    print(f"   Активных сигналов: {len(side[side != 0])}")

    return pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)