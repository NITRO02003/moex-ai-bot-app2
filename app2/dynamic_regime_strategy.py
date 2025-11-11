# dynamic_regime_strategy.py
from app2 import features as F, models as M
from app2.dynamic_regime_detector import DynamicRegimeDetector


def dynamic_regime_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.5):
    """Умная стратегия с динамическим выбором параметров по рыночному режиму"""

    if len(prices) < 100:
        return create_empty_signal(prices)

    # 1. ОПРЕДЕЛЯЕМ РЫНОЧНЫЙ РЕЖИМ
    regime_detector = DynamicRegimeDetector()
    current_regime = regime_detector.detect_regime(prices).iloc[-1]

    # 2. ПОЛУЧАЕМ ПАРАМЕТРЫ ДЛЯ РЕЖИМА
    regime_threshold, position_multiplier = regime_detector.get_strategy_params(current_regime)

    # 3. ГЕНЕРИРУЕМ СИГНАЛЫ С УЧЕТОМ РЕЖИМА
    X = F.build(prices)
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))

    # Пороги в зависимости от режима
    long_cond = p > regime_threshold
    short_cond = p < (1 - regime_threshold)

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # 4. ДИНАМИЧЕСКИЙ РАЗМЕР ПОЗИЦИИ
    from app2 import risk as R
    try:
        base_size = R.position_size(prices['close'].astype(float), p, equity, rp)
        dynamic_size = base_size * position_multiplier
    except:
        # Фолбэк на фиксированный размер
        dynamic_size = side * (equity * 0.02 * position_multiplier)

    result = pd.DataFrame({
        'p': p,
        'side': side,
        'size': dynamic_size,
        'regime': current_regime,
        'regime_threshold': regime_threshold
    }, index=prices.index)

    # ДИАГНОСТИКА
    active_signals = result[result['side'] != 0]
    print(f"🎯 ДИНАМИЧЕСКАЯ СТРАТЕГИЯ:")
    print(f"   Режим: {current_regime}")
    print(f"   Порог: {regime_threshold}, Множитель: {position_multiplier}")
    print(f"   Сигналов: {len(active_signals)}")
    print(f"   Средняя уверенность: {result[result['side'] != 0]['p'].mean():.3f}")

    return result