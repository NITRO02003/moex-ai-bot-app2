# dynamic_regime_detector.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class MarketRegime:
    name: str
    threshold: float
    position_multiplier: float
    description: str


class DynamicRegimeDetector:
    """
    Упрощенный детектор рыночных режимов на основе 3 ключевых факторов
    """

    def __init__(self, lookback_period: int = 100):
        self.lookback = lookback_period
        self.regimes = {
            'TRENDING': MarketRegime('TRENDING', 0.52, 1.2, "Тренд - умеренно"),
            'VOLATILE': MarketRegime('VOLATILE', 0.55, 0.5, "Волатильность - консервативно"),
            'RANGING': MarketRegime('RANGING', 0.53, 0.8, "Флэт - стандартно"),
            'BREAKOUT': MarketRegime('BREAKOUT', 0.51, 1.0, "Пробой - нейтрально")
        }

    def calculate_regime_indicators(self, prices: pd.DataFrame) -> Dict[str, pd.Series]:
        """Рассчитываем ключевые индикаторы для определения режима"""
        close = prices['close'].astype(float)
        high = prices['high'].astype(float)
        low = prices['low'].astype(float)
        volume = prices.get('volume', pd.Series(1, index=close.index))

        indicators = {}

        # 1. СИЛА ТРЕНДА (главный показатель)
        ma_fast = close.rolling(10).mean()
        ma_slow = close.rolling(30).mean()
        indicators['trend_strength'] = abs(ma_fast - ma_slow) / close

        # 2. ВОЛАТИЛЬНОСТЬ (второй по важности)
        atr = (high - low).rolling(14).mean()
        indicators['volatility'] = atr / close

        # 3. ОБЪЕМ (подтверждение)
        volume_ma = volume.rolling(20).mean()
        indicators['volume_z'] = (volume - volume_ma) / volume.rolling(20).std()

        # 4. ДИАПАЗОН (дополнительный фильтр)
        daily_range = (high - low) / close
        indicators['range_pct'] = daily_range

        return indicators

    def detect_regime(self, prices: pd.DataFrame) -> pd.Series:
        """Определяем текущий рыночный режим"""
        indicators = self.calculate_regime_indicators(prices)

        # ПРОСТЫЕ И ЭФФЕКТИВНЫЕ ПРАВИЛА
        trend_strong = indicators['trend_strength'] > indicators['trend_strength'].quantile(0.7)
        volatility_high = indicators['volatility'] > indicators['volatility'].quantile(0.8)
        volume_high = indicators['volume_z'] > 1.0
        range_wide = indicators['range_pct'] > indicators['range_pct'].quantile(0.7)

        # ЛОГИКА ОПРЕДЕЛЕНИЯ РЕЖИМА
        regime = pd.Series('RANGING', index=prices.index)  # по умолчанию

        # Трендовый режим (самый важный)
        regime = np.where(trend_strong & ~volatility_high, 'TRENDING', regime)

        # Волатильный режим (ограничиваем риск)
        regime = np.where(volatility_high, 'VOLATILE', regime)

        # Пробой (тренд + объем)
        regime = np.where(trend_strong & volume_high & range_wide, 'BREAKOUT', regime)

        # ГИСТЕРЕЗИС - избегаем частых переключений
        regime_series = pd.Series(regime, index=prices.index)
        regime_smooth = regime_series.rolling(5, min_periods=1).apply(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'RANGING'
        )

        return regime_smooth

    def get_strategy_params(self, regime: str) -> Tuple[float, float]:
        """Возвращает параметры стратегии для режима"""
        regime_obj = self.regimes.get(regime, self.regimes['RANGING'])
        return regime_obj.threshold, regime_obj.position_multiplier