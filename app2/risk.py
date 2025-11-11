from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import config


# ================== params ==================
@dataclass
class RiskParams:
    max_dd: float = 0.10
    per_trade_risk: float = 0.001
    daily_risk_cap: float = 0.01
    vol_lb: int = 48
    conf_floor_stable: float = 0.55
    conf_floor_volatile: float = 0.60
    max_gross: float = 2.0


def from_config() -> RiskParams:
    rp = RiskParams()
    rc = getattr(config, "risk_cfg", None)
    if rc is None:
        return rp
    for k in ("max_dd", "per_trade_risk", "daily_risk_cap", "vol_lb",
              "conf_floor_stable", "conf_floor_volatile", "max_gross"):
        if hasattr(rc, k):
            setattr(rp, k, getattr(rc, k))
    return rp


# ================== helpers ==================
def realized_vol(close: pd.Series, lb: int) -> pd.Series:
    r = close.astype(float).pct_change().fillna(0.0)
    vol = r.rolling(lb, min_periods=max(2, lb // 4)).std()
    return vol.replace(0, np.nan).fillna(vol.mean())


def regime(close: pd.Series, lb: int) -> pd.Series:
    rv = realized_vol(close, lb)
    med = rv.rolling(lb, min_periods=max(2, lb // 4)).median()
    return (rv > med).rename("volatile")


def regime_by_vol(close: pd.Series, lb: int = 96) -> pd.Series:
    """Для обратной совместимости с models.py"""
    return regime(close, lb)


def conf_gate(p: pd.Series, close: pd.Series, rp: RiskParams) -> pd.Series:
    reg = regime(close, rp.vol_lb).reindex(p.index).fillna(False)
    floor = np.where(reg, rp.conf_floor_volatile, rp.conf_floor_stable)
    return p.where(p >= pd.Series(floor, index=p.index), 0.0)


# ================== sizing & limits ==================
def position_size(close: pd.Series, p_up: pd.Series, equity: float, rp: RiskParams) -> pd.Series:
    close = close.astype(float)
    p_eff = conf_gate(p_up, close, rp)
    rv = realized_vol(close, rp.vol_lb).reindex(close.index).ffill().fillna(
        0.0)  # Исправлено: ffill() вместо fillna(method="ffill")
    risk_dollars = max(0.0, float(equity) * float(rp.per_trade_risk))
    denom = (rv * close).replace(0, np.nan)
    base_size = (risk_dollars / denom).clip(lower=0).fillna(0.0)
    base_notional = (base_size * close).clip(lower=0).fillna(0.0)
    conf_scale = ((p_eff - 0.5).clip(lower=0) / 0.5).fillna(0.0)
    notional = (base_notional * conf_scale).clip(upper=equity * rp.max_gross)
    return notional.rename("size")


def apply_daily_risk_cap(pnl: pd.Series, equity0: float, rp: RiskParams) -> pd.Series:
    if pnl.empty:
        return pnl
    pnl = pnl.fillna(0.0)
    idx = pnl.index
    try:
        day_key = idx.date
    except AttributeError:
        day_key = pd.to_datetime(idx).date

    day_key = pd.Index(day_key)
    cum_day = pnl.groupby(day_key).cumsum()
    cap = -float(equity0) * float(rp.daily_risk_cap)
    cum_day_capped = cum_day.where(cum_day >= cap, cap)
    pnl_capped = cum_day_capped.groupby(day_key).diff().fillna(cum_day_capped)
    pnl_capped.index = idx
    return pnl_capped.rename("pnl")


def dd_halt(equity_curve: pd.Series, rp: RiskParams) -> pd.Series:
    peak = equity_curve.cummax()
    dd = (equity_curve / peak - 1.0).fillna(0.0)
    return (dd <= -float(rp.max_dd)).rename("dd_halt")


def adaptive_position_size(close: pd.Series, p_up: pd.Series, equity: float,
                           rp: RiskParams, volatility: pd.Series) -> pd.Series:
    """Адаптивный размер позиции с учетом волатильности"""
    close = close.astype(float)
    p_eff = conf_gate(p_up, close, rp)
    # БАЗОВЫЙ РАСЧЕТ
    rv = realized_vol(close, rp.vol_lb).reindex(close.index).ffill().fillna(0.0)
    risk_dollars = max(0.0, float(equity) * float(rp.per_trade_risk))
    denom = (rv * close).replace(0, np.nan)
    base_size = (risk_dollars / denom).clip(lower=0).fillna(0.0)
    base_notional = (base_size * close).clip(lower=0).fillna(0.0)

    # АДАПТИВНОЕ МАСШТАБИРОВАНИЕ
    vol_factor = np.where(
        volatility > volatility.quantile(0.7), 0.5,  # Уменьшаем при высокой волатильности
        np.where(volatility < volatility.quantile(0.3), 1.2, 1.0)  # Увеличиваем при низкой
    )

    conf_scale = ((p_eff - 0.5).clip(lower=0) / 0.5).fillna(0.0)
    adaptive_scale = conf_scale * vol_factor

    notional = (base_notional * adaptive_scale).clip(upper=equity * rp.max_gross)
    return notional.rename("size")


def advanced_market_regime(close: pd.Series, volume: pd.Series = None, lb: int = 96) -> pd.Series:
    """AI-определение режима рынка"""
    # ВОЛАТИЛЬНОСТЬ
    rv = realized_vol(close, lb)
    vol_regime = (rv > rv.rolling(lb).quantile(0.7))

    # ТРЕНД
    returns = close.pct_change()
    trend_strength = returns.rolling(lb).mean() / (returns.rolling(lb).std() + 1e-9)
    trending = (trend_strength.abs() > 0.5)

    # ОБЪЕМ (если доступен)
    if volume is not None:
        volume_z = (volume - volume.rolling(lb).mean()) / (volume.rolling(lb).std() + 1e-9)
        high_volume = (volume_z > 1.0)
    else:
        high_volume = pd.Series(False, index=close.index)

    # КОМБИНИРОВАННЫЙ РЕЖИМ
    regime = pd.Series('normal', index=close.index)
    regime = np.where(vol_regime & trending, 'high_vol_trend', regime)
    regime = np.where(vol_regime & ~trending, 'high_vol_range', regime)
    regime = np.where(~vol_regime & trending, 'trending', regime)
    regime = np.where(~vol_regime & ~trending, 'ranging', regime)
    regime = np.where(high_volume, regime + '_volume', regime)

    return pd.Series(regime, index=close.index)


def ai_optimized_position_size(close: pd.Series, p_up: pd.Series, equity: float,
                               rp: RiskParams, market_regime: pd.Series) -> pd.Series:
    """AI-оптимизированный размер позиции"""
    # БАЗОВЫЙ РАСЧЕТ
    base_size = position_size(close, p_up, equity, rp)

    # РЕЖИМ-ЗАВИСИМЫЕ МОДИФИКАТОРЫ
    regime_multipliers = {
        'high_vol_trend': 0.6,
        'high_vol_range': 0.3,
        'trending': 1.0,
        'ranging': 0.4,
        'high_vol_trend_volume': 0.8,
        'high_vol_range_volume': 0.5,
        'trending_volume': 1.2,
        'ranging_volume': 0.6,
        'normal': 0.8
    }

    # ПРИМЕНЯЕМ МОДИФИКАТОРЫ
    multiplier = market_regime.map(lambda x: regime_multipliers.get(x, 0.8))
    optimized_size = base_size * multiplier
    return optimized_size.clip(upper=equity * rp.max_gross)


def momentum_filter(close: pd.Series, window: int = 10) -> pd.Series:
    """Фильтр по моментуму для избежания входа против тренда"""
    returns = close.pct_change(window)
    sma_short = close.rolling(5).mean()
    sma_long = close.rolling(20).mean()
    trend_aligned = (sma_short > sma_long) & (returns > 0) | (sma_short < sma_long) & (returns < 0)
    return trend_aligned


def conservative_position_size(close: pd.Series, p_up: pd.Series, equity: float,
                               rp: RiskParams, volatility: pd.Series) -> pd.Series:
    """СУПЕР-КОНСЕРВАТИВНЫЙ расчет размера позиции"""
    close = close.astype(float)
    p_eff = conf_gate(p_up, close, rp)

    # БАЗОВЫЙ РАСЧЕТ С МИНИМАЛЬНЫМ РИСКОМ
    risk_dollars = max(0.0, float(equity) * 0.002)  # Всего 0.2% риска!
    rv = realized_vol(close, rp.vol_lb).reindex(close.index).ffill().fillna(0.02)

    # ОЧЕНЬ КОНСЕРВАТИВНЫЙ ЗНАМЕНАТЕЛЬ
    denom = (rv * close * 2).replace(0, np.nan)  # Удваиваем волатильность для безопасности
    base_size = (risk_dollars / denom).clip(lower=0).fillna(0.0)
    base_notional = (base_size * close).clip(lower=0).fillna(0.0)

    # ЖЕСТКОЕ МАСШТАБИРОВАНИЕ
    vol_factor = np.where(
        volatility > 0.03, 0.1,  # Крайне мало при высокой волатильности
        np.where(volatility < 0.01, 0.3, 0.2)  # Все равно мало
    )

    conf_scale = ((p_eff - 0.6).clip(lower=0) / 0.4).fillna(0.0)  # Только при p > 0.6
    adaptive_scale = conf_scale * vol_factor

    notional = (base_notional * adaptive_scale).clip(upper=equity * 0.5)  # Макс 50% капитала

    return notional.rename("size")


def strict_conf_gate(p: pd.Series, close: pd.Series, rp: RiskParams) -> pd.Series:
    """Строгая фильтрация по уверенности"""
    reg = regime(close, rp.vol_lb).reindex(p.index).fillna(False)

    # ПОВЫШАЕМ ПОРОГИ
    floor_stable = 0.65  # было 0.55
    floor_volatile = 0.70  # было 0.60

    floor = np.where(reg, floor_volatile, floor_stable)
    return p.where(p >= pd.Series(floor, index=p.index), 0.0)


def simple_conf_gate(p: pd.Series, close: pd.Series, rp: RiskParams) -> pd.Series:
    """Упрощенная фильтрация для тестирования"""
    # Базовый порог без сложной логики режимов
    base_threshold = 0.55
    return p.where(p >= base_threshold, 0.0)


def fixed_position_size(close: pd.Series, p_up: pd.Series, equity: float, rp: RiskParams) -> pd.Series:
    """Фиксированный размер позиции для тестирования"""
    # Просто 5% капитала при наличии сигнала
    signal_strength = (p_up - 0.5).clip(lower=0) / 0.5
    size = equity * 0.05 * signal_strength  # 5% капитала, масштабируемое по уверенности
    return pd.Series(size, index=close.index, name="size")


def dynamic_position_sizing(close: pd.Series, p_up: pd.Series, equity: float,
                            rp: RiskParams, volatility: pd.Series) -> pd.Series:
    """Динамический размер позиции на основе волатильности и уверенности"""

    # БАЗОВЫЙ РАСЧЕТ
    base_size = position_size(close, p_up, equity, rp)

    # ДИНАМИЧЕСКОЕ МАСШТАБИРОВАНИЕ
    vol_factor = np.where(
        volatility > volatility.quantile(0.8), 0.3,  # Меньше при высокой волатильности
        np.where(volatility < volatility.quantile(0.2), 1.5, 1.0)  # Больше при низкой
    )

    # МАСШТАБИРОВАНИЕ ПО УВЕРЕННОСТИ
    confidence_factor = ((p_up - 0.5).clip(lower=0) / 0.5) ** 0.5  # Квадратный корень для сглаживания

    dynamic_size = base_size * vol_factor * confidence_factor
    return dynamic_size.clip(upper=equity * rp.max_gross)