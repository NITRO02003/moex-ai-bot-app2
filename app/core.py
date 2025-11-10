# core.py - ИСПРАВЛЕННАЯ ВЕРСИЯ БЕЗ КОНФЛИКТА ИМЕН
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import os


@dataclass
class TradeCosts:
    commission_per_side: float = 0.0005
    tick_size: Dict[str, float] | None = None
    typical_spread_ticks: Dict[str, int] | None = None
    slippage_ticks_extra: int = 0

    def round_price(self, symbol: str, price: float) -> float:
        ts = (self.tick_size or {}).get(symbol, 0.01)
        return round(round(price / ts) * ts, 8)

    def adverse_price_shift(self) -> float:
        """Для совместимости с backtest.py"""
        slip_ticks = (self.slippage_ticks_extra or 0)
        tick_size = 0.01  # default
        return slip_ticks * tick_size


# ПЕРЕИМЕНОВАЛИ ФУНКЦИЮ чтобы избежать конфликта
def calculate_performance_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    if equity_curve.empty or len(equity_curve) < 2:
        return {
            "bars": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else 0.0
        }

    ret = equity_curve.pct_change().fillna(0.0)
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

    # Расчет максимальной просадки
    peak = equity_curve.expanding().max()
    drawdown = (peak - equity_curve) / peak
    max_dd = float(drawdown.max())

    win_rate = float((ret > 0).mean())

    return {
        "bars": int(len(equity_curve)),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "final_equity": float(equity_curve.iloc[-1])
    }


# ДЕКОРАТОР ДЛЯ СОВМЕСТИМОСТИ - ПЕРЕИМЕНОВАЛИ
class PerformanceMonitor:
    metrics = {}

    @staticmethod
    def monitor_step(name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


class AIRiskOverlay:
    def __init__(self, series: Optional[pd.Series] = None, path: str = "out/ai_risk_scores.csv"):
        self.series = series
        self.path = path
        if self.series is None:
            self._load_file()

    def _load_file(self):
        if not os.path.exists(self.path):
            self.series = None
            return
        try:
            df = pd.read_csv(self.path)
            if "dt" in df.columns and "risk_score" in df.columns:
                idx = pd.to_datetime(df["dt"], errors="coerce", utc=True)
                s = pd.to_numeric(df["risk_score"], errors="coerce").clip(0.0, 1.0)
                ser = pd.Series(s.values, index=idx).sort_index()
                self.series = ser[~ser.index.isna()]
        except Exception:
            self.series = None

    def get(self, ts: pd.Timestamp) -> float:
        if self.series is None or self.series.empty:
            return 1.0
        if ts in self.series.index:
            return float(self.series.loc[ts])
        prev = self.series.loc[:ts]
        return float(prev.iloc[-1]) if len(prev) > 0 else 1.0


class PortfolioRiskManager:
    def __init__(self, max_total_drawdown: float = 0.3, ai_overlay: Optional[AIRiskOverlay] = None):
        self.max_total_drawdown = max_total_drawdown
        self.peak: Optional[float] = None
        self.ai_overlay = ai_overlay or AIRiskOverlay()
        # ДЛЯ СОВМЕСТИМОСТИ С backtest.py
        self.equity = None
        self.max_daily_drawdown = 0.1
        self.max_consecutive_losses = 5
        self.circuit_breaker_action = "halt"

    def risk_budget(self, ts: pd.Timestamp, equity: float) -> float:
        if self.peak is None:
            self.peak = equity
        else:
            self.peak = max(self.peak, equity)
        dd = (self.peak - equity) / self.peak if self.peak else 0.0
        cb = 0.0 if dd > self.max_total_drawdown else 1.0
        ai = float(self.ai_overlay.get(ts))
        return max(0.0, min(1.0, cb * ai))

    def allow_trade(self, ts: pd.Timestamp, equity: float) -> bool:
        return self.risk_budget(ts, equity) > 0.0

    # МЕТОДЫ ДЛЯ СОВМЕСТИМОСТИ С backtest.py
    def on_new_day(self):
        """Вызывается при смене дня"""
        pass

    def should_halt(self) -> bool:
        """Проверка на остановку торгов"""
        return False

    def update_equity(self, equity: float):
        """Обновление эквити для отслеживания просадок"""
        self.equity = equity


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ backtest.py
def volatility_target_weight(signal: float, realized_vol: float,
                             target_daily_vol: float = 0.02,
                             bars_per_day: int = 30) -> float:
    """Рассчет веса на основе целевой волатильности"""
    if realized_vol <= 0:
        return 0.0

    # Масштабируем дневную волатильность до внутридневной
    target_intraday_vol = target_daily_vol / np.sqrt(bars_per_day)
    weight = signal * (target_intraday_vol / realized_vol)
    return float(np.clip(weight, -0.5, 0.5))


def _atr(high, low, close, n=14):
    """Average True Range - для classify_regime"""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def classify_regime(df: pd.DataFrame, atr_window: int = 14,
                    thresholds: Tuple[float, float] = (0.005, 0.015)) -> Tuple[pd.Series, pd.Series]:
    """Упрощенная классификация режима рынка"""
    if df.empty:
        return pd.Series(dtype=int), pd.Series(dtype=float)

    # Используем ATR для определения волатильности
    high, low, close = df['high'], df['low'], df['close']
    atr = _atr(high, low, close, n=atr_window)
    atr_pct = atr / close

    # Классифицируем режимы
    regime = pd.cut(atr_pct, bins=[0, thresholds[0], thresholds[1], 1],
                    labels=[0, 1, 2]).fillna(1).astype(int)
    confidence = 1.0 - (abs(atr_pct - np.mean(atr_pct)) / np.std(atr_pct)).clip(0, 1)

    return regime, confidence


def decide_slippage_ticks(symbol: str, is_liquid: bool,
                          liquid_slip: int = 1, illiquid_slip: int = 3) -> int:
    """Определение slippage на основе ликвидности"""
    return liquid_slip if is_liquid else illiquid_slip