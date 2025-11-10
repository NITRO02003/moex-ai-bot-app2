# app/technical_fallback.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
import ta  # Technical Analysis library


@dataclass
class TechnicalStrategy:
    """Чисто техническая стратегия без AI"""

    def calculate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Расчет сигналов на основе технических индикаторов"""

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        signals = pd.Series(0.0, index=df.index)

        # 1. RSI + Momentum
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        rsi_signal = np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0))

        # 2. MACD
        macd = ta.trend.MACD(close)
        macd_signal = np.where(macd.macd_diff() > 0, 1.0, -1.0)

        # 3. Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_position = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        bb_signal = np.where(bb_position < 0.2, 1.0, np.where(bb_position > 0.8, -1.0, 0.0))

        # 4. Volume-weighted price action
        volume_sma = volume.rolling(20).mean()
        volume_spike = volume / volume_sma
        price_change = close.pct_change(5)
        volume_signal = np.where((volume_spike > 1.5) & (price_change > 0.01), 1.0,
                                 np.where((volume_spike > 1.5) & (price_change < -0.01), -1.0, 0.0))

        # Комбинируем сигналы
        combined = (rsi_signal * 0.3 + macd_signal * 0.3 + bb_signal * 0.2 + volume_signal * 0.2)

        # Фильтруем слабые сигналы
        signals = np.where(np.abs(combined) > 0.3, np.sign(combined), 0.0)

        return pd.Series(signals, index=df.index)


def backtest_technical():
    """Бэктест чисто технической стратегии"""
    print("=== PURE TECHNICAL STRATEGY BACKTEST ===")

    from app.backtest import run_simple_backtest
    from app.config import config
    from app.utils import load_all, slice_by_date, ensure_dtindex_utc

    # Загружаем данные
    data_dir = "data"
    symbols = config.symbols_cfg.symbols

    data = load_all(data_dir, symbols)
    for symbol in list(data.keys()):
        data[symbol] = slice_by_date(data[symbol], config.bt_cfg.start_date, config.bt_cfg.end_date)
        data[symbol] = ensure_dtindex_utc(data[symbol])

    # Создаем технические сигналы
    strategy = TechnicalStrategy()
    technical_signals = {}

    for symbol, df in data.items():
        technical_signals[symbol] = strategy.calculate_signals(df)
        signal_stats = technical_signals[symbol].value_counts()
        print(
            f"{symbol}: Buy={signal_stats.get(1.0, 0)}, Sell={signal_stats.get(-1.0, 0)}, Hold={signal_stats.get(0.0, 0)}")

    # Модифицируем бэктест для использования технических сигналов
    # (нужно адаптировать run_simple_backtest)

    print("Technical signals generated. Need to adapt backtest...")
    return technical_signals


if __name__ == "__main__":
    backtest_technical()