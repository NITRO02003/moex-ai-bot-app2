# app/analyze_results.py
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_backtest_results():
    out_dir = Path("out")

    # Загружаем результаты
    equity = pd.read_csv(out_dir / "equity.csv", index_col=0, parse_dates=True)
    trades = pd.read_csv(out_dir / "trades.csv", parse_dates=True) if (
                out_dir / "trades.csv").exists() else pd.DataFrame()

    print("=== ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ===")

    # Анализ эквити
    print(f"\nЭКВИТИ:")
    print(f"Начальный капитал: 1,000,000.00")
    print(f"Конечный капитал: {equity['equity'].iloc[-1]:.2f}")
    print(f"Общая доходность: {(equity['equity'].iloc[-1] / 1000000 - 1) * 100:.2f}%")

    # Анализ просадок
    equity['peak'] = equity['equity'].cummax()
    equity['drawdown'] = (equity['peak'] - equity['equity']) / equity['peak']
    max_dd = equity['drawdown'].max()
    print(f"Максимальная просадка: {max_dd * 100:.2f}%")

    # Анализ сделок
    if not trades.empty:
        print(f"\nСДЕЛКИ:")
        print(f"Всего сделок: {len(trades)}")
        print(f"Сделок в день: {len(trades) / len(equity):.2f}")

        # Анализ по символам
        symbol_stats = trades.groupby('symbol').agg({
            'quantity': 'count',
            'signal': 'mean',
            'risk_score': 'mean'
        }).rename(columns={'quantity': 'trades', 'signal': 'avg_signal'})
        print(f"\nСТАТИСТИКА ПО СИМВОЛАМ:")
        print(symbol_stats)

        # Анализ PnL по сделкам (упрощенный)
        trades['pnl'] = 0  # В реальности нужно рассчитывать из позиций
        win_rate = (trades['pnl'] > 0).mean() if 'pnl' in trades.columns else 0
        print(f"Винрейт: {win_rate * 100:.2f}%")

    # Визуализация
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    equity['equity'].plot(title='Кривая капитала')
    plt.ylabel('Капитал')

    plt.subplot(2, 2, 2)
    equity['drawdown'].plot(title='Просадка', color='red')
    plt.ylabel('Просадка')

    if not trades.empty:
        plt.subplot(2, 2, 3)
        trades['symbol'].value_counts().plot(kind='bar', title='Сделки по символам')

        plt.subplot(2, 2, 4)
        trades.groupby('symbol')['signal'].mean().plot(kind='bar', title='Средний сигнал по символам')

    plt.tight_layout()
    plt.savefig(out_dir / 'analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nГрафики сохранены в: {out_dir / 'analysis.png'}")


if __name__ == "__main__":
    analyze_backtest_results()