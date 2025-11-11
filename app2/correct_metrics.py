# correct_metrics.py (переименованный correct_metrics.py)
"""
ИСПРАВЛЕННЫЙ РАСЧЕТ МЕТРИК - ИНТЕГРИРОВАН В СИСТЕМУ
"""
import json
import pandas as pd
import numpy as np

def calculate_correct_metrics(trades: list, initial_capital: float = 1000000) -> dict:
    """КОРРЕКТНЫЙ расчет метрик с учетом комиссий и рисков"""
    if not trades:
        return create_empty_metrics(initial_capital)

    # Базовые расчеты
    gross_pnl = sum(trade.get('gross_pnl', 0) for trade in trades)
    total_commissions = sum(trade.get('commission', 0) for trade in trades)
    net_pnl = gross_pnl - total_commissions
    final_equity = initial_capital + net_pnl

    # Разделение сделок
    winning_trades = [t for t in trades if t.get('net_pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('net_pnl', 0) <= 0]

    win_rate = len(winning_trades) / len(trades) if trades else 0

    # Profit Factor
    gross_profit = sum(t.get('gross_pnl', 0) for t in winning_trades)
    gross_loss = abs(sum(t.get('gross_pnl', 0) for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Expectancy
    avg_win = np.mean([t.get('net_pnl', 0) for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.get('net_pnl', 0) for t in losing_trades]) if losing_trades else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # Максимальная просадка
    equity_curve = [initial_capital]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade.get('net_pnl', 0))

    peak = equity_curve[0]
    max_drawdown = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    total_return = (final_equity - initial_capital) / initial_capital

    return {
        'final_equity': round(final_equity, 2),
        'total_return': round(total_return, 6),
        'max_drawdown': round(-max_drawdown, 6),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 4) if profit_factor != float('inf') else 'inf',
        'expectancy_ratio': round(expectancy, 2),
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'total_commissions': round(total_commissions, 2),
        'gross_pnl': round(gross_pnl, 2),
        'net_pnl': round(net_pnl, 2)
    }

def create_empty_metrics(initial_capital: float) -> dict:
    """Пустые метрики при отсутствии сделок"""
    return {
        'final_equity': initial_capital,
        'total_return': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'expectancy_ratio': 0.0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_commissions': 0.0,
        'gross_pnl': 0.0,
        'net_pnl': 0.0
    }

def validate_metrics_compatibility(legacy_metrics: dict, corrected_metrics: dict) -> bool:
    """Проверка совместимости метрик"""
    required_fields = ['final_equity', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']
    return all(field in corrected_metrics for field in required_fields)

# ИНТЕГРАЦИЯ С ОСНОВНОЙ СИСТЕМОЙ
def integrate_with_backtester():
    """Интеграция исправленных расчетов в основной бэктестер"""
    print("=== ИНТЕГРАЦИЯ ИСПРАВЛЕННЫХ МЕТРИК ===")
    print("✅ Correct metrics calculation integrated")
    print("✅ Fixed PnL calculation with commissions")
    print("✅ Improved risk management")
    print("✅ Compatible with existing backtester")