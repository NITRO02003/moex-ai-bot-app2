# app/quick_optimize.py
import pandas as pd
import numpy as np
import json
from pathlib import Path


def analyze_and_optimize():
    """Быстрый анализ и оптимизация на основе результатов"""

    # Анализируем текущие результаты
    out_dir = Path("out")

    # Загружаем сделки для анализа
    trades_path = out_dir / "trades.csv"
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        print("=== АНАЛИЗ ТЕКУЩИХ РЕЗУЛЬТАТОВ ===")
        print(f"Всего сделок: {len(trades)}")

        if 'signal' in trades.columns:
            signal_stats = trades['signal'].describe()
            print("\nСтатистика сигналов:")
            print(f"Средний: {signal_stats['mean']:.3f}")
            print(f"Стандартное отклонение: {signal_stats['std']:.3f}")
            print(f"Минимальный: {signal_stats['min']:.3f}")
            print(f"Максимальный: {signal_stats['max']:.3f}")

            # Анализ силы сигналов
            weak_signals = len(trades[abs(trades['signal']) < 0.3])
            strong_signals = len(trades[abs(trades['signal']) > 0.7])
            print(f"\nСлабые сигналы (|signal| < 0.3): {weak_signals} ({weak_signals / len(trades) * 100:.1f}%)")
            print(f"Сильные сигналы (|signal| > 0.7): {strong_signals} ({strong_signals / len(trades) * 100:.1f}%)")

    # Рекомендации по оптимизации
    print("\n=== РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ ===")

    recommendations = []

    # 1. Оптимизация порога сигнала
    recommendations.append("1. Увеличить минимальный порог AI сигнала до 0.4-0.5")
    recommendations.append("2. Уменьшить базовый размер позиции до 1-1.5%")
    recommendations.append("3. Добавить фильтр по объему (min volume ratio > 1.5)")
    recommendations.append("4. Добавить фильтр по волатильности (max volatility < 0.03)")
    recommendations.append("5. Увеличить минимальную сумму сделки до 5000 руб")

    for rec in recommendations:
        print(f"• {rec}")

    # Создаем конфиг с оптимизированными параметрами
    optimized_config = {
        "min_signal_strength": 0.4,
        "base_position_size": 0.012,
        "min_volume_ratio": 1.5,
        "max_volatility": 0.03,
        "min_trade_value": 5000,
        "momentum_threshold": 0.003,
        "use_aggressive_filters": True
    }

    # Сохраняем рекомендации
    with open(out_dir / "optimization_recommendations.json", "w", encoding="utf-8") as f:
        json.dump({
            "recommendations": recommendations,
            "parameters": optimized_config
        }, f, indent=2, ensure_ascii=False)

    print(f"\nРекомендации сохранены в: {out_dir / 'optimization_recommendations.json'}")


def create_aggressive_strategy():
    """Создание агрессивной стратегии для тестирования"""

    aggressive_params = {
        "min_signal_strength": 0.5,  # Только сильные сигналы
        "min_volume_factor": 1.5,  # Высокий объем
        "max_volatility": 0.025,  # Низкая волатильность
        "min_trend_strength": 0.15,  # Сильный тренд
        "momentum_threshold": 0.004,  # Явный моментум
        "base_position_size": 0.02,  # Стандартный размер
        "max_position_size": 0.06  # Лимит размера
    }

    print("=== АГРЕССИВНАЯ СТРАТЕГИЯ ===")
    print("Параметры для высококачественных сигналов:")
    for key, value in aggressive_params.items():
        print(f"  {key}: {value}")

    return aggressive_params


def create_conservative_strategy():
    """Создание консервативной стратегии"""

    conservative_params = {
        "min_signal_strength": 0.3,  # Умеренные сигналы
        "min_volume_factor": 1.2,  # Средний объем
        "max_volatility": 0.04,  # Средняя волатильность
        "min_trend_strength": 0.08,  # Слабый тренд
        "momentum_threshold": 0.002,  # Легкий моментум
        "base_position_size": 0.008,  # Малый размер
        "max_position_size": 0.04  # Строгий лимит
    }

    print("\n=== КОНСЕРВАТИВНАЯ СТРАТЕГИЯ ===")
    print("Параметры для минимизации рисков:")
    for key, value in conservative_params.items():
        print(f"  {key}: {value}")

    return conservative_params


if __name__ == "__main__":
    analyze_and_optimize()
    print("\n" + "=" * 50)
    aggressive = create_aggressive_strategy()
    print("\n" + "=" * 50)
    conservative = create_conservative_strategy()

    print(f"\nЗапустите improved_strategy с этими параметрами для тестирования!")