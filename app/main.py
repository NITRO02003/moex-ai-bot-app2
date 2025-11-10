from backtest import run_backtest
from config import config
from ai_analyzer import ai_analyzer


def main():
    # Обучение AI на исторических данных (опционально)
    print("Training AI models...")
    # Здесь нужно загрузить исторические данные для обучения
    # historical_data = load_historical_data()
    # ai_analyzer.train_models(historical_data)

    # Запуск бэктеста
    print("Starting backtest...")
    result = run_backtest(config)

    print("\n=== FINAL RESULTS ===")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")

    print(f"\nPerformance stats:")
    for step, stats in result.performance_stats.items():
        print(f"{step}: {stats['time_ms']:.2f}ms")


if __name__ == "__main__":
    main()