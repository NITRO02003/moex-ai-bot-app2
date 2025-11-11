# comprehensive_test.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from app2 import data as D, models as M, risk as R, strategy as S
from app2.backtest import run_symbol, BtParams
from app2.model_improvement import train_gb_model, create_improved_features


def run_comprehensive_test():
    """Комплексный тест на всех тикерах за период 2022-2024"""

    symbols = ['GAZP', 'GMKN', 'LKOH', 'ROSN', 'SBER', 'YNDX']
    start_date = '2022-01-15'
    end_date = '2024-12-31'

    print("=== КОМПЛЕКСНЫЙ ТЕСТ МОДЕЛИ ===")
    print(f"Период: {start_date} - {end_date}")
    print(f"Тикеры: {', '.join(symbols)}")
    print()

    # Загружаем и обновляем данные
    prices_data = {}
    for symbol in symbols:
        print(f"📥 Загружаем данные для {symbol}...")
        try:
            # Пробуем загрузить существующие данные
            prices = D.load_csv(symbol)

            # Проверяем покрытие периода
            if not prices.empty:
                first_date = prices.index.min().strftime('%Y-%m-%d')
                last_date = prices.index.max().strftime('%Y-%m-%d')
                print(f"   Данные: {first_date} - {last_date}")

                # Если данных недостаточно, обновляем
                if first_date > start_date or last_date < end_date:
                    print(f"   ⚠️  Обновляем данные...")
                    new_prices = D.fetch_range(symbol, start_date, end_date, '10min', verbose=False)
                    if not new_prices.empty:
                        prices = new_prices
                        D.save_csv(symbol, prices)
                        print(f"   ✅ Обновлено: {len(prices)} баров")
            else:
                print(f"   ⚠️  Нет данных, загружаем...")
                prices = D.fetch_range(symbol, start_date, end_date, '10min', verbose=False)
                if not prices.empty:
                    D.save_csv(symbol, prices)
                    print(f"   ✅ Загружено: {len(prices)} баров")

            prices_data[symbol] = prices

        except Exception as e:
            print(f"   ❌ Ошибка загрузки {symbol}: {e}")
            prices_data[symbol] = pd.DataFrame()

    print("\n" + "=" * 50)

    # Обучаем GB модель
    print("🤖 Обучаем GradientBoosting модель...")
    valid_prices = {s: p for s, p in prices_data.items() if not p.empty and len(p) > 100}

    if not valid_prices:
        print("❌ Нет данных для обучения")
        return

    gb_model = train_gb_model(valid_prices, horizon=2)

    if gb_model is None:
        print("❌ Не удалось обучить GB модель")
        return

    print("✅ GB модель обучена успешно")

    # Тестируем на каждом тикере
    print("\n" + "=" * 50)
    print("🧪 ЗАПУСК БЭКТЕСТОВ:")

    rp = R.from_config()
    bt = BtParams(commission=0.0005, slippage_bps=1.0, horizon=2)

    results = {}

    for symbol in symbols:
        if symbol not in valid_prices:
            continue

        prices = valid_prices[symbol]
        print(f"\n📊 {symbol}: {len(prices)} баров")

        try:
            # Создаем bundle для GB модели
            gb_bundle = {
                'model': gb_model,
                'feature_names': getattr(gb_model, 'feature_names_', []),
                'predict_proba': lambda X, close: gb_predict(gb_model, X, close)
            }

            # Запускаем бэктест
            result = run_symbol(prices, gb_bundle, rp, bt, 1000000.0, threshold=0.65)
            results[symbol] = result

            metrics = result['metrics']
            print(f"   Итоговый капитал: {metrics['final_equity']:,.0f} руб")
            print(f"   Доходность: {metrics['total_return']:.2%}")
            print(f"   Сделок: {metrics['total_trades']}")
            print(f"   Win Rate: {metrics['win_rate']:.1%}")
            print(f"   Комиссии: {metrics['total_commissions']:,.0f} руб")
            print(f"   Макс просадка: {metrics['max_drawdown']:.2%}")

            # Проверка на аномалии
            if metrics['final_equity'] > 5000000 or metrics['final_equity'] < 500000:
                print(f"   ⚠️  ПОДОЗРИТЕЛЬНЫЙ РЕЗУЛЬТАТ!")

        except Exception as e:
            print(f"   ❌ Ошибка бэктеста {symbol}: {e}")

    # Сводка
    print("\n" + "=" * 50)
    print("📈 ИТОГОВАЯ СВОДКА:")

    total_return = 0
    successful_symbols = 0

    for symbol, result in results.items():
        metrics = result['metrics']
        total_return += metrics['total_return']
        successful_symbols += 1

        status = "✅" if metrics['final_equity'] > 1000000 else "❌"
        print(
            f"{status} {symbol}: {metrics['total_return']:+.2%} | Сделок: {metrics['total_trades']} | WR: {metrics['win_rate']:.1%}")

    if successful_symbols > 0:
        avg_return = total_return / successful_symbols
        print(f"\n📊 Средняя доходность: {avg_return:.2%}")
        print(f"🎯 Успешных тикеров: {successful_symbols}/{len(symbols)}")

    return results


def gb_predict(model, X: pd.DataFrame, close: pd.Series) -> pd.Series:
    """Предсказание с GB моделью"""
    if hasattr(model, 'feature_names_'):
        # Выравниваем фичи
        available_cols = [col for col in model.feature_names_ if col in X.columns]
        missing_cols = [col for col in model.feature_names_ if col not in X.columns]

        if available_cols:
            X_aligned = X[available_cols].copy()
            for col in missing_cols:
                X_aligned[col] = 0.0
            X_aligned = X_aligned[model.feature_names_]
        else:
            X_aligned = pd.DataFrame(0, index=X.index, columns=model.feature_names_)
    else:
        X_aligned = X

    try:
        probabilities = model.predict_proba(X_aligned.values)[:, 1]
        return pd.Series(probabilities, index=X.index, name='p_up')
    except Exception as e:
        print(f"⚠️  Ошибка предсказания GB: {e}")
        return pd.Series(0.5, index=X.index, name='p_up')


def conservative_gb_strategy(prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.7):
    """Консервативная стратегия с GB моделью"""
    from app2 import features as F

    if len(prices) < 100:
        return create_empty_signal(prices)

    # Используем улучшенные фичи
    X = create_improved_features(prices)
    p = model_bundle['predict_proba'](X, prices['close'].astype(float))

    # СТРОГИЕ ФИЛЬТРЫ
    close = prices['close'].astype(float)
    volume = prices.get('volume', pd.Series(1, index=p.index))

    # Фильтр объема
    volume_ma = volume.rolling(20).mean()
    volume_ok = volume > volume_ma * 1.3

    # Фильтр волатильности
    volatility = close.pct_change().rolling(20).std()
    volatility_ok = volatility.between(volatility.quantile(0.3), volatility.quantile(0.7))

    # Высокие пороги
    long_cond = (p > 0.75) & volume_ok & volatility_ok
    short_cond = (p < 0.25) & volume_ok & volatility_ok

    side = pd.Series(0, index=p.index)
    side[long_cond] = 1
    side[short_cond] = -1

    # Маленький размер позиции
    size = side * (equity * 0.01)  # 1% капитала

    result = pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)

    active_count = len(result[result['side'] != 0])
    print(f"   Консервативная GB: {active_count} сигналов")

    return result


def create_empty_signal(prices):
    return pd.DataFrame({
        'p': 0.5,
        'side': 0,
        'size': 0.0
    }, index=prices.index)


if __name__ == "__main__":
    # Тестируем
    results = run_comprehensive_test()

    # Сохраняем результаты
    if results:
        import json
        from datetime import datetime

        summary = {}
        for symbol, result in results.items():
            summary[symbol] = result['metrics']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_test_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Результаты сохранены в {filename}")