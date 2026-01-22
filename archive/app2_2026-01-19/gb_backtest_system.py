# gb_backtest_system.py
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from app2 import data as D, risk as R, features as F, labels as L
from app2.backtest import BtParams, run_symbol
from app2.model_improvement import create_improved_features


class GBBacktestSystem:
    """Полная система бэктестирования для Gradient Boosting модели"""

    def __init__(self):
        self.gb_model = None
        self.feature_names = []
        self.calibration = {}

    def load_and_prepare_data(self, symbols: list, start_date: str, end_date: str) -> dict:
        """Загрузка и подготовка данных"""
        print("📥 ЗАГРУЗКА ДАННЫХ...")

        prices_data = {}
        for symbol in symbols:
            print(f"   {symbol}...", end=" ")
            try:
                # Пробуем загрузить существующие данные
                prices = D.load_csv(symbol)

                if prices.empty or len(prices) < 100:
                    print("загружаем заново...", end=" ")
                    prices = D.fetch_range(symbol, start_date, end_date, '10min', verbose=False)
                    if not prices.empty:
                        D.save_csv(symbol, prices)

                if not prices.empty:
                    prices_data[symbol] = prices
                    print(f"✅ {len(prices)} баров")
                else:
                    print("❌ нет данных")

            except Exception as e:
                print(f"❌ ошибка: {e}")

        return prices_data

    def train_gb_model(self, prices_data: dict, horizon: int = 2) -> bool:
        """Обучение GB модели на всех тикерах"""
        print("\n🤖 ОБУЧЕНИЕ GB МОДЕЛИ...")

        Xs, ys = [], []

        for symbol, prices in prices_data.items():
            if len(prices) < 100:
                continue

            try:
                # Используем улучшенные фичи
                X = create_improved_features(prices)
                y = L.y_updown(prices['close'].astype(float), horizon=horizon)

                # Очистка и выравнивание
                X, y = L.clean_xy(X, y)
                common_idx = X.index.intersection(y.index)

                if len(common_idx) > 50:
                    Xs.append(X.loc[common_idx])
                    ys.append(y.loc[common_idx])
                    print(f"   {symbol}: {len(common_idx)} samples")

            except Exception as e:
                print(f"   {symbol}: ошибка - {e}")

        if not Xs:
            print("❌ Недостаточно данных для обучения")
            return False

        # Объединяем все данные
        X_all = pd.concat(Xs)
        y_all = pd.concat(ys)

        # Сохраняем названия фич
        self.feature_names = X_all.columns.tolist()

        # Обучаем модель
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        self.gb_model.fit(X_all.values, y_all.values)

        # Сохраняем feature names вручную
        self.gb_model.feature_names_ = self.feature_names

        print(f"✅ Модель обучена: {len(X_all)} samples, {len(self.feature_names)} features")
        return True

    def calibrate_thresholds(self, prices_data: dict, horizon: int = 2) -> dict:
        """Калибровка порогов для всех тикеров"""
        print("\n🎯 КАЛИБРОВКА ПОРОГОВ...")

        calibration_results = {}

        for symbol, prices in prices_data.items():
            if len(prices) < 1000:
                continue

            print(f"   {symbol}...", end=" ")

            try:
                X = create_improved_features(prices)
                y_true = L.y_updown(prices['close'].astype(float), horizon=horizon)

                # Предсказания GB модели
                p_pred = self.gb_predict(X, prices['close'].astype(float))

                # Выравниваем индексы
                common_idx = X.index.intersection(y_true.index)
                if len(common_idx) < 100:
                    print("недостаточно данных")
                    continue

                y_t = y_true.loc[common_idx]
                p_t = p_pred.loc[common_idx]

                # Тестируем разные пороги
                thresholds = np.arange(0.45, 0.75, 0.02)
                best_f1 = 0
                best_threshold = 0.55

                for threshold in thresholds:
                    y_pred = (p_t >= threshold).astype(int)
                    f1 = f1_score(y_t, y_pred, zero_division=0)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

                # Также считаем AUC
                auc_score = roc_auc_score(y_t, p_t)

                calibration_results[symbol] = {
                    'optimal_threshold': round(best_threshold, 3),
                    'f1_score': round(best_f1, 4),
                    'auc_score': round(auc_score, 4),
                    'samples': len(common_idx)
                }

                print(f"порог: {best_threshold:.3f}, F1: {best_f1:.3f}, AUC: {auc_score:.3f}")

            except Exception as e:
                print(f"ошибка: {e}")

        self.calibration = calibration_results
        return calibration_results

    def gb_predict(self, X: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Предсказание с GB моделью"""
        if self.gb_model is None:
            return pd.Series(0.5, index=X.index, name='p_up')

        # Выравниваем фичи
        available_cols = [col for col in self.feature_names if col in X.columns]
        missing_cols = [col for col in self.feature_names if col not in X.columns]

        if available_cols:
            X_aligned = X[available_cols].copy()
            for col in missing_cols:
                X_aligned[col] = 0.0
            X_aligned = X_aligned[self.feature_names]
        else:
            X_aligned = pd.DataFrame(0, index=X.index, columns=self.feature_names)

        try:
            probabilities = self.gb_model.predict_proba(X_aligned.values)[:, 1]
            return pd.Series(probabilities, index=X.index, name='p_up')
        except Exception as e:
            print(f"⚠️ Ошибка предсказания: {e}")
            return pd.Series(0.5, index=X.index, name='p_up')

    def gb_strategy(self, prices: pd.DataFrame, model_bundle, rp, equity: float, threshold: float = 0.55):
        """Стратегия для GB модели"""
        if len(prices) < 100:
            return self.create_empty_signal(prices)

        # Используем улучшенные фичи
        X = create_improved_features(prices)
        p = self.gb_predict(X, prices['close'].astype(float))

        # БАЗОВЫЕ УСЛОВИЯ (без сложных фильтров для теста)
        long_cond = p > threshold
        short_cond = p < (1 - threshold)

        side = pd.Series(0, index=p.index)
        side[long_cond] = 1
        side[short_cond] = -1

        # ФИКСИРОВАННЫЙ РАЗМЕР для тестирования
        size = side * (equity * 0.01)  # 1% капитала

        result = pd.DataFrame({'p': p, 'side': side, 'size': size}, index=prices.index)

        # ДИАГНОСТИКА
        active_signals = result[result['side'] != 0]
        total_signals = len(active_signals)
        long_signals = (result['side'] == 1).sum()
        short_signals = (result['side'] == -1).sum()

        print(f"      Сигналы: {total_signals} (Long: {long_signals}, Short: {short_signals})")
        if total_signals > 0:
            avg_confidence = result[result['side'] != 0]['p'].mean()
            print(f"      Средняя уверенность: {avg_confidence:.3f}")

        return result

    def create_empty_signal(self, prices):
        return pd.DataFrame({
            'p': 0.5,
            'side': 0,
            'size': 0.0
        }, index=prices.index)

    def run_comprehensive_backtest(self, symbols: list, start_date: str, end_date: str):
        """Полный цикл бэктестирования"""
        print("=" * 60)
        print("🚀 ЗАПУСК КОМПЛЕКСНОГО БЭКТЕСТА GB МОДЕЛИ")
        print("=" * 60)

        # 1. Загрузка данных
        prices_data = self.load_and_prepare_data(symbols, start_date, end_date)

        if not prices_data:
            print("❌ Нет данных для тестирования")
            return None

        # 2. Обучение модели
        if not self.train_gb_model(prices_data):
            return None

        # 3. Калибровка порогов
        calibration = self.calibrate_thresholds(prices_data)

        if not calibration:
            print("❌ Не удалось провести калибровку")
            return None

        # 4. Бэктест
        print("\n📊 ЗАПУСК БЭКТЕСТОВ:")
        results = {}

        rp = R.from_config()
        bt = BtParams(commission=0.0005, slippage_bps=1.0, horizon=2)

        for symbol in symbols:
            if symbol not in prices_data or symbol not in calibration:
                continue

            prices = prices_data[symbol]
            threshold = calibration[symbol]['optimal_threshold']

            print(f"\n🔍 {symbol}:")
            print(f"   Период: {prices.index.min().strftime('%Y-%m-%d')} - {prices.index.max().strftime('%Y-%m-%d')}")
            print(f"   Баров: {len(prices):,}")
            print(f"   Порог: {threshold}, AUC: {calibration[symbol]['auc_score']}")

            try:
                # Создаем совместимый bundle
                gb_bundle = {
                    'model': self.gb_model,
                    'feature_names': self.feature_names,
                    'predict_proba': lambda X, close: self.gb_predict(X, close)
                }

                # Временно подменяем стратегию
                import app2.strategy as S
                original_strategy = S.signal_and_size
                S.signal_and_size = self.gb_strategy

                # Запускаем бэктест
                result = run_symbol(prices, gb_bundle, rp, bt, 1000000.0, threshold=threshold)
                results[symbol] = result

                # Восстанавливаем стратегию
                S.signal_and_size = original_strategy

                metrics = result['metrics']
                print(f"   📈 Результат:")
                print(f"      Итоговый капитал: {metrics['final_equity']:,.0f} руб")
                print(f"      Доходность: {metrics['total_return']:+.2%}")
                print(f"      Сделок: {metrics['total_trades']}")
                print(f"      Win Rate: {metrics['win_rate']:.1%}")
                print(f"      Комиссии: {metrics['total_commissions']:,.0f} руб")

                # Проверка аномалий
                if metrics['final_equity'] > 2000000 or metrics['final_equity'] < 500000:
                    print(f"      ⚠️  ВНИМАНИЕ: Подозрительный результат!")

            except Exception as e:
                print(f"   ❌ Ошибка бэктеста: {e}")
                import traceback
                traceback.print_exc()

        # 5. Сводка
        self.print_summary(results, calibration)

        return results

    def print_summary(self, results: dict, calibration: dict):
        """Печать итоговой сводки"""
        print("\n" + "=" * 60)
        print("📈 ИТОГОВАЯ СВОДКА")
        print("=" * 60)

        total_return = 0
        successful_symbols = 0
        total_trades = 0

        for symbol, result in results.items():
            metrics = result['metrics']
            calib = calibration.get(symbol, {})

            status = "✅" if metrics['final_equity'] > 1000000 else "❌"
            return_pct = metrics['total_return'] * 100

            print(f"{status} {symbol:6} | {return_pct:+6.1f}% | Сделок: {metrics['total_trades']:3d} | "
                  f"WR: {metrics['win_rate']:5.1%} | Порог: {calib.get('optimal_threshold', 0):.2f} | "
                  f"AUC: {calib.get('auc_score', 0):.3f}")

            if metrics['total_trades'] > 0:
                total_return += metrics['total_return']
                successful_symbols += 1
                total_trades += metrics['total_trades']

        if successful_symbols > 0:
            avg_return = total_return / successful_symbols * 100
            avg_trades = total_trades / successful_symbols

            print(f"\n📊 СРЕДНИЕ ПОКАЗАТЕЛИ:")
            print(f"   Доходность: {avg_return:+.1f}%")
            print(f"   Сделок на тикер: {avg_trades:.0f}")
            print(f"   Успешных тикеров: {successful_symbols}/{len(results)}")

        # Сохраняем результаты
        self.save_results(results, calibration)

    def save_results(self, results: dict, calibration: dict):
        """Сохранение результатов"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'calibration': calibration,
            'results': {}
        }

        for symbol, result in results.items():
            output['results'][symbol] = result['metrics']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        from app2.paths import REPORTS_DIR

        filename = f"gb_comprehensive_results_{timestamp}.json"
        path = REPORTS_DIR / filename

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Результаты сохранены в: {path}")


def main():
    """Основная функция тестирования"""
    system = GBBacktestSystem()

    # Настройки теста
    symbols = ['GAZP', 'GMKN', 'LKOH', 'ROSN', 'SBER', 'YNDX']
    start_date = '2022-01-15'
    end_date = '2024-12-31'

    # Запуск теста
    results = system.run_comprehensive_backtest(symbols, start_date, end_date)

    if results:
        print(f"\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    else:
        print(f"\n💥 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО С ОШИБКАМИ!")


def run_high_threshold_test():
    symbols = ['GAZP', 'GMKN', 'LKOH', 'ROSN', 'SBER', 'YNDX']

    from gb_backtest_system import GBBacktestSystem
    system = GBBacktestSystem()

    # Загружаем данные
    prices_data = system.load_and_prepare_data(symbols, '2022-01-15', '2024-12-31')

    # Обучаем модель
    system.train_gb_model(prices_data)

    # Бэктест с высокими порогами
    results = {}
    rp = R.from_config()
    bt = BtParams(commission=0.0005, slippage_bps=1.0, horizon=2)

    for symbol in symbols:
        if symbol not in prices_data:
            continue

        print(f"🔍 {symbol} с порогом 0.65...")

        # ВРЕМЕННО ПОДМЕНЯЕМ СТРАТЕГИЮ
        import app2.strategy as S
        original_strategy = S.signal_and_size

        # Используем консервативную стратегию
        S.signal_and_size = lambda prices, model_bundle, rp, equity, threshold: conservative_strategy(
            prices, model_bundle, rp, equity
        )

        # Запускаем бэктест
        result = run_symbol(prices_data[symbol], system.gb_model, rp, bt, 1000000.0, threshold=0.65)
        results[symbol] = result

        # ВОССТАНАВЛИВАЕМ СТРАТЕГИЮ
        S.signal_and_size = original_strategy

        # ВЫВОДИМ РЕЗУЛЬТАТЫ
        metrics = result['metrics']
        print(f"   Сделок: {metrics['total_trades']} (было 8000+)")
        print(f"   Комиссии: {metrics['total_commissions']:,.0f} руб")
        print(f"   Чистая прибыль: {metrics['net_pnl']:,.0f} руб")

    return results

if __name__ == "__main__":
    main()