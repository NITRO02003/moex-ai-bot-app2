"""
СИСТЕМА ПОЛНОЙ ДИАГНОСТИКИ ПРОБЛЕМЫ 0 СДЕЛОК
"""
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from . import data as D, models as M, features as F, labels as L


class ZeroTradesDebugger:
    def __init__(self):
        self.results = {}

    def run_comprehensive_debug(self, symbols: list):
        """Запуск комплексной диагностики"""
        print("🔍 ЗАПУСК КОМПЛЕКСНОЙ ДИАГНОСТИКИ 0 СДЕЛОК")
        print("=" * 60)

        for symbol in symbols:
            print(f"\n📊 ДИАГНОСТИКА {symbol}:")
            self.debug_symbol(symbol)

        self.generate_report()

    def debug_symbol(self, symbol: str):
        """Диагностика для одного символа"""
        # 1. Проверка данных
        prices = D.load_csv(symbol)
        if prices.empty:
            print(f"   ❌ Нет данных для {symbol}")
            return

        print(f"   ✅ Данные: {len(prices)} баров")
        print(f"   📅 Период: {prices.index.min()} - {prices.index.max()}")

        # 2. Проверка фич
        try:
            X = F.build(prices)
            print(f"   ✅ Фичи построены: {X.shape[1]} признаков")

            # Проверка на NaN/Inf
            nan_count = X.isna().sum().sum()
            inf_count = np.isinf(X.values).sum()
            print(f"   📊 Качество фич: NaN={nan_count}, Inf={inf_count}")

        except Exception as e:
            print(f"   ❌ Ошибка построения фич: {e}")
            return

        # 3. Проверка модели
        bundle = M.load()
        if not bundle.get('cols'):
            print(f"   ❌ Модель не обучена или нет признаков")
            return

        print(f"   ✅ Модель загружена: {len(bundle['cols'])} признаков в модели")

        # 4. Проверка предсказаний
        try:
            p = M.predict_proba(bundle, X, prices['close'].astype(float))

            # Детальная статистика предсказаний
            stats = {
                'min': p.min(),
                'max': p.max(),
                'mean': p.mean(),
                'std': p.std(),
                'count_>0.5': (p > 0.5).sum(),
                'count_>0.6': (p > 0.6).sum(),
                'count_>0.7': (p > 0.7).sum(),
                'count_<0.5': (p < 0.5).sum(),
                'count_<0.4': (p < 0.4).sum(),
                'count_<0.3': (p < 0.3).sum()
            }

            print(f"   📈 Статистика предсказаний:")
            for key, value in stats.items():
                print(f"      {key}: {value}")

            # Проверка распределения
            bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
            hist = pd.cut(p, bins=bins).value_counts().sort_index()
            print(f"   📊 Распределение предсказаний:")
            for interval, count in hist.items():
                print(f"      {interval}: {count} баров")

            self.results[symbol] = {
                'prices_count': len(prices),
                'features_count': X.shape[1],
                'prediction_stats': stats,
                'prediction_histogram': hist.to_dict()
            }

        except Exception as e:
            print(f"   ❌ Ошибка предсказаний: {e}")

    def generate_report(self):
        """Генерация отчета диагностики"""
        report = []
        report.append("# 🐛 ДИАГНОСТИКА ПРОБЛЕМЫ 0 СДЕЛОК")
        report.append(f"**Время:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for symbol, data in self.results.items():
            report.append(f"## {symbol}")
            report.append(f"- Баров данных: {data['prices_count']}")
            report.append(f"- Признаков: {data['features_count']}")

            stats = data['prediction_stats']
            report.append("### Статистика предсказаний:")
            report.append(f"- Min: {stats['min']:.3f}")
            report.append(f"- Max: {stats['max']:.3f}")
            report.append(f"- Mean: {stats['mean']:.3f}")
            report.append(f"- >0.5: {stats['count_>0.5']}")
            report.append(f"- >0.6: {stats['count_>0.6']}")
            report.append(f"- <0.5: {stats['count_<0.5']}")
            report.append(f"- <0.4: {stats['count_<0.4']}")

        report.append("")
        report.append("## 💡 ВОЗМОЖНЫЕ ПРИЧИНЫ И РЕШЕНИЯ")
        report.append("""
### ❌ ВСЕ ПРЕДСКАЗАНИЯ ~0.5
- **Причина:** Модель не обучена или данные не соответствуют обучению
- **Решение:** Переобучить модель на актуальных данных

### ❌ ПРЕДСКАЗАНИЯ В ОЧЕНЬ УЗКОМ ДИАПАЗОНЕ (например, 0.48-0.52)
- **Причина:** Слишком сильная регуляризация или недостаточно признаков
- **Решение:** Уменьшить регуляризацию, добавить более информативные признаки

### ❌ ПРЕДСКАЗАНИЯ ЕСТЬ, НО СТРАТЕГИЯ НЕ ВИДИТ СИГНАЛОВ
- **Причина:** Ошибка в логике стратегии или фильтрах
- **Решение:** Упростить стратегию до минимальной рабочей версии

### ❌ ПРОБЛЕМА С ВЫРАВНИВАНИЕМ ПРИЗНАКОВ
- **Причина:** Признаки при обучении и предсказании не совпадают
- **Решение:** Проверить выравнивание признаков в models.py
""")

        # Сохранение отчета
        from .paths import REPORTS_DIR
        report_path = REPORTS_DIR / f"zero_trades_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"\n💾 Отчет диагностики сохранен: {report_path}")
        return report_path


def run_zero_trades_debug():
    """Запуск диагностики проблемы 0 сделок"""
    debugger = ZeroTradesDebugger()
    symbols = ['SBER', 'GAZP']
    return debugger.run_comprehensive_debug(symbols)


if __name__ == "__main__":
    run_zero_trades_debug()