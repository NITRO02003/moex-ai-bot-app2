"""
ТЕСТ УЛЬТРА-ПРОСТОЙ СТРАТЕГИИ
"""
from app2 import data as D, models as M, risk as R, backtest as B
from app2.strategy import ultra_simple_test_strategy


def test_ultra_simple():
    print("🚀 ТЕСТ УЛЬТРА-ПРОСТОЙ СТРАТЕГИИ")
    print("=" * 50)

    symbols = ['SBER', 'GAZP']

    # Настройки
    rp = R.from_config()
    bt = B.BtParams(commission=0.0005, slippage_bps=1.0, horizon=1)
    bundle = M.load()

    for symbol in symbols:
        print(f"\n🔍 Тестируем {symbol}...")

        # Загружаем данные
        prices = D.load_csv(symbol)
        if prices.empty:
            print(f"❌ Нет данных для {symbol}")
            continue

        print(f"✅ Данные: {len(prices)} баров")

        # Временная подмена стратегии
        import app2.strategy as S
        original_strategy = S.signal_and_size
        S.signal_and_size = ultra_simple_test_strategy

        try:
            # Запускаем бэктест
            result = B.run_symbol(prices, bundle, rp, bt, 1000000.0, threshold=0.5)

            # Восстанавливаем стратегию
            S.signal_and_size = original_strategy

            # Результаты
            metrics = result['metrics']
            print(f"📊 РЕЗУЛЬТАТЫ:")
            print(f"   Сделок: {metrics['total_trades']}")
            print(f"   Доходность: {metrics['total_return']:.2%}")
            print(f"   Комиссии: {metrics['total_commissions']:,.0f} руб")

            if metrics['total_trades'] > 0:
                print("🎉 УРА! СДЕЛКИ ПОЯВИЛИСЬ!")
            else:
                print("😞 Сделок все еще нет...")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

            # Восстанавливаем стратегию в случае ошибки
            S.signal_and_size = original_strategy


if __name__ == "__main__":
    test_ultra_simple()