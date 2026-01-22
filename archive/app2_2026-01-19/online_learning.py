# online_learning.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
from __future__ import annotations
import datetime as dt
import pandas as pd
from . import data as D, models as M, risk as R

class OnlineLearner:
    def __init__(self, model_bundle, symbols, update_interval_hours: int = 4):
        self.model_bundle = model_bundle
        self.symbols = symbols
        self.update_interval = update_interval_hours
        self.last_update = None

    def should_update(self) -> bool:
        """ИСПРАВЛЕННАЯ ЛОГИКА ВРЕМЕНИ"""
        if self.last_update is None:
            return True
        elapsed_hours = (dt.datetime.now() - self.last_update).total_seconds() / 3600
        return elapsed_hours >= self.update_interval

    def incremental_learn(self, new_data: dict):
        """Инкрементальное обучение на новых данных"""
        Xs, ys = [], []

        for symbol, prices in new_data.items():
            if prices is None or prices.empty:
                continue

            X, y = M.dataset(prices, horizon=2)
            if len(X) > 10:
                Xs.append(X)
                ys.append(y)

        if not Xs:
            return False

        X_new = pd.concat(Xs).sort_index()
        y_new = pd.concat(ys).reindex(X_new.index).fillna(0).astype(int)

        # Инкрементальное обучение - ТЕПЕРЬ РАБОТАЕТ
        try:
            M.partial_fit(self.model_bundle, X_new, y_new)
            self.last_update = dt.datetime.now()
            return True
        except Exception as e:
            print(f"Online learning failed: {e}")
            return False

    def fetch_recent_data(self, days: int = 1):
        """Получение свежих данных для обучения"""
        end = dt.date.today().isoformat()
        start = (dt.date.today() - dt.timedelta(days=days)).isoformat()

        new_prices = {}
        for symbol in self.symbols:
            try:
                df = D.fetch_range(symbol, start, end, interval='10min')
                if df is not None and not df.empty:
                    new_prices[symbol] = df
                    D.save_csv(symbol, df)
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")

        return new_prices