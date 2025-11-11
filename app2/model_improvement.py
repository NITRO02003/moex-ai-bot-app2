# model_improvement.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from . import features as F, labels as L


def create_improved_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Улучшенные фичи для MOEX"""
    base = F.build(prices)

    # ДОБАВЛЯЕМ КЛЮЧЕВЫЕ ФИЧИ
    close = prices['close'].astype(float)
    high = prices['high'].astype(float)
    low = prices['low'].astype(float)
    volume = prices.get('volume', pd.Series(1, index=prices.index))

    improved = pd.DataFrame(index=base.index)

    # ЦЕНА ОТНОСИТЕЛЬНО СКОЛЬЗЯЩИХ
    for window in [5, 10, 20]:
        sma = close.rolling(window).mean()
        improved[f'price_sma_ratio_{window}'] = close / sma - 1

    # МОМЕНТУМ
    improved['momentum_5'] = close.pct_change(5)
    improved['momentum_10'] = close.pct_change(10)

    # ВОЛАТИЛЬНОСТЬ ОТНОСИТЕЛЬНО ИСТОРИИ
    vol_5 = close.pct_change().rolling(5).std()
    vol_20 = close.pct_change().rolling(20).std()
    improved['vol_ratio'] = vol_5 / vol_20

    # ОБЪЕМ ОТНОСИТЕЛЬНО СРЕДНЕГО
    volume_ma = volume.rolling(20).mean()
    improved['volume_ratio'] = volume / volume_ma

    # ПРОСТЫЕ ПАТТЕРНЫ
    improved['high_low_range'] = (high - low) / close
    improved['close_position'] = (close - low) / (high - low + 1e-9)

    # ОБЪЕДИНЯЕМ С БАЗОВЫМИ ФИЧАМИ
    result = pd.concat([base, improved], axis=1)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_gb_model(prices_by_symbol: dict, horizon: int = 2):
    """Обучение GradientBoosting как временное решение"""
    Xs, ys = [], []

    for symbol, prices in prices_by_symbol.items():
        if prices is None or len(prices) < 100:
            continue

        X = create_improved_features(prices)
        y = L.y_updown(prices['close'].astype(float), horizon=horizon)

        # Очистка и выравнивание
        X, y = L.clean_xy(X, y)
        common_idx = X.index.intersection(y.index)

        if len(common_idx) > 50:
            Xs.append(X.loc[common_idx])
            ys.append(y.loc[common_idx])

    if not Xs:
        return None

    X_all = pd.concat(Xs)
    y_all = pd.concat(ys)

    # Обучаем GradientBoosting
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_all.values, y_all.values)

    # СОХРАНЯЕМ СПИСОК ФИЧ ВРУЧНУЮ
    model.feature_names_ = X_all.columns.tolist()

    return model


def run_gb_experiment():
    """Эксперимент с GradientBoosting моделью - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    from . import data as D, models as M, labels as L

    symbols = ['SBER', 'GAZP']
    prices_data = {s: D.load_csv(s) for s in symbols}

    # Обучаем GB модель
    gb_model = train_gb_model(prices_data, horizon=2)

    if gb_model is not None:
        print("✅ GradientBoosting модель обучена успешно")

        # Сравниваем предсказания
        for symbol in symbols:
            prices = prices_data[symbol]
            if len(prices) < 100:
                continue

            X_new = create_improved_features(prices)
            y_true = L.y_updown(prices['close'].astype(float), horizon=2)

            # Предсказания текущей модели
            bundle = M.load()
            p_current = M.predict_proba(bundle, X_new, prices['close'])

            # Предсказания GB модели - ИСПРАВЛЕННЫЙ КОД
            if hasattr(gb_model, 'feature_names_'):
                # Используем сохраненные названия фич
                X_aligned = X_new.reindex(columns=gb_model.feature_names_, fill_value=0)
                p_gb = gb_model.predict_proba(X_aligned.values)[:, 1]
            else:
                # Если нет названий, используем напрямую (может вызвать ошибку если фичи не совпадают)
                p_gb = gb_model.predict_proba(X_new.values)[:, 1]

            # Сравнение AUC
            from sklearn.metrics import roc_auc_score
            common_idx = X_new.index.intersection(y_true.index)
            if len(common_idx) > 0:
                auc_current = roc_auc_score(y_true.loc[common_idx], p_current.loc[common_idx])

                # Правильно индексируем p_gb
                p_gb_series = pd.Series(p_gb, index=X_new.index)
                auc_gb = roc_auc_score(y_true.loc[common_idx], p_gb_series.loc[common_idx])
            else:
                auc_current = 0.5
                auc_gb = 0.5

            print(f"{symbol}: AUC текущей = {auc_current:.3f}, AUC GB = {auc_gb:.3f}")

        return gb_model
    else:
        print("❌ Не удалось обучить GB модель")
        return None