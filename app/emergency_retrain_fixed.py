# app/emergency_retrain_fixed.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
from pathlib import Path


def emergency_retrain_fixed():
    """Экстренное переобучение со стационарными фичами и правильными метками"""
    print("=== EMERGENCY RETRAIN WITH FIXED DATA ===")

    # Импортируем наши исправленные функции
    from app.stationary_features import calculate_stationary_features
    from app.fixed_labels import make_better_labels
    from app.utils import load_all
    from app.config import config

    # Загружаем данные
    data = load_all("data", config.symbols_cfg.symbols)

    frames = []
    for s, df in data.items():
        # Исправленные фичи
        feats = calculate_stationary_features(df)
        feats["symbol"] = s
        feats["close"] = df["close"]
        frames.append(feats)

    # Объединяем датасет
    ds = pd.concat(frames).sort_index()

    # Исправленные метки
    y = make_better_labels(ds["close"])

    # Фичи (исключаем нечисловые колонки)
    feature_cols = [col for col in ds.columns if col not in ['symbol', 'close']]
    X = ds[feature_cols].copy()

    # Убираем NaN
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[mask]
    y = y[mask]

    print(f"Fixed dataset: {X.shape}, labels: {y.value_counts().to_dict()}")

    # Балансировка классов
    from sklearn.utils import class_weight
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    print(f"Class weights: {class_weights}")

    # Тестируем разные модели
    models = {
        'rf_balanced': RandomForestClassifier(
            n_estimators=200, max_depth=20,
            class_weight=class_weights, random_state=42, n_jobs=-1
        ),
        'gbm_balanced': GradientBoostingClassifier(
            n_estimators=150, max_depth=10, random_state=42
        ),
        'logreg_balanced': LogisticRegression(
            class_weight=class_weights, random_state=42, max_iter=1000
        )
    }

    # Кросс-валидация
    tscv = TimeSeriesSplit(n_splits=3)
    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Для многоклассовой классификации используем accuracy
        scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'model': pipeline
        }
        print(f"{name}: Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

    # Выбираем лучшую модель
    best_name = max(results.keys(), key=lambda x: results[x]['mean_accuracy'])
    best_model = results[best_name]['model']
    best_score = results[best_name]['mean_accuracy']

    print(f"\n🎯 BEST MODEL: {best_name} with Accuracy = {best_score:.4f}")

    # Дообучаем на всех данных
    best_model.fit(X, y)

    # Сохраняем
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "ai_strategy_fixed.pkl")

    # Анализ важности фич
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feat_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\nTOP 10 FEATURES:")
        print(feat_importance.head(10))

    return best_model


if __name__ == "__main__":
    emergency_retrain_fixed()