# app/retrain_emergency.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
from pathlib import Path

from app.train_ai import build_dataset


def retrain_emergency_models():
    """Экстренное переобучение моделей с разными алгоритмами"""
    print("=== EMERGENCY MODEL RETRAINING ===")

    # Загружаем данные
    X, y_sig, y_reg = build_dataset()
    print(f"Dataset: {X.shape}, signal labels: {y_sig.value_counts()}")

    # Балансируем классы если нужно
    from sklearn.utils import class_weight
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_sig), y=y_sig)
    class_weights = dict(zip(np.unique(y_sig), weights))
    print(f"Class weights: {class_weights}")

    # Разные алгоритмы для тестирования
    models = {
        'rf_balanced': RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight=class_weights, random_state=42, n_jobs=-1
        ),
        'gbm': GradientBoostingClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'logreg': LogisticRegression(
            class_weight=class_weights, random_state=42, max_iter=1000
        ),
        'svm': SVC(
            class_weight=class_weights, random_state=42, probability=True
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

        # Оценка качества
        scores = cross_val_score(pipeline, X, y_sig, cv=tscv, scoring='roc_auc')
        results[name] = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'model': pipeline
        }
        print(f"{name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    # Выбираем лучшую модель
    best_name = max(results.keys(), key=lambda x: results[x]['mean_auc'])
    best_model = results[best_name]['model']
    best_score = results[best_name]['mean_auc']

    print(f"\n🎯 BEST MODEL: {best_name} with AUC = {best_score:.4f}")

    # Дообучаем на всех данных
    best_model.fit(X, y_sig)

    # Сохраняем
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "ai_strategy_emergency.pkl")

    # Быстрая проверка
    y_pred = best_model.predict_proba(X)[:, 1]
    print(f"Training predictions - mean: {y_pred.mean():.3f}, std: {y_pred.std():.3f}")

    return best_model


if __name__ == "__main__":
    retrain_emergency_models()