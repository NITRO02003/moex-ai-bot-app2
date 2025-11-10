# app/optimize_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from .train_ai import build_dataset


def optimize_models():
    print("=== ОПТИМИЗАЦИЯ МОДЕЛЕЙ ===")

    # Загружаем данные
    X, y_sig, y_reg = build_dataset()
    print(f"Данные для оптимизации: {X.shape}")

    # Оптимизация гиперпараметров для сигнальной модели
    print("\nОптимизация сигнальной модели...")

    param_grid_signal = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__class_weight': [{0: 1, 1: 1.5}, {0: 1, 1: 2}]
    }

    pipeline_signal = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    tscv = TimeSeriesSplit(n_splits=3)
    grid_signal = GridSearchCV(
        pipeline_signal, param_grid_signal,
        cv=tscv, scoring='roc_auc', n_jobs=1, verbose=1
    )

    grid_signal.fit(X, y_sig)
    print(f"Лучшие параметры сигнальной модели: {grid_signal.best_params_}")
    print(f"Лучший ROC-AUC: {grid_signal.best_score_:.4f}")

    # Оптимизация риск-модели
    print("\nОптимизация риск-модели...")

    param_grid_risk = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 20],
        'rf__min_samples_split': [2, 5]
    }

    pipeline_risk = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    grid_risk = GridSearchCV(
        pipeline_risk, param_grid_risk,
        cv=tscv, scoring='accuracy', n_jobs=1, verbose=1
    )

    grid_risk.fit(X, y_reg)
    print(f"Лучшие параметры риск-модели: {grid_risk.best_params_}")
    print(f"Лучшая точность: {grid_risk.best_score_:.4f}")

    # Сохраняем оптимизированные модели
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(grid_signal.best_estimator_, models_dir / "ai_strategy_optimized.pkl")
    joblib.dump(grid_risk.best_estimator_, models_dir / "ai_risk_optimized.pkl")

    print(f"\nОптимизированные модели сохранены в {models_dir}")

    return grid_signal.best_estimator_, grid_risk.best_estimator_


if __name__ == "__main__":
    optimize_models()