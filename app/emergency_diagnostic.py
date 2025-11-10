# app/emergency_diagnostic.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix


def emergency_diagnostic():
    """Срочная диагностика проблем с AI моделями"""
    print("=== EMERGENCY AI DIAGNOSTIC ===")

    # Проверяем модели
    models_dir = Path("models")
    out_dir = Path("out")

    # 1. Проверка существования моделей
    model_files = {
        "strategy": "ai_strategy.pkl",
        "strategy_optimized": "ai_strategy_optimized.pkl",
        "risk": "ai_risk.pkl",
        "risk_optimized": "ai_risk_optimized.pkl"
    }

    print("1. MODEL FILES CHECK:")
    for name, file in model_files.items():
        path = models_dir / file
        exists = path.exists()
        print(f"   {name}: {'✅' if exists else '❌'} {file}")
        if exists:
            try:
                model = joblib.load(path)
                print(f"      Type: {type(model)}")
                if hasattr(model, 'steps'):
                    print(f"      Steps: {[type(s[1]).__name__ for s in model.steps]}")
            except Exception as e:
                print(f"      Error loading: {e}")

    # 2. Проверка метрик обучения
    metrics_path = out_dir / "train_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"\n2. TRAINING METRICS:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
    else:
        print("\n2. TRAINING METRICS: ❌ No metrics file")

    # 3. Проверка данных для обучения
    try:
        from app.train_ai import build_dataset
        X, y_sig, y_reg = build_dataset()
        print(f"\n3. DATASET ANALYSIS:")
        print(f"   Features shape: {X.shape}")
        print(f"   Signal labels: {pd.Series(y_sig).value_counts().to_dict()}")
        print(f"   Regime labels: {pd.Series(y_reg).value_counts().to_dict()}")

        # Проверка распределения фич
        print(f"   Feature stats:")
        for col in X.columns[:5]:  # Первые 5 фич
            print(f"     {col}: mean={X[col].mean():.4f}, std={X[col].std():.4f}")

    except Exception as e:
        print(f"\n3. DATASET ANALYSIS: ❌ Error - {e}")

    # 4. Тестирование предсказаний
    print(f"\n4. PREDICTION TEST:")
    try:
        from app.ai_models import AIStrategyModel
        model = AIStrategyModel()
        if model.available():
            # Создаем тестовые данные
            test_features = pd.DataFrame({
                'price_efficiency': [0.1, -0.1, 0.5, -0.5],
                'volume_anomaly': [1.0, -1.0, 2.0, -2.0],
                'trend_strength': [0.5, -0.5, 1.0, -1.0],
                'momentum_5': [0.01, -0.01, 0.02, -0.02]
            })

            predictions = model.predict_series(test_features)
            print(f"   Test predictions: {predictions.tolist()}")
            print(f"   Unique predictions: {predictions.unique()}")
        else:
            print("   ❌ Model not available")
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")

    # 5. Рекомендации
    print(f"\n5. EMERGENCY RECOMMENDATIONS:")
    recommendations = [
        "🚨 ПЕРЕОБУЧИТЬ МОДЕЛИ с другими параметрами",
        "🚨 УВЕЛИЧИТЬ ОБЪЕМ ДАННЫХ для обучения",
        "🚨 ПРОВЕРИТЬ КАЧЕСТВО МЕТОК в features.py",
        "🚨 ИСПОЛЬЗОВАТЬ ПРОСТЫЕ СТРАТЕГИИ как fallback",
        "🚨 ПРОВЕРИТЬ РАСПРЕДЕЛЕНИЕ ФИЧ на нормальность"
    ]

    for rec in recommendations:
        print(f"   {rec}")


if __name__ == "__main__":
    emergency_diagnostic()