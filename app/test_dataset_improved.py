# app/test_dataset_improved.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.features import build_features_10m, FINAL_FEATURE_SET, make_signal_labels, make_regime_labels


def build_dataset_improved():
    """Улучшенная версия построения датасета с лучшей обработкой ошибок"""
    data_dir = Path("data")
    symbols = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN"]

    frames = []
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)

        print(f"Processing {symbol}: {len(df)} rows")

        try:
            # Строим фичи
            feats = build_features_10m(df)
            print(f"  Features shape: {feats.shape}")
            print(f"  NaN in features: {feats.isna().sum().sum()}")

            feats["symbol"] = symbol
            feats["close"] = df["close"]
            frames.append(feats)

        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            continue

    if not frames:
        raise RuntimeError("No data loaded")

    # Объединяем все данные
    ds = pd.concat(frames).sort_index()
    print(f"Combined dataset: {len(ds)} rows")
    print(f"NaN in combined dataset: {ds.isna().sum().sum()}")

    # Создаем фичи и метки
    X = ds[FINAL_FEATURE_SET].copy()

    print("Creating signal labels...")
    y_sig = make_signal_labels(ds["close"])
    print(f"Signal labels - NaN: {y_sig.isna().sum()}")

    print("Creating regime labels...")
    y_reg = make_regime_labels(ds["close"])
    print(f"Regime labels - NaN: {y_reg.isna().sum()}")

    # Проверяем данные перед маскированием
    print(f"X shape: {X.shape}, NaN in X: {X.isna().sum().sum()}")
    print(f"y_sig shape: {y_sig.shape}")
    print(f"y_reg shape: {y_reg.shape}")

    # Убираем строки с NaN
    mask = (~X.isna().any(axis=1)) & (~y_sig.isna()) & (~y_reg.isna())
    print(f"Rows after masking: {mask.sum()}")

    X = X[mask]
    y_sig = y_sig[mask]
    y_reg = y_reg[mask]

    print(f"After cleaning - X: {X.shape}, y_sig: {y_sig.shape}, y_reg: {y_reg.shape}")

    # Проверяем распределения меток
    print(f"Signal labels distribution:\n{y_sig.value_counts()}")
    print(f"Regime labels distribution:\n{y_reg.value_counts()}")

    return X.astype("float32"), y_sig.astype("int8"), y_reg.astype("int8")


def test_dataset_improved():
    print("=== IMPROVED DATASET TEST ===")

    try:
        X, y_sig, y_reg = build_dataset_improved()

        print(f"✅ Dataset created successfully!")
        print(f"Features shape: {X.shape}")
        print(f"Signal labels shape: {y_sig.shape}")
        print(f"Regime labels shape: {y_reg.shape}")

        print(f"\nFeature columns: {list(X.columns)}")

        print(f"\nSample features:")
        print(X.head(3))

        return True

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dataset_improved()