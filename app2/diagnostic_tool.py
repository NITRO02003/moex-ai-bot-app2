# app2/diagnostic_tool.py
from app2 import data as D, models as M, features as F
import pandas as pd


def run_comprehensive_diagnostics(symbols):
    for symbol in symbols:
        print(f"\n🔍 COMPREHENSIVE DIAGNOSTICS FOR {symbol}")
        prices = D.load_csv(symbol)

        if prices.empty:
            print("   No data")
            continue

        print(f"   Data range: {prices.index.min()} to {prices.index.max()}")
        print(f"   Bars: {len(prices)}")

        # Проверяем фичи
        X = F.build(prices)
        print(f"   Features: {len(X.columns)} columns, {len(X)} rows")

        # Проверяем предсказания
        bundle = M.load()
        p = M.predict_proba(bundle, X, prices['close'].astype(float))

        print(f"   Predictions analysis:")
        print(f"      Min: {p.min():.3f}, Max: {p.max():.3f}, Mean: {p.mean():.3f}")
        for threshold in [0.5, 0.51, 0.55, 0.6]:
            count = (p > threshold).sum()
            print(f"      >{threshold}: {count} signals ({count / len(p):.1%})")
        