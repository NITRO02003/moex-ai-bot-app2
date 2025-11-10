# app/emergency_optimization.py
import subprocess
import sys
from pathlib import Path


def run_emergency_optimization():
    """Запуск всех экстренных оптимизаций"""
    print("🚨 LAUNCHING EMERGENCY OPTIMIZATION PROTOCOL 🚨")
    print("=" * 60)

    steps = [
        ("DIAGNOSTIC", "app.emergency_diagnostic"),
        ("DATA QUALITY", "app.data_optimizer"),
        ("MODEL RETRAINING", "app.retrain_emergency"),
        ("TECHNICAL FALLBACK", "app.technical_fallback"),
    ]

    for step_name, module in steps:
        print(f"\n🎯 STEP: {step_name}")
        print("-" * 40)

        try:
            subprocess.run([sys.executable, "-m", module], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in {step_name}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in {step_name}: {e}")

    print("\n" + "=" * 60)
    print("🚨 EMERGENCY OPTIMIZATION COMPLETE 🚨")
    print("\nNEXT STEPS:")
    print("1. Check diagnostic results above")
    print("2. If AI models improved, run: python -m app.final_optimized")
    print("3. If not, focus on technical strategies")
    print("4. Consider collecting more training data")


if __name__ == "__main__":
    run_emergency_optimization()