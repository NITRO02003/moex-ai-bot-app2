# app2/RUN.py

from app2.feature_sweep import FeatureSweepConfig, run_feature_sweep
import multiprocessing


def main() -> None:
    cfg = FeatureSweepConfig(
        symbols=["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "YNDX"],
        horizon=1,
        # можешь сразу задать меньше потоков, если хочешь
        n_jobs=6,
    )
    res = run_feature_sweep(cfg)
    print(res)


if __name__ == "__main__":
    # для Windows и/или когда скрипт "замораживается" в exe
    multiprocessing.freeze_support()
    main()