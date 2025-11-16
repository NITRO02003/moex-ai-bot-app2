from __future__ import annotations

import os
import itertools
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from .config import load_config
from .utils import load_symbols
from .rule_core import RuleBtParams, run_rule_symbol
from .rule_strategies import MeanRevParams, generate_meanrev_signals


def _iter_param_grid(grid: Dict[str, list]):
    """Перебор всех комбинаций параметров свипа."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _load_prices(sym: str, interval: str = "30min") -> pd.DataFrame | None:
    """
    Загружаем данные по тикеру.

    Приоритет:
      1) processed/{sym}_{interval}.csv
      2) data/{sym}.csv
    """
    candidates = [
        os.path.join("processed", f"{sym}_{interval}.csv"),
        os.path.join("data", f"{sym}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "begin" in df.columns and "datetime" not in df.columns:
                df["datetime"] = pd.to_datetime(df["begin"])
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df
    return None


def _build_meanrev_params(param_set: Dict[str, Any]) -> MeanRevParams:
    """
    Собираем MeanRevParams из словаря параметров.

    Поддерживаем:
      - rsi_len, rsi_low, rsi_high
      - bb_len / bb_k
      - алиасы boll_window -> bb_len, boll_mult -> bb_k
      - min_gap_bars
    """
    p = MeanRevParams()
    if "rsi_len" in param_set:
        p.rsi_len = int(param_set["rsi_len"])
    if "rsi_low" in param_set:
        p.rsi_low = float(param_set["rsi_low"])
    if "rsi_high" in param_set:
        p.rsi_high = float(param_set["rsi_high"])
    if "boll_window" in param_set:
        p.bb_len = int(param_set["boll_window"])
    if "boll_mult" in param_set:
        p.bb_k = float(param_set["boll_mult"])
    if "min_gap_bars" in param_set:
        p.min_gap_bars = int(param_set["min_gap_bars"])
    return p


def _run_meanrev_sweep(
    config: Dict[str, Any],
    symbols: List[str],
    equity0: float,
    interval: str = "30min",
) -> pd.DataFrame:
    """
    Свип для mean-reversion на основе config["sweep"]["MeanRevParams"].
    """
    grid_cfg = config.get("sweep", {}).get("MeanRevParams")
    if not grid_cfg:
        raise ValueError("В config.json нет секции 'sweep.MeanRevParams'")

    print(f"[sweep-meanrev] symbols={symbols}")
    print(f"[sweep-meanrev] grid keys={list(grid_cfg.keys())}")

    rows: list[Dict[str, Any]] = []

    # Риск-параметры берём из defaults.RuleBtParams (пока без свипа)
    bt_defaults = config.get("defaults", {}).get("RuleBtParams", {})
    bt_params = RuleBtParams(**bt_defaults)

    combos = list(_iter_param_grid(grid_cfg))
    print(f"[sweep-meanrev] total combinations={len(combos)}")

    for idx, param_set in enumerate(combos):
        if idx % 50 == 0:
            print(f"[sweep-meanrev] combo {idx}/{len(combos)}: {param_set}")
        s_params = _build_meanrev_params(param_set)

        for sym in symbols:
            df = _load_prices(sym, interval=interval)
            if df is None:
                print(f"[sweep-meanrev] no data for {sym}, skip")
                continue

            # Генерация сигналов mean-reversion
            side = generate_meanrev_signals(df, s_params)

            # Подготовка данных под новый run_rule_symbol
            df2 = df.copy()
            df2["signal"] = side

            # Запуск бэктеста
            res = run_rule_symbol(df2, bt_params, equity0)
            metrics = res.get("metrics", {})

            row = {
                "strategy": "meanrev",
                "symbol": sym,
                **param_set,
                "total_return": metrics.get("total_return", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def run_sweep(
    strategy: str,
    config_path: str,
    csv_path: str,
    symbols: List[str],
    equity0: float = 1_000_000.0,
    use_breakout_in_high_vol: bool = False,
) -> Dict[str, Any]:
    """
    Точка входа для CLI (совместима с текущим cli.py).

    Параметры:
      - strategy
      - config_path
      - csv_path
      - symbols
      - equity0
      - use_breakout_in_high_vol (пока не используется)
    """
    config = load_config(config_path)
    symbols = load_symbols(symbols)

    print(f"[sweep] strategy={strategy}, symbols={symbols}, equity0={equity0}")
    print(f"[sweep] config={config_path}, out={csv_path}")

    if strategy == "meanrev":
        df = _run_meanrev_sweep(config, symbols, equity0, interval="30min")
    else:
        raise NotImplementedError(f"run_sweep: strategy '{strategy}' пока не реализован")

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[sweep] done, rows={len(df)}, saved to {out_path}")
    return {"rows_written": len(df), "csv": str(out_path)}
