from __future__ import annotations

import os
import itertools
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

import pandas as pd
import concurrent.futures as cf

from .config import load_config
from .utils import load_symbols
from .rule_core import RuleBtParams, run_rule_symbol
from .rule_strategies import MeanRevParams, generate_meanrev_signals


# ---------- генерация сетки параметров ----------


def _iter_param_grid(grid: Dict[str, Iterable]) -> Iterable[Dict[str, Any]]:
    """Перебор всех комбинаций параметров свипа."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ---------- загрузка данных ----------


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
            # Стандартизируем колонку времени:
            if "begin" in df.columns and "datetime" not in df.columns:
                df["datetime"] = pd.to_datetime(df["begin"])
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            else:
                # если нет явной колонки времени — считаем, что это проблема данных
                print(f"[sweep-meanrev] {sym}: no datetime/begin column in {path}, skip")
                return None
            return df
    print(f"[sweep-meanrev] {sym}: no data file found")
    return None


# ---------- сборка параметров стратегии ----------


def _build_meanrev_params(param_set: Dict[str, Any]) -> MeanRevParams:
    """
    Собираем MeanRevParams из словаря параметров.

    Поддерживаем:
      - rsi_len, rsi_low, rsi_high
      - boll_window -> bb_len
      - boll_mult   -> bb_k
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


# ---------- worker для одного тикера ----------


def _eval_meanrev_for_symbol(
    args: Tuple[str, List[Dict[str, Any]], float, str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Рабочая функция, обрабатывающая ОДИН тикер по ВСЕМ комбинациям параметров.
    Вызывается либо в отдельном процессе, либо в основном потоке.
    """
    sym, combos, equity0, interval, bt_params_dict = args
    rows: List[Dict[str, Any]] = []

    df = _load_prices(sym, interval=interval)
    if df is None:
        print(f"[sweep-meanrev] {sym}: no data, skip symbol")
        return rows

    bt_params = RuleBtParams(**bt_params_dict)

    print(f"[sweep-meanrev] {sym}: start, combos={len(combos)}, rows={len(df)}")

    for idx, param_set in enumerate(combos):
        # можно логировать прогресс по конкретному символу раз в N комбинаций
        if idx and idx % 200 == 0:
            print(f"[sweep-meanrev] {sym}: combo {idx}/{len(combos)}")

        s_params = _build_meanrev_params(param_set)
        side = generate_meanrev_signals(df, s_params)

        df2 = df.copy()
        df2["signal"] = side

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

    print(f"[sweep-meanrev] {sym}: done, rows={len(rows)}")
    return rows


# ---------- публичный API ----------


def run_sweep(
    strategy: str,
    config_path: str,
    csv_path: str,
    symbols: List[str],
    equity0: float = 1_000_000.0,
    use_breakout_in_high_vol: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Точка входа для CLI (совместима с app2.cli).

    Параметры:
      - strategy: пока реализован только 'meanrev'
      - config_path: путь к config.json
      - csv_path: куда сохранять результаты свипа
      - symbols: список тикеров или ['all']
      - equity0: стартовый капитал
      - use_breakout_in_high_vol: зарезервировано для regime-свипа
      - n_jobs: -1 = все ядра, 1 = без multiprocessing, N > 1 = N процессов
    """
    if strategy != "meanrev":
        raise NotImplementedError(
            f"run_sweep: strategy '{strategy}' пока не реализован (есть только 'meanrev')"
        )

    config = load_config(config_path)
    symbols = load_symbols(symbols)

    print(
        f"[sweep] strategy={strategy}, symbols={symbols}, "
        f"equity0={equity0}, n_jobs={n_jobs}"
    )
    print(f"[sweep] config={config_path}, out={csv_path}")

    grid_cfg = config.get("sweep", {}).get("MeanRevParams")
    if not grid_cfg:
        raise ValueError("В config.json нет секции 'sweep.MeanRevParams'")

    print(f"[sweep-meanrev] grid keys={list(grid_cfg.keys())}")

    combos = list(_iter_param_grid(grid_cfg))
    total_combos = len(combos)
    print(f"[sweep-meanrev] total combinations per symbol={total_combos}")

    bt_defaults = config.get("defaults", {}).get("RuleBtParams", {})
    bt_params_dict: Dict[str, Any] = dict(bt_defaults)

    # пока фиксированный интервал для свипа
    interval = "30min"

    # задачи теперь по ТИКЕРАМ, не по комбинациям
    tasks: List[Tuple[str, List[Dict[str, Any]], float, str, Dict[str, Any]]] = [
        (sym, combos, equity0, interval, bt_params_dict) for sym in symbols
    ]

    all_rows: List[Dict[str, Any]] = []

    # ---- однопроцессный режим ----
    if n_jobs == 1:
        print("[sweep] running in single-process mode")
        for sym, combos_, eq0, interval_, bt_params_ in tasks:
            rows = _eval_meanrev_for_symbol((sym, combos_, eq0, interval_, bt_params_))
            all_rows.extend(rows)

    # ---- мультипроцессинг по символам ----
    else:
        max_workers = None
        if n_jobs not in (-1, 0, None):
            max_workers = n_jobs

        print(
            f"[sweep] using ProcessPoolExecutor(max_workers={max_workers}) "
            f"over symbols={len(symbols)}"
        )

        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, rows in enumerate(executor.map(_eval_meanrev_for_symbol, tasks)):
                sym = symbols[idx]
                print(
                    f"[sweep-meanrev] symbol {sym} finished, rows={len(rows)} "
                    f"({idx+1}/{len(symbols)})"
                )
                all_rows.extend(rows)

    if not all_rows:
        print(
            "[sweep-meanrev] WARNING: no rows collected — "
            "возможно, нет данных по тикерам или сетка параметров пустая."
        )

    df = pd.DataFrame(all_rows)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[sweep] done, rows={len(df)}, saved to {out_path}")

    return {"rows_written": len(df), "csv": str(out_path)}
