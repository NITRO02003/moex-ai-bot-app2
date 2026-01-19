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
from .rule_strategies import (
    MeanRevParams,
    MeanRevV2Params,
    generate_meanrev_signals,
    generate_meanrev_v2_signals,
)


# ---------- генерация сетки параметров ----------


def _iter_param_grid(grid: Dict[str, Iterable]) -> Iterable[Dict[str, Any]]:
    """Перебор всех комбинаций параметров свипа."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ---------- загрузка данных ----------


def _load_prices(sym: str, interval: str = "30min") -> pd.DataFrame | None:
    """Загрузка данных по тикеру для свипа.

    Приоритет:
      1) processed/{sym}_{interval}.csv
      2) data/{sym}.csv
    """
    candidates = [
        os.path.join("processed", f"{sym}_{interval}.csv"),
        os.path.join("data", f"{sym}.csv"),
    ]
    path: str | None = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        print(f"[sweep] {sym}: no data file found")
        return None

    df = pd.read_csv(path)

    # выбрасываем бары без цены
    if "close" in df.columns:
        df = df[~df["close"].isna()].copy()

    # стандартизируем колонку времени
    if "begin" in df.columns and "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["begin"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        print(f"[sweep] {sym}: no datetime/begin column in {path}, skip symbol")
        return None

    return df


# ---------- сборка параметров стратегий ----------


def _build_meanrev_params(param_set: Dict[str, Any]) -> MeanRevParams:
    """Собираем MeanRevParams из словаря параметров.

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


def _build_meanrev_v2_params(param_set: Dict[str, Any]) -> MeanRevV2Params:
    """Собираем MeanRevV2Params из словаря параметров."""
    p = MeanRevV2Params()

    if "ma_len" in param_set:
        p.ma_len = int(param_set["ma_len"])
    if "atr_len" in param_set:
        p.atr_len = int(param_set["atr_len"])
    if "z_entry" in param_set:
        p.z_entry = float(param_set["z_entry"])
    if "z_entry_long" in param_set and param_set["z_entry_long"] is not None:
        p.z_entry_long = float(param_set["z_entry_long"])
    if "z_entry_short" in param_set and param_set["z_entry_short"] is not None:
        p.z_entry_short = float(param_set["z_entry_short"])
    if "regime_filter" in param_set and param_set["regime_filter"] is not None:
        p.regime_filter = list(param_set["regime_filter"])

    return p


# ---------- worker'ы для одного тикера ----------


def _eval_meanrev_for_symbol(
    args: Tuple[str, List[Dict[str, Any]], float, str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Обработка ОДНОГО тикера по ВСЕМ комбинациям параметров (meanrev v1)."""
    sym, combos, equity0, interval, bt_params_dict = args
    rows: List[Dict[str, Any]] = []

    df = _load_prices(sym, interval=interval)
    if df is None:
        print(f"[sweep-meanrev] {sym}: no data, skip symbol")
        return rows

    bt_params = RuleBtParams(**bt_params_dict)

    print(f"[sweep-meanrev] {sym}: start, combos={len(combos)}, rows={len(df)}")

    for idx, param_set in enumerate(combos):
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
            "calmar": metrics.get("calmar", 0.0),
            "volatility_ann": metrics.get("volatility_ann", 0.0),
            "sharpe_ann": metrics.get("sharpe_ann", 0.0),
            "pf": metrics.get("pf", 0.0),
            "trade_count": metrics.get("trade_count", 0),
            "win_rate": metrics.get("win_rate", 0.0),
            "avg_trade": metrics.get("avg_trade", 0.0),
            "pnl_sum": metrics.get("pnl_sum", 0.0),
            "pnl_mean": metrics.get("pnl_mean", 0.0),
            "pnl_std": metrics.get("pnl_std", 0.0),
        }
        rows.append(row)

    print(f"[sweep-meanrev] {sym}: done, rows={len(rows)}")
    return rows


def _eval_meanrev_v2_for_symbol(
    args: Tuple[
        str,
        List[Dict[str, Any]],
        float,
        str,
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Dict[str, Any]],
        Dict[str, Any],
    ]
) -> List[Dict[str, Any]]:
    """Обработка ОДНОГО тикера по ВСЕМ комбинациям параметров meanrev_v2 и профилям."""
    (
        sym,
        combos,
        equity0,
        interval,
        bt_params_dict,
        v2_defaults,
        profiles_v2,
        regime_cfg,
    ) = args

    rows: List[Dict[str, Any]] = []

    df = _load_prices(sym, interval=interval)
    if df is None:
        print(f"[sweep-meanrev_v2] {sym}: no data, skip symbol")
        return rows

    bt_params = RuleBtParams(**bt_params_dict)

    print(
        f"[sweep-meanrev_v2] {sym}: start, combos={len(combos)}, "
        f"profiles={list(profiles_v2.keys())}, rows={len(df)}"
    )

    for profile_name, profile_v2 in profiles_v2.items():
        for idx, param_set in enumerate(combos):
            # базовая конфигурация стратегии: defaults + профиль + сетка
            strat_cfg: Dict[str, Any] = dict(v2_defaults or {})
            if profile_v2:
                strat_cfg.update(profile_v2)
            strat_cfg.update(param_set)

            s_params = _build_meanrev_v2_params(strat_cfg)
            side, z_score, regime = generate_meanrev_v2_signals(
                df,
                s_params,
                regime_params=regime_cfg,
            )

            df2 = df.copy()
            df2["signal"] = side

            res = run_rule_symbol(df2, bt_params, equity0)
            metrics = res.get("metrics", {})

            row = {
                "strategy": "meanrev_v2",
                "profile": profile_name,
                "symbol": sym,
                **param_set,
                "total_return": metrics.get("total_return", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "calmar": metrics.get("calmar", 0.0),
                "volatility_ann": metrics.get("volatility_ann", 0.0),
                "sharpe_ann": metrics.get("sharpe_ann", 0.0),
                "pf": metrics.get("pf", 0.0),
                "trade_count": metrics.get("trade_count", 0),
                "win_rate": metrics.get("win_rate", 0.0),
                "avg_trade": metrics.get("avg_trade", 0.0),
                "pnl_sum": metrics.get("pnl_sum", 0.0),
                "pnl_mean": metrics.get("pnl_mean", 0.0),
                "pnl_std": metrics.get("pnl_std", 0.0),
            }
            rows.append(row)

    print(f"[sweep-meanrev_v2] {sym}: done, rows={len(rows)}")
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
    """Точка входа для CLI (совместима с app2.cli).

    Параметры:
      - strategy: 'meanrev' или 'meanrev_v2'
      - config_path: путь к config.json
      - csv_path: куда сохранять результаты свипа
      - symbols: список тикеров или ['all']
      - equity0: стартовый капитал
      - use_breakout_in_high_vol: зарезервировано для regime-свипа
      - n_jobs: -1 = все ядра, 1 = без multiprocessing, N > 1 = N процессов
    """
    config = load_config(config_path)
    symbols = load_symbols(symbols)

    print(
        f"[sweep] strategy={strategy}, symbols={symbols}, "
        f"equity0={equity0}, n_jobs={n_jobs}"
    )
    print(f"[sweep] config={config_path}, out={csv_path}")

    sweep_cfg = config.get("sweep", {}) or {}
    defaults_cfg = config.get("defaults", {}) or {}
    profiles_cfg = config.get("profiles", {}) or {}

    bt_defaults = defaults_cfg.get("RuleBtParams", {}) or {}
    bt_params_dict: Dict[str, Any] = dict(bt_defaults)

    # пока фиксированный интервал для свипа
    interval = "30min"

    all_rows: List[Dict[str, Any]] = []

    if strategy == "meanrev":
        grid_cfg = sweep_cfg.get("MeanRevParams")
        if not grid_cfg:
            raise ValueError("В config.json нет секции 'sweep.MeanRevParams'")

        print(f"[sweep-meanrev] grid keys={list(grid_cfg.keys())}")

        combos = list(_iter_param_grid(grid_cfg))
        total_combos = len(combos)
        print(f"[sweep-meanrev] total combinations per symbol={total_combos}")

        tasks: List[Tuple[str, List[Dict[str, Any]], float, str, Dict[str, Any]]] = [
            (sym, combos, equity0, interval, bt_params_dict) for sym in symbols
        ]

        # однопроцессный режим
        if n_jobs == 1:
            print("[sweep] running in single-process mode")
            for t in tasks:
                rows = _eval_meanrev_for_symbol(t)
                all_rows.extend(rows)
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

    elif strategy == "meanrev_v2":
        grid_cfg = sweep_cfg.get("MeanRevV2Params")
        if not grid_cfg:
            raise ValueError("В config.json нет секции 'sweep.MeanRevV2Params'")

        print(f"[sweep-meanrev_v2] grid keys={list(grid_cfg.keys())}")

        combos = list(_iter_param_grid(grid_cfg))
        total_combos = len(combos)
        print(f"[sweep-meanrev_v2] total combinations per symbol={total_combos}")

        v2_defaults = defaults_cfg.get("MeanRevV2Params", {}) or {}
        regime_cfg = defaults_cfg.get("RegimeParams", {}) or {}

        # берём только те профили, где есть MeanRevV2Params
        profiles_v2: Dict[str, Dict[str, Any]] = {}
        for name, prof in profiles_cfg.items():
            if isinstance(prof, dict) and "MeanRevV2Params" in prof:
                profiles_v2[name] = prof.get("MeanRevV2Params", {}) or {}

        if not profiles_v2:
            print(
                "[sweep-meanrev_v2] WARNING: нет профилей с MeanRevV2Params, "
                "будет использован только defaults.MeanRevV2Params без profile-колонки"
            )
            profiles_v2 = {"default": {}}

        tasks_v2: List[
            Tuple[
                str,
                List[Dict[str, Any]],
                float,
                str,
                Dict[str, Any],
                Dict[str, Any],
                Dict[str, Dict[str, Any]],
                Dict[str, Any],
            ]
        ] = [
            (
                sym,
                combos,
                equity0,
                interval,
                bt_params_dict,
                v2_defaults,
                profiles_v2,
                regime_cfg,
            )
            for sym in symbols
        ]

        if n_jobs == 1:
            print("[sweep] running in single-process mode (meanrev_v2)")
            for t in tasks_v2:
                rows = _eval_meanrev_v2_for_symbol(t)
                all_rows.extend(rows)
        else:
            max_workers = None
            if n_jobs not in (-1, 0, None):
                max_workers = n_jobs

            print(
                f"[sweep] using ProcessPoolExecutor(max_workers={max_workers}) "
                f"over symbols={len(symbols)}"
            )

            with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for idx, rows in enumerate(
                    executor.map(_eval_meanrev_v2_for_symbol, tasks_v2)
                ):
                    sym = symbols[idx]
                    print(
                        f"[sweep-meanrev_v2] symbol {sym} finished, rows={len(rows)} "
                        f"({idx+1}/{len(symbols)})"
                    )
                    all_rows.extend(rows)
    else:
        raise NotImplementedError(
            f"run_sweep: strategy '{strategy}' пока не реализован "
            "(поддерживаются 'meanrev' и 'meanrev_v2')"
        )

    if not all_rows:
        print(
            "[sweep] WARNING: no rows collected — "
            "возможно, нет данных по тикерам или сетка параметров пустая."
        )

    df = pd.DataFrame(all_rows)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[sweep] done, rows={len(df)}, saved to {out_path}")

    return {"rows_written": len(df), "csv": str(out_path)}
