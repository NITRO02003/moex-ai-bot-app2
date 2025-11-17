from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .config import load_config
from .utils import load_symbols
from .rule_core import RuleBtParams, run_rule_symbol
from .rule_strategies import (
    TrendParams,
    MeanRevParams,
    BreakoutParams,
    generate_trend_signals,
    generate_meanrev_signals,
    generate_breakout_signals,
)


def _load_prices(sym: str, interval: str = "30min") -> Optional[pd.DataFrame]:
    """
    Загружает данные по тикеру.

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
            # стандартизируем временную колонку
            if "begin" in df.columns and "datetime" not in df.columns:
                df["datetime"] = pd.to_datetime(df["begin"])
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            else:
                # если нет явной временной колонки — проблема данных
                print(f"[analyze-trades] {sym}: no datetime/begin column in {path}, skip")
                return None
            return df
    print(f"[analyze-trades] {sym}: no data file found")
    return None


def _build_trend_params(cfg: Dict[str, Any]) -> TrendParams:
    p = TrendParams()
    if "ema_fast" in cfg:
        p.ema_fast = int(cfg["ema_fast"])
    if "ema_slow" in cfg:
        p.ema_slow = int(cfg["ema_slow"])
    if "atr_len" in cfg:
        p.atr_len = int(cfg["atr_len"])
    if "trend_thr" in cfg:
        p.trend_thr = float(cfg["trend_thr"])
    if "min_gap_bars" in cfg:
        p.min_gap_bars = int(cfg["min_gap_bars"])
    return p


def _build_meanrev_params(cfg: Dict[str, Any]) -> MeanRevParams:
    p = MeanRevParams()
    if "rsi_len" in cfg:
        p.rsi_len = int(cfg["rsi_len"])
    if "rsi_low" in cfg:
        p.rsi_low = float(cfg["rsi_low"])
    if "rsi_high" in cfg:
        p.rsi_high = float(cfg["rsi_high"])
    if "boll_window" in cfg:
        p.bb_len = int(cfg["boll_window"])
    if "boll_mult" in cfg:
        p.bb_k = float(cfg["boll_mult"])
    if "min_gap_bars" in cfg:
        p.min_gap_bars = int(cfg["min_gap_bars"])
    return p


def _build_breakout_params(cfg: Dict[str, Any]) -> BreakoutParams:
    p = BreakoutParams()
    if "channel_len" in cfg:
        p.channel_len = int(cfg["channel_len"])
    if "confirm_bars" in cfg:
        p.confirm_bars = int(cfg["confirm_bars"])
    if "min_gap_bars" in cfg:
        p.min_gap_bars = int(cfg["min_gap_bars"])
    return p


def run_analyze_trades(
    strategy: str,
    symbols: List[str],
    interval: str,
    equity0: float,
    config_path: str,
    out_prefix: str,
) -> Dict[str, Any]:
    """
    Генерирует bar- и trade-логи для заданной стратегии и тикеров.

    strategy: 'trend' | 'meanrev' | 'breakout'
    symbols: список тикеров (или ['all'])
    interval: таймфрейм ('30min', '1h', ...)
    equity0: стартовый капитал
    config_path: путь к config.json
    out_prefix: префикс имени выходных файлов, например 'out/diag_meanrev'

    Результат:
      - на диск пишутся файлы:
        out_prefix_<SYMBOL>_<strategy>_<interval>_bars.csv
        out_prefix_<SYMBOL>_<strategy>_<interval>_trades.csv

      - возвращается словарь с краткой сводкой по тикерам.
    """
    config = load_config(config_path)
    symbols = load_symbols(symbols)

    defaults = config.get("defaults", {})
    bt_cfg = defaults.get("RuleBtParams", {})
    bt_params = RuleBtParams(**bt_cfg)

    summary: Dict[str, Any] = {}

    print(
        f"[analyze-trades] start, strategy={strategy}, symbols={symbols}, "
        f"interval={interval}, equity0={equity0}, config={config_path}"
    )

    out_path = Path(out_prefix)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        df = _load_prices(sym, interval=interval)
        if df is None:
            continue

        # генерируем сигналы
        if strategy == "trend":
            strat_cfg = defaults.get("TrendParams", {})
            s_params = _build_trend_params(strat_cfg)
            side = generate_trend_signals(df, s_params)
        elif strategy == "meanrev":
            strat_cfg = defaults.get("MeanRevParams", {})
            s_params = _build_meanrev_params(strat_cfg)
            side = generate_meanrev_signals(df, s_params)
        elif strategy == "breakout":
            strat_cfg = defaults.get("BreakoutParams", {})
            s_params = _build_breakout_params(strat_cfg)
            side = generate_breakout_signals(df, s_params)
        else:
            raise NotImplementedError(
                f"run_analyze_trades: strategy '{strategy}' is not supported yet"
            )

        df2 = df.copy()
        df2["signal"] = side

        res = run_rule_symbol(
            df2,
            bt_params,
            equity0=equity0,
            collect_bar_stats=True,
            collect_trades=True,
        )

        bar_stats = res.get("bar_stats")
        trades = res.get("trades")
        metrics = res.get("metrics", {})

        if bar_stats is None or trades is None:
            print(f"[analyze-trades] {sym}: no bar_stats or trades returned, skip")
            continue

        bars_file = out_dir / f"{out_path.name}_{sym}_{strategy}_{interval}_bars.csv"
        trades_file = out_dir / f"{out_path.name}_{sym}_{strategy}_{interval}_trades.csv"

        bar_stats.to_csv(bars_file, index=False)
        trades.to_csv(trades_file, index=False)

        print(
            f"[analyze-trades] {sym}: bars={len(bar_stats)}, trades={len(trades)}, "
            f"metrics_total_return={metrics.get('total_return', 0):.4f}, "
            f"saved to {bars_file.name}, {trades_file.name}"
        )

        summary[sym] = {
            "bars": len(bar_stats),
            "trades": len(trades),
            "metrics": metrics,
            "bars_file": str(bars_file),
            "trades_file": str(trades_file),
        }

    print(f"[analyze-trades] done, symbols_processed={len(summary)}")
    return summary
