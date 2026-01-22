from __future__ import annotations

import json
import os
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from ..range_v3 import RangeV3Params
from .backtest import _list_available_symbols, _load_ohlcv, _load_range_config, _run_trades_from_signals
from .engine import run_core_for_symbol
from .portfolio import build_portfolio_stats


def _default_grid() -> Dict[str, List[Any]]:
    return {
        "min_confirmations": [1, 2, 3],
        "lock_bars_after_breakout": [0, 10, 20],
        "deadzone_min_atr_pct": [0.0, 0.002, 0.004],
        "entry_zone_alpha": [0.1, 0.2, 0.3],
    }


def _iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(overrides)
    return out


def run_sweep(
    symbols: List[str],
    interval: str,
    equity0: float,
    config_path: str,
    grid: Dict[str, List[Any]],
    max_combos: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    base_cfg = _load_range_config(config_path)
    results: List[Dict[str, Any]] = []
    combo_count = 0

    for overrides in _iter_grid(grid):
        combo_count += 1
        if max_combos is not None and combo_count > max_combos:
            break

        params_cfg = _apply_overrides(base_cfg, overrides)
        params = RangeV3Params(params_cfg)

        all_trades: List[pd.DataFrame] = []
        per_symbol = []
        for symbol in symbols:
            try:
                df = _load_ohlcv(symbol, interval)
            except FileNotFoundError:
                continue
            sig_df, _debug = run_core_for_symbol(df, params)
            trades, metrics = _run_trades_from_signals(symbol, sig_df, params, equity0)
            per_symbol.append(metrics)
            if trades:
                all_trades.append(pd.DataFrame([t.__dict__ for t in trades]))

        if all_trades:
            all_trades_df = pd.concat(all_trades, ignore_index=True)
            pnls = all_trades_df["pnl"].tolist()
            portfolio = build_portfolio_stats(pnls, equity0, [m["symbol"] for m in per_symbol])
            trades_count = len(all_trades_df)
        else:
            portfolio = build_portfolio_stats([], equity0, [m["symbol"] for m in per_symbol])
            trades_count = 0

        row = {
            **overrides,
            "trades": trades_count,
            "pf": portfolio["pf"],
            "win_rate": portfolio["win_rate"],
            "total_pnl": portfolio["total_pnl"],
            "total_return": portfolio["total_return"],
            "max_drawdown": portfolio["max_drawdown"],
        }
        results.append(row)

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values(["pf", "win_rate", "trades"], ascending=[False, False, False])
    summary = {
        "symbols": symbols,
        "interval": interval,
        "equity0": equity0,
        "grid_keys": list(grid.keys()),
        "total_combos": combo_count if max_combos is None else min(combo_count, max_combos),
    }
    return df_out, summary


def main(args):
    symbols = list(args.symbols)
    if len(symbols) == 1 and symbols[0].lower() == "all":
        symbols = _list_available_symbols(args.interval)

    grid = _default_grid()
    if args.grid:
        with open(args.grid, "r", encoding="utf-8") as f:
            grid = json.load(f)

    df_out, summary = run_sweep(
        symbols=symbols,
        interval=args.interval,
        equity0=float(args.equity0),
        config_path=args.config_range,
        grid=grid,
        max_combos=args.max_combos,
    )

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(args.out, index=False)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[range-core-sweep] saved results to {args.out}")
    print(f"[range-core-sweep] saved summary to {summary_path}")
    return df_out
