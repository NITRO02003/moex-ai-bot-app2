from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import os
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import REPORTS_DIR, MODELS_DIR
from . import data as D, features as F, models as M, risk as R, regime_backtest as RB
from .feature_sets import FEATURE_SETS

@dataclass
class FeatureSweepConfig:
    symbols: List[str]
    horizon: int = 1
    equity0: float = 1_000_000.0

    # фиксированные regime‑/risk‑параметры для сравнения feature set’ов
    atr_len: int = 14
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    trail_mult: float = 1.0
    min_gap: int = 12
    max_hold: int = 120
    cooldown: int = 10
    vol_lb: int = 48
    z_thr: float = 0.7
    th_long: float = 0.55
    th_short: float = 0.40
    per_trade_risk: float = 0.001

    out_csv: Path = REPORTS_DIR / "feature_sweep_results.csv"
    out_best: Path = REPORTS_DIR / "feature_sweep_top.json"
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)

def _score_row(row: Dict[str, Any]) -> float:
    """Скоринг конфигурации feature set’а.

    Структурно похож на _score_row из run_both, но без жёсткого штрафа за оборот.
    """
    ret = float(row.get("total_return", 0.0))
    dd = abs(float(row.get("max_drawdown", 0.0))) + 1e-12
    pf = float(row.get("profit_factor", 0.0))
    ppt = float(row.get("avg_trade", 0.0))
    trades = float(row.get("trade_count", 0.0))

    base = (ret / dd) + 0.5 * pf + 50.0 * ppt

    # мягкий штраф за чрезмерное число сделок
    trade_penalty = 0.0005 * max(0.0, trades - 200.0)
    return base - trade_penalty

def _featset_worker(args) -> List[Dict[str, Any]]:
    feat_name, feat_cols, cfg = args

    # включаем нужный набор фич
    F.set_active_feature_set(feat_cols)

    # собираем данные по всем символам
    prices_by_symbol: Dict[str, pd.DataFrame] = {}
    for s in cfg.symbols:
        df = D.load_csv(s)
        if df is None or df.empty or "close" not in df.columns:
            continue
        prices_by_symbol[s] = df

    if not prices_by_symbol:
        return []

    # отдельный путь для модели под этот feature set
    model_path = MODELS_DIR / f"ai_strategy_{feat_name}.pkl"

    # train offline multi‑symbol
    M.fit_offline_multi(prices_by_symbol, horizon=cfg.horizon, path=model_path)
    bundle = M.load(model_path)

    rp = R.from_config()
    rp.per_trade_risk = cfg.per_trade_risk

    rows: List[Dict[str, Any]] = []

    for symbol, prices in prices_by_symbol.items():
        params = RB.RegimeBtParams(
            commission=0.0005,
            slippage_bps=2.0,
            atr_len=cfg.atr_len,
            tp_mult=cfg.tp_mult,
            sl_mult=cfg.sl_mult,
            trail_mult=cfg.trail_mult,
            min_gap=cfg.min_gap,
            max_hold=cfg.max_hold,
            cooldown=cfg.cooldown,
            vol_lb=cfg.vol_lb,
            z_thr=cfg.z_thr,
            use_filters=True,
            th_long=cfg.th_long,
            th_short=cfg.th_short,
            per_trade_risk=rp.per_trade_risk,
        )

        res = RB.run_symbol(
            prices,
            bundle,
            rp,
            params,
            equity0=cfg.equity0,
            th_long=cfg.th_long,
            th_short=cfg.th_short,
        )
        m = res.get("metrics", {}) or {}

        rows.append(
            dict(
                symbol=symbol,
                kind="regime",
                param="feature_set",
                value=feat_name,
                total_return=float(m.get("total_return", 0.0)),
                max_drawdown=float(m.get("max_drawdown", 0.0)),
                calmar=float(m.get("calmar", 0.0)),
                profit_factor=float(m.get("profit_factor", 0.0)),
                avg_trade=float(m.get("avg_trade", 0.0)),
                trade_count=int(m.get("trades_est", 0)),
            )
        )

    return rows

def run_feature_sweep(cfg: FeatureSweepConfig) -> Dict[str, Any]:
    """Запускает свип по наборам фич и пишет результаты в CSV/JSON."""
    from .parallel import parallel_map

    tasks = []
    for name, cols in FEATURE_SETS.items():
        tasks.append((name, cols, cfg))

    nested = parallel_map(tasks, _featset_worker, n_jobs=cfg.n_jobs)

    rows: List[Dict[str, Any]] = []
    for chunk in nested:
        rows.extend(chunk)

    if not rows:
        return {"rows": 0, "message": "no results"}

    # сохраняем CSV
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with cfg.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # строим простой топ по (symbol, feature_set)
    by_key: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = f"{r['symbol']}::{r['value']}"
        best = by_key.get(key)
        if best is None or _score_row(r) > _score_row(best):
            by_key[key] = r

    with cfg.out_best.open("w", encoding="utf-8") as f:
        json.dump(by_key, f, ensure_ascii=False, indent=2)

    return {
        "rows": len(rows),
        "csv": str(cfg.out_csv),
        "best_json": str(cfg.out_best),
    }
