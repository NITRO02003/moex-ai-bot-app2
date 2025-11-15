
from __future__ import annotations
import os, json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, List, Iterable
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import models as M, data as D, backtest as B, regime_backtest as RB, risk as R, features as F
from .paths import REPORTS_DIR

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ===================== utils =====================

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if (b not in (0, 0.0)) else default

def _pf(gross_p: float, gross_l: float) -> float:
    return (gross_p / abs(gross_l)) if gross_l < 0 else (float("inf") if gross_p > 0 else 0.0)

def _sr_proxy(pnl_series: pd.Series) -> float:
    x = pnl_series.dropna().values
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return mu / sd if sd > 0 else 0.0

def _sortino_proxy(pnl_series: pd.Series) -> float:
    x = pnl_series.dropna().values
    neg = x[x < 0]
    dd = float(np.sqrt(np.mean(neg**2))) if neg.size else 0.0
    mu = float(np.mean(x)) if x.size else 0.0
    return mu / dd if dd > 0 else 0.0

def _streak_lengths(bools: np.ndarray, value: bool) -> List[int]:
    lengths: List[int] = []
    cur = 0
    for v in bools:
        if bool(v) == value:
            cur += 1
        else:
            if cur > 0: lengths.append(cur)
            cur = 0
    if cur > 0: lengths.append(cur)
    return lengths

def _exposure_stats(side: pd.Series) -> Dict[str, float]:
    nz = (side != 0).astype(int)
    pos = (side > 0).astype(int)
    neg = (side < 0).astype(int)
    tot = int(len(side)) if len(side) else 1
    return dict(
        exposure_total=_safe_div(float(nz.sum()), tot),
        exposure_long=_safe_div(float(pos.sum()), tot),
        exposure_short=_safe_div(float(neg.sum()), tot),
    )

def _holding_stats(side: pd.Series) -> Dict[str, float]:
    nonzero = (side != 0).to_numpy()
    streaks = _streak_lengths(nonzero, True)
    avg_hold = float(np.mean(streaks)) if streaks else 0.0
    max_hold = float(np.max(streaks)) if streaks else 0.0
    return dict(avg_hold_bars=avg_hold, max_hold_bars=max_hold)

def _trade_stats(pnl: pd.Series) -> Dict[str, float]:
    tr = pnl[pnl != 0.0]
    wins = tr[tr > 0]; losses = tr[tr < 0]
    hit = _safe_div(float((tr > 0).sum()), int(len(tr))) if len(tr) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    med = float(tr.median()) if len(tr) else 0.0

    signs = np.sign(tr.to_numpy()) if len(tr) else np.array([])
    win_streaks = _streak_lengths(signs > 0, True) if len(signs) else []
    loss_streaks = _streak_lengths(signs < 0, True) if len(signs) else []
    max_ws = int(max(win_streaks) if win_streaks else 0)
    max_ls = int(max(loss_streaks) if loss_streaks else 0)
    return dict(
        hit_rate=hit, avg_win=avg_win, avg_loss=avg_loss,
        median_trade=med, max_win_streak=max_ws, max_loss_streak=max_ls
    )

def _moments(x: np.ndarray) -> Tuple[float, float]:
    if x.size < 2:
        return 0.0, 0.0
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    if std == 0:
        return 0.0, 0.0
    skew = float(np.mean(((x - mean) / std) ** 3))
    kurt = float(np.mean(((x - mean) / std) ** 4) - 3.0)
    return skew, kurt

def _calmar_proxy(total_return: float, max_dd: float) -> float:
    dd = abs(max_dd)
    return (total_return / dd) if dd > 0 else (float("inf") if total_return > 0 else 0.0)

def _bt_inline_metrics(close: pd.Series, side: pd.Series, size: pd.Series,
                       commission: float = 0.0005, slippage_bps: float = 1.0,
                       equity0: float = 1_000_000.0) -> Dict[str, Any]:
    px = close.astype(float).reindex(side.index).ffill()
    pnl = pd.Series(0.0, index=side.index)
    shares = 0.0
    commission_paid = 0.0
    turnover = 0.0
    trades = 0

    for ts in side.index:
        p = float(px.loc[ts])
        desired = float(size.loc[ts]) * float(side.loc[ts]) / max(p, 1e-9)
        d = desired - shares
        if d != 0.0:
            turn = abs(d) * p
            cost = turn * commission + turn * (slippage_bps / 10_000.0)
            pnl.loc[ts] += (-d) * p - cost
            commission_paid += cost
            turnover += turn
            trades += 1
            shares = desired

    equity = pnl.cumsum() + equity0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    calmar = _calmar_proxy(total_return, max_dd)
    gross_p = float(pnl[pnl > 0].sum())
    gross_l = float(pnl[pnl < 0].sum())
    pf = _pf(gross_p, gross_l)
    avg_trade = float(pnl[pnl != 0].mean()) if (pnl != 0).any() else 0.0
    profit_per_trade = float(pnl.sum() / trades) if trades > 0 else 0.0
    bars = int(len(equity))
    profit_per_bar = float(pnl.sum() / bars) if bars > 0 else 0.0
    sr = _sr_proxy(pnl)
    sortino = _sortino_proxy(pnl)
    skew, kurt = _moments(pnl.values)

    expos = _exposure_stats(side)
    hold = _holding_stats(side)
    tstats = _trade_stats(pnl)

    return dict(
        final_equity=float(equity.iloc[-1]) if len(equity) else equity0,
        total_return=total_return,
        max_drawdown=max_dd,
        calmar_proxy=calmar,
        bars=bars,
        trade_count=int(trades),
        turnover=float(turnover),
        commission_paid=float(commission_paid),
        gross_profit=gross_p,
        gross_loss=gross_l,
        profit_factor=pf,
        avg_trade=avg_trade,
        median_trade=tstats["median_trade"],
        profit_per_trade=profit_per_trade,
        profit_per_bar=profit_per_bar,
        sharpe_proxy=sr,
        sortino_proxy=sortino,
        skewness=skew,
        kurtosis=kurt,
        hit_rate=tstats["hit_rate"],
        avg_win=tstats["avg_win"],
        avg_loss=tstats["avg_loss"],
        max_win_streak=tstats["max_win_streak"],
        max_loss_streak=tstats["max_loss_streak"],
        **expos,
        **hold,
    )

# ===================== calibration (A1) =====================

def _calibrate_asym_profit(symbol: str, horizon: int = 1) -> Tuple[float, float]:
    df = D.load_csv(symbol)
    if df is None or df.empty or "close" not in df.columns:
        return 0.58, 0.42
    bundle = M.load()
    X, y = M.dataset(df, horizon=horizon)
    if len(X) == 0:
        return 0.58, 0.42
    X = X[F.final_columns(X.columns)]
    close = df["close"].astype(float).reindex(X.index).ffill().bfill()
    p = M.predict_proba(bundle, X, close)
    ret = close.pct_change().reindex(p.index).fillna(0.0).values

    best = (0.58, 0.42); best_score = -1e18
    for tl in np.round(np.arange(0.52, 0.701, 0.01), 2):
        for ts in np.round(np.arange(0.30, 0.481, 0.01), 2):
            side = np.where(p.values >= tl, 1, np.where(p.values <= ts, -1, 0))
            score = float((side * ret).mean())
            if score > best_score:
                best_score = score; best = (float(tl), float(ts))
    return best

# ===================== workers =====================

def _bt_ptr_worker(args: Tuple[str, int, float, float, float]) -> Dict[str, Any]:
    symbol, horizon, ptr, thL, thS = args
    df = D.load_csv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "kind": "backtest", "param": "per_trade_risk", "value": ptr, "error": "no_data"}
    bundle = M.load()
    rp = R.RiskParams(per_trade_risk=ptr)
    from .strategy import signal_and_size
    side, size = signal_and_size(df, bundle, rp, 1_000_000.0, th_long=thL, th_short=thS, use_filters=True)
    met = _bt_inline_metrics(df["close"], side, size)
    met.update(dict(kind="backtest", symbol=symbol, param="per_trade_risk", value=ptr, th_long=thL, th_short=thS))
    return met

def _bt_volq_worker(args: Tuple[str, int, float, float, float, float]) -> Dict[str, Any]:
    symbol, horizon, q, ptr, thL, thS = args
    df = D.load_csv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "kind": "backtest", "param": "vol_q", "value": q, "error": "no_data"}
    bundle = M.load()
    from .strategy import signal_and_size
    rp = R.RiskParams(per_trade_risk=ptr)
    side, size = signal_and_size(df, bundle, rp, 1_000_000.0, th_long=thL, th_short=thS, use_filters=False)
    close = df["close"].astype(float).reindex(side.index).ffill()
    vol = R.realized_vol(close, 48).reindex(side.index).ffill().bfill()
    mask = vol >= vol.rolling(200).quantile(q).reindex(side.index).ffill().bfill()
    met = _bt_inline_metrics(close, side.where(mask, 0), size.where(mask, 0.0))
    met.update(dict(kind="backtest", symbol=symbol, param="vol_q", value=q, th_long=thL, th_short=thS, per_trade_risk=ptr))
    return met

def _regime_min_gap_worker(args: Tuple[str, int, int, float, float]) -> Dict[str, Any]:
    symbol, horizon, mg, thL, thS = args
    df = D.load_csv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "kind": "regime", "param": "min_gap", "value": mg, "error": "no_data"}
    bundle = M.load()
    params = RB.RegimeBtParams(min_gap=mg)
    res = RB.run_symbol(df, bundle, R.RiskParams(per_trade_risk=0.002),
                        params, equity0=1_000_000.0, th_long=thL, th_short=thS)
    out = res["metrics"].copy()
    out.update(dict(kind="regime", symbol=symbol, param="min_gap", value=mg, th_long=thL, th_short=thS))
    return out

def _regime_combo_worker(args: Tuple[str, int, float, float, float, int, float, float]) -> Dict[str, Any]:
    symbol, horizon, tp, sl, tr, mh, thL, thS = args
    df = D.load_csv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "kind": "regime", "param": "tp_sl_tr_mh", "value": f"{tp}|{sl}|{tr}|{mh}", "error": "no_data"}
    bundle = M.load()
    params = RB.RegimeBtParams(tp_mult=tp, sl_mult=sl, trail_mult=tr, max_hold=mh)
    res = RB.run_symbol(df, bundle, R.RiskParams(per_trade_risk=0.002),
                        params, equity0=1_000_000.0, th_long=thL, th_short=thS)
    out = res["metrics"].copy()
    out.update(dict(kind="regime", symbol=symbol, param="tp_sl_tr_mh", value=f"{tp}|{sl}|{tr}|{mh}", th_long=thL, th_short=thS))
    return out

def _bt_thresholds_worker(args: Tuple[str, int, float, float]) -> Dict[str, Any]:
    symbol, horizon, thL, thS = args
    df = D.load_csv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "kind": "backtest", "param": "thresh_sweep", "value": f"{thL}|{thS}", "error": "no_data"}
    bundle = M.load()
    rp = R.RiskParams(per_trade_risk=0.002)
    from .strategy import signal_and_size
    side, size = signal_and_size(df, bundle, rp, 1_000_000.0, th_long=thL, th_short=thS, use_filters=True)
    met = _bt_inline_metrics(df["close"], side, size)
    met.update(dict(kind="backtest", symbol=symbol, param="thresh", value=f"{thL}|{thS}"))
    return met

# ===================== top-K selection =====================

def _score_row(row: pd.Series) -> float:
    """Скоринг варианта параметров для свипа.

    Базовая часть: рет / DD + Sharpe/Sortino + profit factor + profit_per_trade.
    Дополнительно штрафуем за избыточное количество сделок и огромный оборот.
    """
    ret = float(row.get("total_return", 0.0))
    dd = abs(float(row.get("max_drawdown", 0.0))) + 1e-12
    sr = float(row.get("sharpe_proxy", 0.0))
    pf = float(row.get("profit_factor", 0.0))
    ppt = float(row.get("profit_per_trade", 0.0))
    sortino = float(row.get("sortino_proxy", 0.0))
    trades = float(row.get("trade_count", 0.0))
    turnover = float(row.get("turnover", 0.0))

    base = (ret / dd) + 0.5 * sr + 0.25 * pf + 50.0 * ppt + 0.5 * sortino

    # штраф за избыточное количество сделок и оборот (нормируем на equity0=1e6)
    trade_penalty = 0.0005 * trades
    turn_penalty = 0.005 * (turnover / 1_000_000.0)

    return base - trade_penalty - turn_penalty


def select_top_k(csv_path: str | Path, k: int = 5, group_by: List[str] | None = None,
                 out_best_json: str | Path | None = None) -> str:
    df = pd.read_csv(csv_path)
    if group_by is None:
        group_by = ["symbol", "kind", "param"]
    if "score" not in df.columns:
        df["score"] = df.apply(_score_row, axis=1)
    best = (
        df.sort_values("score", ascending=False)
          .groupby(group_by, as_index=False)
          .head(k)
          .reset_index(drop=True)
    )
    best_path = Path(out_best_json) if out_best_json else (REPORTS_DIR / "sweep_top.json")
    out: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for _, r in best.iterrows():
        sym = str(r.get("symbol", "NA")); kind = str(r.get("kind", "NA")); param = str(r.get("param","NA"))
        out.setdefault(sym, {}).setdefault(kind, {}).setdefault(param, []).append({k: (None if pd.isna(v) else v) for k,v in r.items()})
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": _now_iso(), "top_k": k, "group_by": group_by, "items": out}, f, ensure_ascii=False, indent=2)
    return str(best_path)

# ===================== config & runner =====================

@dataclass
class SweepConfig:
    symbols: List[str]
    horizon: int = 1
    per_trade_risk_grid: List[float] = (0.0005, 0.001, 0.0015, 0.002)
    vol_q_grid: List[float] = (0.1, 0.2, 0.3)
    min_gap_grid: List[int] = (8, 12, 16)
    tp_grid: List[float] = (1.5, 2.0, 2.5)
    sl_grid: List[float] = (0.7, 1.0, 1.2)
    trail_grid: List[float] = (0.0, 1.0, 1.5)
    max_hold_grid: List[int] = (80, 120, 160)
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)
    do_combined: bool = True
    thresh_window: float = 0.03

def _gen_threshold_pairs(calib: Tuple[float,float], window: float = 0.03, step: float = 0.01) -> Iterable[Tuple[float,float]]:
    tl0, ts0 = calib
    tl_list = np.round(np.arange(tl0 - window, tl0 + window + 1e-9, step), 2)
    ts_list = np.round(np.arange(ts0 - window, ts0 + window + 1e-9, step), 2)
    for tl in tl_list:
        for ts in ts_list:
            yield float(tl), float(ts)

def run_single_param_sweeps(cfg: SweepConfig, out_csv: str | None = None,
                            top_k: int | None = None, out_best_json: str | None = None) -> Dict[str, str]:
    calib = {s: _calibrate_asym_profit(s, cfg.horizon) for s in cfg.symbols}

    tasks: List[tuple] = []
    for s in cfg.symbols:
        thL, thS = calib[s]
        for ptr in cfg.per_trade_risk_grid:
            tasks.append(("bt_ptr", (s, cfg.horizon, float(ptr), thL, thS)))
    for s in cfg.symbols:
        thL, thS = calib[s]
        for q in cfg.vol_q_grid:
            tasks.append(("bt_volq", (s, cfg.horizon, float(q), 0.002, thL, thS)))
    for s in cfg.symbols:
        thL, thS = calib[s]
        for mg in cfg.min_gap_grid:
            tasks.append(("rg_mg", (s, cfg.horizon, int(mg), thL, thS)))
    for s in cfg.symbols:
        thL, thS = calib[s]
        for tp in cfg.tp_grid:
            for sl in cfg.sl_grid:
                for tr in cfg.trail_grid:
                    for mh in cfg.max_hold_grid:
                        tasks.append(("rg_combo", (s, cfg.horizon, float(tp), float(sl), float(tr), int(mh), thL, thS)))
    for s in cfg.symbols:
        for thL, thS in _gen_threshold_pairs(calib[s], cfg.thresh_window, 0.01):
            tasks.append(("bt_thresh", (s, cfg.horizon, thL, thS)))

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(cfg.n_jobs))) as ex:
        futs = []
        for kind, a in tasks:
            if kind == "bt_ptr":
                futs.append(ex.submit(_bt_ptr_worker, a))
            elif kind == "bt_volq":
                futs.append(ex.submit(_bt_volq_worker, a))
            elif kind == "rg_mg":
                futs.append(ex.submit(_regime_min_gap_worker, a))
            elif kind == "rg_combo":
                futs.append(ex.submit(_regime_combo_worker, a))
            elif kind == "bt_thresh":
                futs.append(ex.submit(_bt_thresholds_worker, a))
        for fu in as_completed(futs):
            try:
                rows.append(fu.result())
            except Exception as e:
                rows.append({"error": str(e)})

    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", _now_iso())
    out_path = Path(out_csv) if out_csv else (REPORTS_DIR / "sweep_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    best_path = None
    if top_k and top_k > 0:
        best_path = select_top_k(out_path, k=int(top_k), out_best_json=out_best_json)

    return {"csv": str(out_path), "best_json": str(best_path) if best_path else ""}
