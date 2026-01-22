import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..parallel import parallel_map
from .range_v3 import RangeV3Params, run_range_v3_for_symbol


@dataclass
class Trade:
    symbol: str
    side: int  # +1 / -1
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_rel: float
    bars_held: int


def _load_range_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    range_cfg = cfg.get("RangeV3", {})
    params = dict(range_cfg.get("params", {}))
    profile_name = str(range_cfg.get("risk_profile", "") or "")
    profiles = range_cfg.get("risk_profiles", {})
    if profile_name and isinstance(profiles, dict):
        overrides = profiles.get(profile_name)
        if isinstance(overrides, dict):
            params.update(overrides)
    return params


def _find_data_path(symbol: str, interval: str) -> str:
    """
    Prefer processed/{symbol}_{interval}.csv, otherwise data/{symbol}.csv
    """
    fname_processed = os.path.join("processed", f"{symbol}_{interval}.csv")
    fname_data = os.path.join("data", f"{symbol}.csv")
    if os.path.exists(fname_processed):
        return fname_processed
    if os.path.exists(fname_data):
        return fname_data
    raise FileNotFoundError(f"Cannot find data for {symbol}: tried {fname_processed} and {fname_data}")


def _load_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    path = _find_data_path(symbol, interval)
    df = pd.read_csv(path)

    # try to infer datetime column
    dt_col = None
    for c in df.columns:
        lc = c.lower()
        if "time" in lc or "date" in lc or "dt" in lc:
            dt_col = c
            break
    if dt_col is None:
        for c in df.columns:
            lc = c.lower()
            if lc in ("begin", "end", "timestamp", "ts"):
                dt_col = c
                break
    if dt_col is None:
        dt_col = df.columns[0]

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    df = df.set_index(dt_col)

    # normalize column names
    rename_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("open"):
            rename_map[c] = "open"
        elif lc.startswith("high"):
            rename_map[c] = "high"
        elif lc.startswith("low"):
            rename_map[c] = "low"
        elif lc.startswith("close"):
            rename_map[c] = "close"
        elif lc.startswith("vol"):
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Data for {symbol} missing required column '{col}' in {path}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df



def _run_trades_from_signals(
    symbol: str, df_sig: pd.DataFrame, params: RangeV3Params, equity0: float
) -> Tuple[List[Trade], Dict[str, Any]]:
    trades: List[Trade] = []
    position_side = 0
    entry_idx: Optional[int] = None
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    qty = 0.0
    equity = equity0
    max_equity = equity0
    max_dd = 0.0
    consec_losses = 0

    prices = df_sig["close"]
    if "v3_signal" in df_sig.columns:
        signals = df_sig["v3_signal"]
    else:
        signals = pd.Series(0, index=df_sig.index)
    signals = signals.fillna(0).astype(int)

    for i, (ts, sig) in enumerate(signals.items()):
        price = float(prices.iloc[i])
        if not np.isfinite(price) or price <= 0:
            continue

        if position_side != 0:
            bars_held = i - (entry_idx or 0)
            need_close = False

            # --- price-based SL/TP checks on current bar ---
            if "high" in df_sig.columns:
                bar_high = float(df_sig["high"].iloc[i])
            else:
                bar_high = price
            if "low" in df_sig.columns:
                bar_low = float(df_sig["low"].iloc[i])
            else:
                bar_low = price

            hit_sl = False
            hit_tp = False
            exit_price = price

            if position_side > 0:
                # long: SL ниже, TP выше
                if bar_low <= sl_price < bar_high:
                    hit_sl = True
                    exit_price = sl_price
                elif bar_high >= tp_price > bar_low:
                    hit_tp = True
                    exit_price = tp_price
            else:
                # short: SL выше, TP ниже
                if bar_high >= sl_price > bar_low:
                    hit_sl = True
                    exit_price = sl_price
                elif bar_low <= tp_price < bar_high:
                    hit_tp = True
                    exit_price = tp_price

            if hit_sl or hit_tp:
                need_close = True
            else:
                if sig == -position_side and sig != 0:
                    need_close = True
                if bars_held >= params.max_bars_in_trade:
                    need_close = True

            if need_close:
                pnl = (exit_price - entry_price) * qty * position_side
                pnl_rel = pnl / equity if equity != 0 else 0.0
                equity += pnl
                max_equity = max(max_equity, equity)
                if equity < max_equity and max_equity > 0:
                    dd = (equity - max_equity) / max_equity
                    max_dd = min(max_dd, dd)
                if pnl < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0

                trades.append(
                    Trade(
                        symbol=symbol,
                        side=position_side,
                        entry_time=df_sig.index[entry_idx] if entry_idx is not None else ts,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        pnl=pnl,
                        pnl_rel=pnl_rel,
                        bars_held=bars_held,
                    )
                )
                position_side = 0
                entry_idx = None
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
                qty = 0.0

                if consec_losses >= params.max_consecutive_losses:
                    break
                continue

        # open new position
        if position_side == 0 and sig != 0:
            risk_capital = equity * params.risk_pct_per_trade
            sl_pct = params.sl_pct
            tp_pct = params.tp_pct
            sl_dist = price * sl_pct
            if sl_dist <= 0:
                continue
            qty = risk_capital / sl_dist
            if qty <= 0:
                continue
            position_side = sig
            entry_idx = i
            entry_price = price
            if position_side > 0:
                sl_price = entry_price * (1.0 - sl_pct)
                tp_price = entry_price * (1.0 + tp_pct)
            else:
                sl_price = entry_price * (1.0 + sl_pct)
                tp_price = entry_price * (1.0 - tp_pct)

    pnl_vals = [t.pnl for t in trades]
    wins = [p for p in pnl_vals if p > 0]
    losses = [p for p in pnl_vals if p < 0]
    total_pnl = float(sum(pnl_vals))
    total_return = total_pnl / equity0 if equity0 != 0 else 0.0
    win_rate = len(wins) / len(pnl_vals) if pnl_vals else 0.0
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    metrics = {
        "symbol": symbol,
        "trades": len(trades),
        "total_pnl": total_pnl,
        "total_return": total_return,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "pf": pf,
        "max_drawdown": max_dd,
        "equity0": equity0,
        "equity_final": equity,
    }
    return trades, metrics

def _trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "qty",
                "pnl",
                "pnl_rel",
                "bars_held",
            ]
        )
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "pnl": t.pnl,
                "pnl_rel": t.pnl_rel,
                "bars_held": t.bars_held,
            }
        )
    return pd.DataFrame(rows)


def _run_symbol_task(
    args: tuple[str, str, float, RangeV3Params, str, str],
) -> tuple[str, Dict[str, Any] | None, pd.DataFrame]:
    symbol, interval, equity0, params, out_prefix, tag = args
    try:
        df = _load_ohlcv(symbol, interval)
    except FileNotFoundError:
        return symbol, None, pd.DataFrame()

    sig_df, debug_info = run_range_v3_for_symbol(df, params)
    trades, metrics = _run_trades_from_signals(symbol, sig_df, params, equity0)

    base = f"{out_prefix}_{symbol}_{interval}_{tag}"
    stats_path = f"{base}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    trades_df = _trades_to_df(trades)
    trades_path = f"{base}_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    snaps_path = f"{base}_snapshots.csv"
    sig_df.to_csv(snaps_path)

    debug_path = f"{base}_debug.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)

    print(
        f"[range-v3] {symbol}: trades={len(trades)} "
        f"pf={metrics['pf']:.3f} win_rate={metrics['win_rate']:.3f}"
    )
    return symbol, metrics, trades_df


def main(args):
    symbols: List[str] = list(args.symbols)
    interval: str = args.interval
    equity0: float = float(args.equity0)
    cfg_path: str = args.config_range
    out_prefix: str = args.out_prefix
    tag: str = getattr(args, "tag", "rangeV3")
    n_jobs = getattr(args, "n_jobs", None)

    params_cfg = _load_range_config(cfg_path)
    params = RangeV3Params(params_cfg)

    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    all_symbol_metrics: List[Dict[str, Any]] = []
    all_trades_df_list: List[pd.DataFrame] = []
    tasks = [(symbol, interval, equity0, params, out_prefix, tag) for symbol in symbols]
    for symbol, metrics, trades_df in parallel_map(tasks, _run_symbol_task, n_jobs=n_jobs):
        if metrics is None:
            continue
        all_symbol_metrics.append(metrics)
        if not trades_df.empty:
            all_trades_df_list.append(trades_df)

    # Aggregated portfolio stats
    if all_trades_df_list:
        all_trades_df = pd.concat(all_trades_df_list, ignore_index=True)
        pnl_vals = all_trades_df["pnl"].tolist()
        wins = [p for p in pnl_vals if p > 0]
        losses = [p for p in pnl_vals if p < 0]
        total_pnl = float(sum(pnl_vals))
        win_rate = len(wins) / len(pnl_vals) if pnl_vals else 0.0
        gross_profit = float(sum(wins))
        gross_loss = float(-sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # simple equity curve based on PnL only (relative to portfolio start)
        equity = 0.0
        max_equity = 0.0
        max_dd = 0.0
        for p in pnl_vals:
            equity += p
            max_equity = max(max_equity, equity)
            if equity < max_equity and max_equity > 0:
                dd = (equity - max_equity) / max_equity
                if dd < max_dd:
                    max_dd = dd

        base_all = f"{out_prefix}_ALL_{interval}_{tag}"
        stats_all_path = f"{base_all}_stats.json"
        trades_all_path = f"{base_all}_trades.csv"
        per_symbol_path = f"{base_all}_per_symbol_stats.csv"

        portfolio_stats = {
            "symbols": [m["symbol"] for m in all_symbol_metrics],
            "total_pnl": total_pnl,
            "total_return": total_pnl / (equity0 * max(len(symbols), 1)),
            "win_rate": win_rate,
            "pf": pf,
            "max_drawdown": max_dd,
        }

        with open(stats_all_path, "w", encoding="utf-8") as f:
            json.dump(portfolio_stats, f, ensure_ascii=False, indent=2)

        all_trades_df.to_csv(trades_all_path, index=False)

        per_symbol_df = pd.DataFrame(all_symbol_metrics)
        per_symbol_df.to_csv(per_symbol_path, index=False)
