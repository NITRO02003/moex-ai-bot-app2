from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

from .config import load_config
from .utils import load_symbols
from .rule_core import RuleBtParams, run_rule_symbol
from .rule_strategies import (
    TrendParams,
    MeanRevParams,
    BreakoutParams,
    MeanRevV2Params,
    generate_trend_signals,
    generate_meanrev_signals,
    generate_breakout_signals,
    generate_meanrev_v2_signals,
)


# ---------- загрузка данных ----------


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
            # выбрасываем бары без цены, чтобы не ломать equity/PnL
            if "close" in df.columns:
                df = df[~df["close"].isna()].copy()
            # стандартизируем временную колонку
            if "begin" in df.columns and "datetime" not in df.columns:
                df["datetime"] = pd.to_datetime(df["begin"])
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            else:
                print(f"[analyze-trades] {sym}: no datetime/begin column in {path}, skip")
                return None
            return df

    print(f"[analyze-trades] {sym}: no data file found")
    return None


# ---------- helpers для параметров стратегий ----------


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



def _build_meanrev_v2_params(cfg: Dict[str, Any]) -> MeanRevV2Params:
    p = MeanRevV2Params()
    if "ma_len" in cfg:
        p.ma_len = int(cfg["ma_len"])
    if "atr_len" in cfg:
        p.atr_len = int(cfg["atr_len"])
    if "z_entry" in cfg:
        p.z_entry = float(cfg["z_entry"])
    if "z_entry_long" in cfg and cfg["z_entry_long"] is not None:
        p.z_entry_long = float(cfg["z_entry_long"])
    if "z_entry_short" in cfg and cfg["z_entry_short"] is not None:
        p.z_entry_short = float(cfg["z_entry_short"])
    if "regime_filter" in cfg:
        p.regime_filter = tuple(cfg["regime_filter"])
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


# ---------- простые графики ----------


def _plot_equity(bars: pd.DataFrame, title: str, out_path: Path) -> None:
    if bars.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bars["datetime"], bars["equity"])
    ax.set_title(title)
    ax.set_xlabel("datetime")
    ax.set_ylabel("equity")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pnl_dist(trades: pd.DataFrame, title: str, out_path: Path) -> None:
    if trades.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(trades["pnl_abs"], bins=50)
    ax.set_title(title)
    ax.set_xlabel("PnL per trade (abs)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_holdtime_dist(trades: pd.DataFrame, title: str, out_path: Path) -> None:
    if trades.empty or "bars_in_trade" not in trades.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(trades["bars_in_trade"], bins=30)
    ax.set_title(title)
    ax.set_xlabel("bars_in_trade")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hourly_pnl(trades: pd.DataFrame, title: str, out_path: Path) -> None:
    if trades.empty or "exit_dt" not in trades.columns:
        return
    df = trades.copy()
    df["exit_dt"] = pd.to_datetime(df["exit_dt"])
    df["hour"] = df["exit_dt"].dt.hour
    grp = df.groupby("hour")["pnl_abs"].sum()
    if grp.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_title(title)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("sum pnl_abs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- основная функция ----------


def run_analyze_trades(
    strategy: str,
    symbols: List[str],
    interval: str,
    equity0: float,
    config_path: str,
    out_prefix: str,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Анализ сделок и bar-логов для стратегии и набора тикеров с поддержкой профилей.

    strategy: 'trend' | 'meanrev' | 'breakout'
    symbols: список тикеров (или ['all'])
    interval: таймфрейм ('30min', '1h', ...)
    equity0: стартовый капитал
    config_path: путь к config.json
    out_prefix: префикс имени выходных файлов, например 'out/diag_meanrev'
    profile: имя профиля из config.json["profiles"]
             (например, 'conservative' или 'aggressive')

    Параметры берутся так:
      - defaults.* из config["defaults"]
      - если profile задан:
          config["profiles"][profile]["MeanRevParams"] и/или ["RuleBtParams"]
        переопределяют соответствующие поля defaults.
    """
    config = load_config(config_path)
    symbols = load_symbols(symbols)

    defaults = config.get("defaults", {})
    profiles = config.get("profiles", {}) or {}
    profile_cfg = profiles.get(profile) if profile else None

    # --- RuleBtParams ---
    bt_cfg = dict(defaults.get("RuleBtParams", {}))
    if profile_cfg and "RuleBtParams" in profile_cfg:
        bt_cfg.update(profile_cfg["RuleBtParams"])
    bt_params = RuleBtParams(**bt_cfg)

    # --- параметры режимов ---
    regime_cfg = dict(defaults.get("RegimeParams", {}))
    if profile_cfg and "RegimeParams" in profile_cfg:
        regime_cfg.update(profile_cfg["RegimeParams"])

    # --- базовая конфигурация стратегии (общая для всех тикеров) ---
    if strategy == "trend":
        strat_cfg = dict(defaults.get("TrendParams", {}))
        # при желании можно тоже завести profile_cfg["TrendParams"]
        if profile_cfg and "TrendParams" in profile_cfg:
            strat_cfg.update(profile_cfg["TrendParams"])
    elif strategy == "meanrev":
        strat_cfg = dict(defaults.get("MeanRevParams", {}))
        if profile_cfg and "MeanRevParams" in profile_cfg:
            strat_cfg.update(profile_cfg["MeanRevParams"])
    elif strategy == "meanrev_v2":
        strat_cfg = dict(defaults.get("MeanRevV2Params", {}))
        if profile_cfg and "MeanRevV2Params" in profile_cfg:
            strat_cfg.update(profile_cfg["MeanRevV2Params"])
    elif strategy == "breakout":
        strat_cfg = dict(defaults.get("BreakoutParams", {}))
        if profile_cfg and "BreakoutParams" in profile_cfg:
            strat_cfg.update(profile_cfg["BreakoutParams"])
    else:
        raise NotImplementedError(f"strategy '{strategy}' is not supported")

    profile_info = profile if profile is not None else "default"
    print(
        f"[analyze-trades] start, strategy={strategy}, profile={profile_info}, "
        f"symbols={symbols}, interval={interval}, equity0={equity0}, config={config_path}"
    )
    print(f"[analyze-trades] RuleBtParams effective: {bt_cfg}")
    print(f"[analyze-trades] {strategy} params effective: {strat_cfg}")

    out_path = Path(out_prefix)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for sym in symbols:
        df = _load_prices(sym, interval=interval)
        if df is None:
            continue

        print(f"[analyze-trades] {sym}: run strategy={strategy}, profile={profile_info}")

        # генерируем сигналы
        if strategy == "trend":
            s_params = _build_trend_params(strat_cfg)
            side = generate_trend_signals(df, s_params)
            df2 = df.copy()
            df2["signal"] = side
        elif strategy == "meanrev":
            s_params = _build_meanrev_params(strat_cfg)
            side = generate_meanrev_signals(df, s_params)
            df2 = df.copy()
            df2["signal"] = side
        elif strategy == "meanrev_v2":
            s_params = _build_meanrev_v2_params(strat_cfg)
            side, z_score, regime = generate_meanrev_v2_signals(
                df,
                s_params,
                regime_params=regime_cfg,
            )
            df2 = df.copy()
            df2["signal"] = side
            df2["z_score"] = z_score
            df2["regime"] = regime
        elif strategy == "breakout":
            s_params = _build_breakout_params(strat_cfg)
            side = generate_breakout_signals(df, s_params)
            df2 = df.copy()
            df2["signal"] = side
        else:
            raise NotImplementedError(f"strategy '{strategy}' is not supported")

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
            print(f"[analyze-trades] {sym}: no bar_stats or trades, skip")
            continue

        prof_tag = profile if profile is not None else "default"
        base_name = f"{out_path.name}_{prof_tag}_{sym}_{strategy}_{interval}"

        bars_file = out_dir / f"{base_name}_bars.csv"
        trades_file = out_dir / f"{base_name}_trades.csv"

        bar_stats.to_csv(bars_file, index=False)
        trades.to_csv(trades_file, index=False)

        print(
            f"[analyze-trades] {sym}: bars={len(bar_stats)}, "
            f"trades={len(trades)}, total_return={metrics.get('total_return', 0):.4f}"
        )

        # графики
        _plot_equity(
            bar_stats,
            title=f"{sym} {strategy} ({prof_tag}) equity",
            out_path=out_dir / f"{base_name}_equity.png",
        )
        _plot_pnl_dist(
            trades,
            title=f"{sym} ({prof_tag}) PnL dist",
            out_path=out_dir / f"{base_name}_pnl_dist.png",
        )
        _plot_holdtime_dist(
            trades,
            title=f"{sym} ({prof_tag}) holdtime dist",
            out_path=out_dir / f"{base_name}_holdtime.png",
        )
        _plot_hourly_pnl(
            trades,
            title=f"{sym} ({prof_tag}) hourly PnL",
            out_path=out_dir / f"{base_name}_hourly_pnl.png",
        )

        row = {
            "symbol": sym,
            "profile": prof_tag,
            "strategy": strategy,
            "interval": interval,
            **metrics,
        }
        summary_rows.append(row)

        summary.setdefault(sym, {})
        summary[sym][prof_tag] = {
            "bars": len(bar_stats),
            "trades": len(trades),
            "metrics": metrics,
            "bars_file": str(bars_file),
            "trades_file": str(trades_file),
        }

    # сводный CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = out_dir / f"{out_path.name}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"[analyze-trades] summary saved to {summary_file}")

    print(f"[analyze-trades] done, symbols_processed={len(summary)}")
    return summary
