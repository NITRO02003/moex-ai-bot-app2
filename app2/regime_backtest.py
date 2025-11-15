from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from . import features as F, models as M, risk as R, metrics as MX

@dataclass
class RegimeBtParams:
    commission: float = 0.0005
    slippage_bps: float = 2.0
    atr_len: int = 14
    tp_mult: float = 1.5
    sl_mult: float = 1.0
    trail_mult: float = 1.0
    min_gap: int = 8
    max_hold: int = 120
    cooldown: int = 10
    vol_lb: int = 48
    z_thr: float = 0.5
    use_filters: bool = True
    th_long: float = 0.58
    th_short: float = 0.42
    per_trade_risk: float = 0.001

def _ema_trend(close: pd.Series, fast=20, slow=60):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    slope = ema_f.diff()
    return (ema_f > ema_s) & (slope > 0), (ema_f < ema_s) & (slope < 0)

def _vol_ok(vol: pd.Series, q=0.2):
    thr = vol.rolling(100).quantile(q)
    return (vol >= thr).fillna(False)

def _zscore(s: pd.Series, lb: int):
    m = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    return (s - m) / sd.replace(0, np.nan)

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _turnover_cost(price: float, d_shares: float, commission: float, slippage_bps: float) -> float:
    turn = abs(d_shares) * price
    comm = turn * float(commission)
    slip = turn * float(slippage_bps) / 10_000.0
    return comm + slip

def _entry_filter(df: pd.DataFrame, p: RegimeBtParams):
    close = df['close'].astype(float)
    ret1 = close.pct_change().fillna(0.0)
    z = _zscore(ret1, p.vol_lb).fillna(0.0)
    long_mom = z >= p.z_thr
    short_mom = z <= -p.z_thr
    if p.use_filters:
        up, dn = _ema_trend(close)
        vol_ok = _vol_ok(df['volume'].fillna(0), 0.2)
        long_mom &= up & vol_ok
        short_mom &= dn & vol_ok
    return long_mom, short_mom

def run_symbol(prices: pd.DataFrame, model_bundle, rp_global: R.RiskParams | None, p: RegimeBtParams,
               equity0: float = 1_000_000.0, th_long: float | None = None, th_short: float | None = None):
    # features + proba
    X = F.build(prices); X = X[F.final_columns(X.columns)]
    close = prices['close'].astype(float)
    prob_up = M.predict_proba(model_bundle, X, close).reindex(prices.index).ffill().bfill()

    # confidence-gate
    if rp_global is None:
        rp_global = R.RiskParams()
    floor = R.conf_gate(prob_up, close, rp_global)
    conf_mask = prob_up >= floor

    # momentum + тренд/объём фильтры
    long_mom, short_mom = _entry_filter(prices, p)

    # ассиметричные пороги
    TL_raw = float(th_long) if th_long is not None else float(p.th_long)
    TS_raw = float(th_short) if th_short is not None else float(p.th_short)

    TL = max(TL_raw, 0.55)
    TS = min(TS_raw, 0.40)
    if TL <= TS + 0.05:
        TL = TS + 0.05

    atr = _atr(prices, p.atr_len).reindex(prices.index).ffill()
    equity = equity0
    pos = 0          # +1 long, -1 short, 0 flat
    shares = 0.0
    entry_px = 0.0
    peak = 0.0; trough = 0.0
    bars_in_pos = 0
    last_trade_i = -1

    pnl = pd.Series(0.0, index=prices.index)
    position_series = pd.Series(0, index=prices.index, dtype="int8")

    for i, ts in enumerate(prices.index):
        px = float(close.loc[ts])
        this_atr = float(atr.loc[ts]) if not np.isnan(atr.loc[ts]) else 0.0

        # 1) управление позицией (TP/SL/Trail/MaxHold)
        exit_now = False
        if pos != 0:
            bars_in_pos += 1
            if pos > 0:
                peak = max(peak, px)
                hit_sl = px <= entry_px - p.sl_mult * this_atr
                hit_tp = px >= entry_px + p.tp_mult * this_atr
                trail_ok = p.trail_mult > 0 and px <= peak - p.trail_mult * this_atr
            else:
                trough = min(trough, px)
                hit_sl = px >= entry_px + p.sl_mult * this_atr
                hit_tp = px <= entry_px - p.tp_mult * this_atr
                trail_ok = p.trail_mult > 0 and px >= trough + p.trail_mult * this_atr

            time_exit = p.max_hold > 0 and bars_in_pos >= p.max_hold
            exit_now = hit_sl or hit_tp or trail_ok or time_exit

        if exit_now and pos != 0:
            d_sh = -shares
            cash_flow = -d_sh * px
            cost = _turnover_cost(px, d_sh, p.commission, p.slippage_bps)
            pnl.iloc[i] += cash_flow - cost
            equity += cash_flow - cost
            shares = 0.0; pos = 0; bars_in_pos = 0

        # 2) проверка входа (confidence + momentum + тренд/объём)
        enter_long = enter_short = False
        if conf_mask.loc[ts]:
            pr = float(prob_up.loc[ts])
            if pr >= TL and bool(long_mom.loc[ts]):
                enter_long = True
            if pr <= TS and bool(short_mom.loc[ts]):
                enter_short = True

        # 3) открытие новой позиции (с ограничением плеча и риском на сделку)
        if pos == 0 and (enter_long or enter_short):
            # риск на сделку в деньгах
            dollar_risk = max(float(rp_global.per_trade_risk) * float(equity), 1.0)
            # расстояние до стопа ~ sl_mult * ATR, но не меньше 1% цены
            stop_dist = max(p.sl_mult * this_atr, 0.01 * px)
            if stop_dist <= 0:
                stop_dist = 0.01 * px
            sh = dollar_risk / stop_dist

            # ограничиваем плечо через max_gross
            max_gross = float(getattr(rp_global, "max_gross", 1.0))
            max_notional = max_gross * float(equity)
            notional = min(sh * px, max_notional)
            sh = notional / px if px > 0 else 0.0

            if sh <= 0:
                position_series.iloc[i] = pos
                continue

            if enter_short:
                pos = -1
                sh = -sh
            else:
                pos = 1
                sh = abs(sh)

            entry_px = px
            peak = px
            trough = px
            d_sh = sh
            cash_flow = -d_sh * px
            cost = _turnover_cost(px, d_sh, p.commission, p.slippage_bps)
            pnl.iloc[i] += cash_flow - cost
            equity += cash_flow - cost
            shares = sh
            bars_in_pos = 0
            last_trade_i = i

        position_series.iloc[i] = pos

    equity_curve = pnl.cumsum() + equity0
    stats = MX.summarize(equity_curve, pnl)
    stats['trades_est'] = int((position_series.diff().abs() > 0).sum() // 2)
    return {'equity': equity_curve, 'pnl': pnl, 'position': position_series, 'metrics': stats}

