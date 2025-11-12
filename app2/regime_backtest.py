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
    floor = R.conf_gate(prob_up, close, rp_global)
    prob_up = prob_up.where(prob_up >= floor, 0.0)

    long_mom, short_mom = _entry_filter(prices, p)
    TL = float(th_long) if th_long is not None else float(p.th_long)
    TS = float(th_short) if th_short is not None else float(p.th_short)

    atr = _atr(prices, p.atr_len).reindex(prices.index).ffill()
    equity = equity0
    pos = 0          # +1 long, -1 short, 0 flat
    entry_px = 0.0; peak = 0.0; trough = 0.0
    shares = 0.0; bars_in_pos = 0; cool = 0

    pnl = pd.Series(0.0, index=prices.index)
    position_series = pd.Series(0, index=prices.index, dtype='int8')
    last_trade_i = -10_000

    for i, ts in enumerate(prices.index):
        px = float(close.iloc[i])
        this_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0

        if cool > 0:
            cool -= 1

        # === EXIT ===
        if pos != 0:
            bars_in_pos += 1
            if pos > 0:
                peak = max(peak, px)
                hit_sl = px <= entry_px - p.sl_mult * this_atr
                hit_tp = px >= entry_px + p.tp_mult * this_atr
                trail_ok = p.trail_mult > 0 and this_atr > 0 and px <= (peak - p.trail_mult * this_atr)
                time_exit = (p.max_hold > 0 and bars_in_pos >= p.max_hold)
                if hit_sl or hit_tp or trail_ok or time_exit:
                    d_sh = -shares                # sell to close long
                    cash_flow = -d_sh * px        # self-financing: sell => cash_flow > 0
                    cost = _turnover_cost(px, d_sh, p.commission, p.slippage_bps)
                    pnl.iloc[i] += cash_flow - cost; equity += cash_flow - cost
                    pos = 0; shares = 0.0; entry_px = 0.0; peak = 0.0; trough = 0.0
                    cool = p.cooldown; bars_in_pos = 0; last_trade_i = i
            else:
                trough = min(trough, px)
                hit_sl = px >= entry_px + p.sl_mult * this_atr
                hit_tp = px <= entry_px - p.tp_mult * this_atr
                trail_ok = p.trail_mult > 0 and this_atr > 0 and px >= (trough + p.trail_mult * this_atr)
                time_exit = (p.max_hold > 0 and bars_in_pos >= p.max_hold)
                if hit_sl or hit_tp or trail_ok or time_exit:
                    d_sh = -shares                # buy to cover short (shares<0 => d_sh>0)
                    cash_flow = -d_sh * px        # buy => cash_flow < 0
                    cost = _turnover_cost(px, d_sh, p.commission, p.slippage_bps)
                    pnl.iloc[i] += cash_flow - cost; equity += cash_flow - cost
                    pos = 0; shares = 0.0; entry_px = 0.0; peak = 0.0; trough = 0.0
                    cool = p.cooldown; bars_in_pos = 0; last_trade_i = i

        # === ENTRY ===
        if pos == 0 and cool == 0 and (i - last_trade_i) >= p.min_gap:
            pr = float(prob_up.iloc[i])
            enter_long = (pr >= TL) and bool(long_mom.iloc[i])
            enter_short = (pr <= TS) and bool(short_mom.iloc[i])
            if enter_long or enter_short:
                rp_tmp = rp_global or R.RiskParams(per_trade_risk=p.per_trade_risk)
                base_size = R.position_size(close.iloc[:i+1], pd.Series(prob_up.iloc[:i+1]), equity, rp_tmp).iloc[-1]
                if base_size <= 0 or np.isnan(base_size):
                    base_size = equity * p.per_trade_risk / max(px, 1e-9)
                sh = base_size / max(px, 1e-9)
                if enter_short:
                    sh = -abs(sh); pos = -1
                else:
                    sh = abs(sh); pos = 1
                entry_px = px; peak = px; trough = px
                d_sh = sh                         # from 0 to sh
                cash_flow = -d_sh * px            # buy (d_sh>0) -> cash -, short (d_sh<0) -> cash +
                cost = _turnover_cost(px, d_sh, p.commission, p.slippage_bps)
                pnl.iloc[i] += cash_flow - cost; equity += cash_flow - cost
                shares = sh; bars_in_pos = 0; last_trade_i = i

        position_series.iloc[i] = pos

    equity_curve = pnl.cumsum() + equity0
    stats = MX.summarize(equity_curve, pnl)
    stats['trades_est'] = int((position_series.diff().abs() > 0).sum() // 2)
    return {'equity': equity_curve, 'pnl': pnl, 'position': position_series, 'metrics': stats}
