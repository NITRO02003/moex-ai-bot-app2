from __future__ import annotations
import numpy as np, pandas as pd
from . import features as F, models as M, risk as R

def signal_and_size(prices: pd.DataFrame, model_bundle, rp: R.RiskParams, equity0: float,
                    threshold: float | None = None, th_long: float | None = None, th_short: float | None = None,
                    use_filters: bool = True):
    X = F.build(prices); X = X[F.final_columns(X.columns)]
    close = prices["close"].astype(float)
    p = M.predict_proba(model_bundle, X, close).reindex(prices.index).ffill().bfill()

    # 1) Confidence-gate: где нет уверенности модели — вообще не торгуем
    floor = R.conf_gate(p, close, rp)
    conf_mask = p >= floor

    # 2) Ассиметричные пороги: шорт реже и только по сильному сигналу
    if th_long is not None and th_short is not None:
        thL_raw = float(th_long)
        thS_raw = float(th_short)
    else:
        # по умолчанию только лонг, если задан threshold
        thL_raw = float(threshold) if threshold is not None else 0.55
        thS_raw = 0.0  # шортов нет

    # лонг не мягче 0.55, шорт только при p <= 0.40
    thL = max(thL_raw, 0.55)
    thS = min(thS_raw, 0.40)

    # гарантируем зазор между лонг/шорт
    if thL <= thS + 0.05:
        thL = thS + 0.05

    # базовый сайд
    side = np.where(p >= thL, 1,
            np.where(p <= thS, -1, 0))
    side = pd.Series(side, index=prices.index, dtype="int8")

    # где нет уверенности модели — сидим в кэше
    side = side.where(conf_mask, 0)

    # базовый размер позиции по риску
    size = R.position_size(close, p, equity0, rp).reindex(prices.index).ffill().bfill()
    size = size.where(conf_mask, 0.0)

    if use_filters:
        # 3) Фильтр по волатильности и тренду (EMA 20/60)
        vol = R.realized_vol(close, rp.vol_lb).reindex(prices.index).ffill().bfill()
        vol_thr = vol.rolling(200, min_periods=20).quantile(0.2).reindex(prices.index).ffill().bfill()
        vol_ok = vol >= vol_thr

        ema_fast = close.ewm(span=20, adjust=False).mean()
        ema_slow = close.ewm(span=60, adjust=False).mean()

        up_trend = ema_fast > ema_slow
        down_trend = ema_fast < ema_slow

        long_ok = (side > 0) & up_trend
        short_ok = (side < 0) & down_trend
        trend_ok = long_ok | short_ok

        ok = vol_ok & trend_ok
        side = side.where(ok, 0)
        size = size.where(ok, 0.0)

    return side, size
