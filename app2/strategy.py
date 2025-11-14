from __future__ import annotations
import numpy as np, pandas as pd
from . import features as F, models as M, risk as R

def signal_and_size(prices: pd.DataFrame, model_bundle, rp: R.RiskParams, equity0: float,
                    threshold: float | None = None, th_long: float | None = None, th_short: float | None = None,
                    use_filters: bool = True):
    X = F.build(prices); X = X[F.final_columns(X.columns)]
    close = prices["close"].astype(float)
    p = M.predict_proba(model_bundle, X, close).reindex(prices.index).ffill().bfill()
    floor = R.conf_gate(p, close, rp)
    p = p.where(p >= floor, 0.0)

    if th_long is not None and th_short is not None:
        side = np.where(p >= float(th_long), 1, np.where(p <= float(th_short), -1, 0))
    else:
        th = float(threshold) if threshold is not None else 0.55
        side = np.where(p >= th, 1, -1)
    side = pd.Series(side, index=prices.index, dtype="int8")

    size = R.position_size(close, p, equity0, rp).reindex(prices.index).ffill().bfill()
    if use_filters:
        vol = R.realized_vol(close, rp.vol_lb).reindex(prices.index).ffill().bfill()
        q = vol.rolling(200).quantile(0.2).reindex(prices.index).ffill().bfill()
        ok = vol >= q
        side = side.where(ok, 0)
        size = size.where(ok, 0.0)

    return side, size
