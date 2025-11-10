
from __future__ import annotations
import pandas as pd
from . import features as F, models as M, risk as R

def signal_and_size(prices: pd.DataFrame, model_bundle, rp: R.RiskParams, equity: float):
    X = F.build(prices); X = X[F.final_columns(X.columns)]
    p = M.predict_proba(model_bundle, X, prices['close'].astype(float))
    side = ((p >= 0.5).astype(int)*2 - 1).where(p >= R.conf_gate(p, prices['close'], rp), 0)
    size = R.position_size(prices['close'].astype(float), p, equity, rp)
    return pd.DataFrame({'p': p, 'side': side, 'size': size}).reindex(prices.index).fillna(0.0)
