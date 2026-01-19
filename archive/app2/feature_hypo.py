from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from . import features as F, data as D, models as M
from .paths import REPORTS_DIR


def permutation_importance(X: pd.DataFrame, y: pd.Series, model_bundle, n: int = 5, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base = M.evaluate_auc(model_bundle, X, y)
    out: dict[str, float] = {}
    for col in X.columns:
        scores = []
        for _ in range(n):
            Xp = X.copy()
            Xp[col] = Xp[col].sample(frac=1.0, replace=False,
                                     random_state=int(rng.integers(0, 1_000_000))).values
            scores.append(M.evaluate_auc(model_bundle, Xp, y))
        out[col] = float(base - np.mean(scores))
    # убывание по «насколько ухудшилась AUC»
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def run(symbols, horizon: int = 1, out: str | None = None) -> str:
    bundle = M.load()
    rows = []
    for s in symbols:
        df = D.load_csv(s)
        if df is None or df.empty:
            continue
        X, y = M.dataset(df, horizon=horizon)
        X = X[F.final_columns(X.columns)]
        imp = permutation_importance(X, y, bundle, n=5)
        top = list(imp.items())[:30]
        for k, v in top:
            rows.append({'symbol': s, 'feature': k, 'pi_loss_auc': v})
    res = pd.DataFrame(rows).sort_values(['symbol', 'pi_loss_auc'], ascending=[True, False])
    path = Path(out) if out else (REPORTS_DIR / 'feature_hypothesis.csv')
    res.to_csv(path, index=False)
    return str(path)
