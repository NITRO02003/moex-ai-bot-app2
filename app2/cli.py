from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from .config import config
from . import data as D, models as M, backtest as B, risk as R
from . import validation as V, diagnostics as DX
from .paths import OUT_DIR, REPORTS_DIR, MODELS_DIR, DATA_DIR

CALIB_PATH = REPORTS_DIR / 'calibration.json'


def _ensure_dirs() -> None:
    for p in (OUT_DIR, REPORTS_DIR, MODELS_DIR, DATA_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _load_prices(symbols):
    return {s: D.load_csv(s) for s in symbols}


def _load_calibration() -> dict:
    try:
        with open(CALIB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# ------------------ core commands ------------------

def cmd_update(a):
    _ensure_dirs()
    for s in a.symbols:
        df = D.fetch_range(s, a.start, a.end, a.interval, a.verbose)
        D.save_csv(s, df)


def cmd_train(a):
    _ensure_dirs()
    prices = _load_prices(a.symbols)
    info = M.fit_offline_multi(prices, horizon=a.horizon)
    print(json.dumps({'trained': info}, ensure_ascii=False))


def cmd_validate(a):
    _ensure_dirs()
    calib = _load_calibration() if getattr(a, 'use_calibrated', False) else {}
    out = {}
    for s in a.symbols:
        df = D.load_csv(s)
        if df is None or df.empty:
            out[s] = []
            continue
        folds = V.walk_forward(df, horizon=a.horizon)
        sym = calib.get(s, {})
        th = sym.get('threshold', 0.55)
        th_long = sym.get('th_long')
        th_short = sym.get('th_short')
        out[s] = [{**r, 'threshold': th, 'th_long': th_long, 'th_short': th_short} for r in folds]
    path = Path(a.out) if getattr(a, 'out', None) else (REPORTS_DIR / 'validate.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(str(path))


def cmd_calibrate(a):
    _ensure_dirs()
    bundle = M.load()
    res = {}
    for s in a.symbols:
        df = D.load_csv(s)
        if df is None or df.empty or 'close' not in df.columns:
            res[s] = {'threshold': 0.55, 'n': 0, 'objective': a.objective}
            continue
        X, y = M.dataset(df, horizon=a.horizon)
        if len(X) == 0:
            res[s] = {'threshold': 0.55, 'n': 0, 'objective': a.objective}
            continue
        close = df['close'].astype(float).reindex(X.index).ffill().bfill()
        p = M.predict_proba(bundle, X, close)
        ybin = y.values.astype(int)
        grid = np.linspace(0.40, 0.65, 51)
        best_score = -1e18
        best = {'threshold': 0.55}

        def f1(pred, yb):
            tp = int(((pred == 1) & (yb == 1)).sum())
            fp = int(((pred == 1) & (yb == 0)).sum())
            fn = int(((pred == 0) & (yb == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        def f1p(pred, yb, beta=0.5):
            tp = int(((pred == 1) & (yb == 1)).sum())
            fp = int(((pred == 1) & (yb == 0)).sum())
            fn = int(((pred == 0) & (yb == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            b2 = beta * beta
            return (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0

        ret = close.pct_change().reindex(p.index).fillna(0.0)

        if a.asymmetric:
            best_asym = {'th_long': 0.58, 'th_short': 0.42}
            for tl in grid:
                for ts in grid:
                    if a.objective == 'profit':
                        side = np.where(p.values >= tl, 1, np.where(p.values <= ts, -1, 0))
                        score = float((side * ret.values).mean())
                    elif a.objective == 'f1':
                        score = f1((p.values >= tl).astype(int), ybin)
                    elif a.objective == 'f1p':
                        score = f1p((p.values >= tl).astype(int), ybin, beta=0.5)
                    else:  # auc proxy by rank corr
                        score = float(pd.Series(p.values).rank().corr(pd.Series(ybin)))
                    if score > best_score:
                        best_score = score
                        best_asym = {'th_long': float(tl), 'th_short': float(ts)}
                        best = {'threshold': float((tl + (1 - ts)) / 2)}
            res[s] = {**best, **best_asym, 'n': int(len(X)), 'objective': a.objective, 'score': float(best_score)}
        else:
            for th in grid:
                if a.objective == 'profit':
                    side = np.where(p.values >= th, 1, -1)
                    score = float((side * ret.values).mean())
                elif a.objective == 'f1':
                    score = f1((p.values >= th).astype(int), ybin)
                elif a.objective == 'f1p':
                    score = f1p((p.values >= th).astype(int), ybin, beta=0.5)
                else:
                    score = float(pd.Series(p.values).rank().corr(pd.Series(ybin)))
                if score > best_score:
                    best_score = score
                    best = {'threshold': float(th)}
            res[s] = {**best, 'n': int(len(X)), 'objective': a.objective, 'score': float(best_score)}

    path = Path(a.out) if getattr(a, 'out', None) else (REPORTS_DIR / 'calibration.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(str(path))


def cmd_explain(a):
    _ensure_dirs()
    bundle = M.load()
    cols = bundle.get('cols', [])
    rows = []
    rf = bundle.get('rf', None)
    rf_imp = getattr(rf, 'feature_importances_', None)
    if rf_imp is not None and len(cols) == len(rf_imp):
        for c, imp in zip(cols, rf_imp):
            rows.append({'feature': c, 'rf_importance': float(imp)})
    base = bundle.get('base', None)
    sgd_coef = getattr(base, 'coef_', None)
    if sgd_coef is not None and len(sgd_coef.shape) >= 2 and len(cols) == sgd_coef.shape[1]:
        coefs = np.abs(sgd_coef[0])
        for i, c in enumerate(cols):
            row = next((r for r in rows if r['feature'] == c), None)
            if row is None:
                rows.append({'feature': c, 'sgd_abs_coef': float(coefs[i])})
            else:
                row['sgd_abs_coef'] = float(coefs[i])
    df = pd.DataFrame(rows).fillna(0.0)
    if 'rf_importance' in df.columns:
        df['rf_rank'] = df['rf_importance'].rank(ascending=False, method='min')
    if 'sgd_abs_coef' in df.columns:
        df['sgd_rank'] = df['sgd_abs_coef'].rank(ascending=False, method='min')
    if set(['rf_rank', 'sgd_rank']).issubset(df.columns):
        df['rank_sum'] = df['rf_rank'] + df['sgd_rank']
    order_cols = [c for c in ['rank_sum', 'rf_importance', 'sgd_abs_coef'] if c in df.columns]
    if order_cols:
        df = df.sort_values(by=order_cols, ascending=[True] + [False] * (len(order_cols) - 1))
    csv_path = REPORTS_DIR / 'feature_importance.csv'
    df.to_csv(csv_path, index=False)
    print(str(csv_path))


def cmd_backtest(a):
    _ensure_dirs()
    from . import backtest as _B, risk as _R, data as _D, models as _M
    bundle = _M.load()
    rp = getattr(_R, 'from_config', lambda: _R.RiskParams())()
    # поддержка обоих имён из конфигурации
    comm = getattr(config.bt_cfg, 'commission', getattr(config.bt_cfg, 'commission_rate', 0.0005))
    slip = getattr(config.bt_cfg, 'slippage_bps', getattr(config.bt_cfg, 'slippage_bp', 1.0))

    bt = _B.BtParams(commission=comm, slippage_bps=slip, horizon=a.horizon)

    calib = _load_calibration() if getattr(a, 'use_calibrated', False) else {}

    summary = {}
    for s in a.symbols:
        prices = _D.load_csv(s)
        if prices is None or prices.empty:
            continue

        # если есть калибровка — берём sym-threshold, иначе дефолт
        sym_cal = calib.get(s, {})
        th = float(sym_cal.get('threshold', 0.55))

        # ВАЖНО: не передаём th_long/th_short — текущее backtest.run_symbol их не поддерживает
        res = _B.run_symbol(
            prices, bundle, rp, bt,
            equity0=1_000_000.0,
            threshold=th
        )

        _B.save_report(s, res)
        summary[s] = res['metrics']

    path = Path(a.out) if getattr(a, 'out', None) else (REPORTS_DIR / 'backtest_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(str(path))


# ------------------ NEW: regime backtest & hypo ------------------

def cmd_regime_backtest(a):
    _ensure_dirs()
    from . import regime_backtest as RB
    from . import models as _M, data as _D, risk as _R
    bundle = _M.load()
    rp = getattr(_R, 'from_config', lambda: _R.RiskParams())()
    params = RB.RegimeBtParams(
        commission=getattr(config.bt_cfg, 'commission', 0.0005),
        slippage_bps=getattr(config.bt_cfg, 'slippage_bps', 2.0),
        atr_len=getattr(config.bt_cfg, 'atr_len', 14),
        tp_mult=getattr(config.bt_cfg, 'tp_mult', 1.5),
        sl_mult=getattr(config.bt_cfg, 'sl_mult', 1.0),
        trail_mult=getattr(config.bt_cfg, 'trail_mult', 1.0),
        min_gap=getattr(config.bt_cfg, 'min_gap', 8),
        max_hold=getattr(config.bt_cfg, 'max_hold', 120),
        cooldown=getattr(config.bt_cfg, 'cooldown', 10),
        vol_lb=getattr(config.bt_cfg, 'vol_lb', 48),
        z_thr=getattr(config.bt_cfg, 'z_thr', 0.5),
        use_filters=True,
        th_long=0.58, th_short=0.42,
        per_trade_risk=getattr(rp, 'per_trade_risk', 0.001),
    )
    calib = _load_calibration() if getattr(a, 'use_calibrated', False) else {}
    summary = {}
    for s in a.symbols:
        prices = _D.load_csv(s)
        if prices is None or prices.empty:
            continue
        sc = calib.get(s, {})
        thl = sc.get('th_long'); ths = sc.get('th_short')
        res = RB.run_symbol(prices, bundle, rp, params, equity0=1_000_000.0, th_long=thl, th_short=ths)
        summary[s] = res['metrics']
    path = Path(a.out) if getattr(a, 'out', None) else (REPORTS_DIR / 'regime_backtest_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(str(path))


def cmd_hypo(a):
    _ensure_dirs()
    from . import feature_hypo as H
    path = H.run(a.symbols, horizon=a.horizon, out=a.out)
    print(str(path))


def cmd_diag(a):
    _ensure_dirs()
    info = DX.data_summary(a.symbols)
    path = Path(a.out) if getattr(a, 'out', None) else (REPORTS_DIR / 'diag.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(str(path))


def main():
    ap = argparse.ArgumentParser(prog='app2.cli', description='MOEX APP2+ CLI')
    sub = ap.add_subparsers()

    p_u = sub.add_parser('update-data')
    p_u.add_argument('--symbols', nargs='+', required=True)
    p_u.add_argument('--start', default=None)
    p_u.add_argument('--end', default=None)
    p_u.add_argument('--interval', default='10min')
    p_u.add_argument('--verbose', action='store_true')
    p_u.set_defaults(func=cmd_update)

    p_t = sub.add_parser('train')
    p_t.add_argument('--symbols', nargs='+', required=True)
    p_t.add_argument('--horizon', type=int, default=1)
    p_t.set_defaults(func=cmd_train)

    p_v = sub.add_parser('validate')
    p_v.add_argument('--symbols', nargs='+', required=True)
    p_v.add_argument('--horizon', type=int, default=1)
    p_v.add_argument('--use-calibrated', action='store_true')
    p_v.add_argument('--out', default=str(REPORTS_DIR / 'validate.json'))
    p_v.set_defaults(func=cmd_validate)

    p_c = sub.add_parser('calibrate')
    p_c.add_argument('--symbols', nargs='+', required=True)
    p_c.add_argument('--horizon', type=int, default=1)
    p_c.add_argument('--objective', choices=['f1', 'f1p', 'profit', 'auc'], default='f1')
    p_c.add_argument('--asymmetric', action='store_true')
    p_c.add_argument('--out', default=str(REPORTS_DIR / 'calibration.json'))
    p_c.set_defaults(func=cmd_calibrate)

    p_e = sub.add_parser('explain')
    p_e.set_defaults(func=cmd_explain)

    p_b = sub.add_parser('backtest')
    p_b.add_argument('--symbols', nargs='+', required=True)
    p_b.add_argument('--horizon', type=int, default=1)
    p_b.add_argument('--use-calibrated', action='store_true')
    p_b.add_argument('--out', default=str(REPORTS_DIR / 'backtest_summary.json'))
    p_b.set_defaults(func=cmd_backtest)

    # NEW:
    p_rb = sub.add_parser('regime-backtest')
    p_rb.add_argument('--symbols', nargs='+', required=True)
    p_rb.add_argument('--use-calibrated', action='store_true')
    p_rb.add_argument('--out', default=str(REPORTS_DIR / 'regime_backtest_summary.json'))
    p_rb.set_defaults(func=cmd_regime_backtest)

    # NEW:
    p_h = sub.add_parser('hypo')
    p_h.add_argument('--symbols', nargs='+', required=True)
    p_h.add_argument('--horizon', type=int, default=1)
    p_h.add_argument('--out', default=str(REPORTS_DIR / 'feature_hypothesis.csv'))
    p_h.set_defaults(func=cmd_hypo)

    p_d = sub.add_parser('diag')
    p_d.add_argument('--symbols', nargs='+', required=True)
    p_d.add_argument('--out', default=str(REPORTS_DIR / 'diag.json'))
    p_d.set_defaults(func=cmd_diag)

    args = ap.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
