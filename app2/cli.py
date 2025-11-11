from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from .config import config
from . import data as D, models as M, metrics as MX
from . import validation as V, diagnostics as DX
from .paths import OUT_DIR, REPORTS_DIR, MODELS_DIR, DATA_DIR

CALIB_PATH = REPORTS_DIR / 'calibration.json'


def _ensure_dirs():
    for p in (OUT_DIR, REPORTS_DIR, MODELS_DIR, DATA_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _load_prices(symbols):
    return {s: D.load_csv(s) for s in symbols}


def _load_calibration():
    try:
        with open(CALIB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


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
        th = calib.get(s, {}).get('threshold', 0.5)
        out[s] = [{**r, 'threshold': th} for r in folds]
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
            res[s] = {'threshold': 0.5, 'auc': None, 'n': 0}
            continue
        X, y = M.dataset(df, horizon=a.horizon)
        if len(X) == 0:
            res[s] = {'threshold': 0.5, 'auc': None, 'n': 0}
            continue
        p = M.predict_proba(bundle, X, df['close'].astype(float))
        ths = np.linspace(0.4, 0.6, 41)
        best_f1, best_th = -1.0, 0.5
        ybin = y.values.astype(int)
        for th in ths:
            pred = (p.values >= th).astype(int)
            tp = int(((pred == 1) & (ybin == 1)).sum())
            fp = int(((pred == 1) & (ybin == 0)).sum())
            fn = int(((pred == 0) & (ybin == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)
        auc = M.auc(bundle, X, y, df['close'].astype(float))
        res[s] = {'threshold': best_th, 'f1': best_f1, 'auc': auc, 'n': int(len(X))}
    path = Path(a.out) if getattr(a, 'out', None) else (REPORTS_DIR / 'calibration.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(str(path))


def cmd_explain(a):
    _ensure_dirs()
    bundle = M.load()
    cols = bundle.get('cols', [])
    rows = []

    # Исправление: проверяем наличие моделей и атрибутов
    rf = bundle.get('rf', None)
    if rf is not None and hasattr(rf, 'feature_importances_'):
        rf_imp = rf.feature_importances_
        if len(cols) == len(rf_imp):
            for c, imp in zip(cols, rf_imp):
                rows.append({'feature': c, 'rf_importance': float(imp)})

    base = bundle.get('base', None)
    if base is not None and hasattr(base, 'coef_'):
        sgd_coef = base.coef_
        if len(sgd_coef.shape) >= 2 and len(cols) == sgd_coef.shape[1]:
            coefs = np.abs(sgd_coef[0])
            for i, c in enumerate(cols):
                row = next((r for r in rows if r['feature'] == c), None)
                if row is None:
                    rows.append({'feature': c, 'sgd_abs_coef': float(coefs[i])})
                else:
                    row['sgd_abs_coef'] = float(coefs[i])

    if not rows:
        print("No feature importance data available")
        return

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
    # Ленивый импорт для избежания циклических зависимостей
    from . import backtest as B, risk as R

    bundle = M.load()
    rp = R.from_config()
    comm = getattr(config.bt_cfg, 'commission', getattr(config.bt_cfg, 'commission_rate', 0.0005))
    slip = getattr(config.bt_cfg, 'slippage_bps', getattr(config.bt_cfg, 'slippage_bp', 1.0))
    bt = B.BtParams(commission=comm, slippage_bps=slip, horizon=a.horizon)
    calib = _load_calibration() if getattr(a, 'use_calibrated', False) else {}
    summary = {}
    for s in a.symbols:
        prices = D.load_csv(s)
        if prices is None or prices.empty:
            continue
        th = calib.get(s, {}).get('threshold', 0.5)
        res = B.run_symbol(prices, bundle, rp, bt, equity0=1_000_000.0, threshold=th)
        B.save_report(s, res)
        summary[s] = res['metrics']
    path = Path(a.out) if getattr(a, 'out', None) else (REPORTS_DIR / 'backtest_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(str(path))


def cmd_backtest_conservative(args):
    """Бэктест с консервативными настройками"""
    # Используем conservative_strategy вместо signal_and_size
    # с порогами 0.65/0.35 и фиксированным размером 1%


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