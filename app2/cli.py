
from __future__ import annotations
import argparse, json
from .config import config
from . import data as D, models as M, backtest as B, risk as R
from . import validation as V

def cmd_update(a):
    D.fetch_range(a.symbols[0], a.start, a.end, a.interval, a.verbose)  # allow side effects in user's code too
    for s in a.symbols:
        df = D.fetch_range(s, a.start, a.end, a.interval, a.verbose)
        D.save_csv(s, df)

def cmd_train(a):
    prices = {s: D.load_csv(s) for s in a.symbols}
    info = M.fit_offline_multi(prices, horizon=a.horizon)
    print('trained:', info)

def cmd_train_wf(a):
    prices = D.load_csv(a.symbols[0])
    res = V.walk_forward(prices, horizon=a.horizon)
    print(json.dumps(res, ensure_ascii=False, indent=2))

def cmd_backtest(a):
    prices = D.load_csv(a.symbols[0])
    bundle = M.load()
    rp = R.from_config()
    out = B.run_symbol(prices, bundle, rp, B.BtParams(horizon=a.horizon))
    path = B.save_report(f'bt_{a.symbols[0]}', out)
    print('report:', path, 'metrics:', out['metrics'])

def main():
    ap = argparse.ArgumentParser(prog='app2')
    sub = ap.add_subparsers()

    p_u = sub.add_parser('update-data')
    p_u.add_argument('--symbols', nargs='+', default=config.symbols_cfg.symbols)
    p_u.add_argument('--start', type=str, default=config.bt_cfg.start_date)
    p_u.add_argument('--end', type=str, default=config.bt_cfg.end_date)
    p_u.add_argument('--interval', type=str, default=config.bt_cfg.interval)
    p_u.add_argument('--verbose', action='store_true')
    p_u.set_defaults(func=cmd_update)

    p_t = sub.add_parser('train')
    p_t.add_argument('--symbols', nargs='+', default=config.symbols_cfg.symbols)
    p_t.add_argument('--horizon', type=int, default=config.model_cfg.horizon if hasattr(config,'model_cfg') else 1)
    p_t.set_defaults(func=cmd_train)

    p_tw = sub.add_parser('train-wf')
    p_tw.add_argument('--symbols', nargs='+', default=config.symbols_cfg.symbols[:1])
    p_tw.add_argument('--horizon', type=int, default=1)
    p_tw.set_defaults(func=cmd_train_wf)

    p_b = sub.add_parser('backtest')
    p_b.add_argument('--symbols', nargs='+', default=config.symbols_cfg.symbols[:1])
    p_b.add_argument('--horizon', type=int, default=1)
    p_b.set_defaults(func=cmd_backtest)

    args = ap.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        ap.print_help()

if __name__ == '__main__':
    main()
