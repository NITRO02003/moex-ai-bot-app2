
def main(args):
    from .forward_test import run_forward_test
    run_forward_test(
        strategy=args.strategy,
        symbols=args.symbols,
        interval=args.interval,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        equity0=args.equity0,
        out_path=args.out,
        use_breakout_in_high_vol=args.use_breakout_in_high_vol
    )
