from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import List

from app3.config.settings import Settings
from app3.dataio.loader import load_bars
from app3.dataio.signals import load_signals
from app3.engine.filters import apply_filters
from app3.engine.simulator import simulate, SimConfig
from app3.engine.risk import RiskConfig
from app3.analytics.metrics import summarize
from app3.reports.writer import write_json_report

def _load_config_file(path: str | None) -> dict:
    if not path:
        for name in ("app3.config.json","app3.config.toml","app3.config.yaml","app3.config.yml"):
            p = Path(name)
            if p.exists():
                path = str(p); break
        if not path:
            return {}
        print(f"[config] auto-discovered {path}", flush=True)

    p = Path(path)
    if not p.exists():
        print(f"[config] file not found: {p}", file=sys.stderr)
        return {}
    try:
        if p.suffix.lower() == ".json":
            import json
            return json.loads(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() == ".toml":
            try:
                import tomllib  # py3.11+
            except Exception:
                import tomli as tomllib  # type: ignore
            return tomllib.loads(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # requires PyYAML
            except Exception:
                print("[config] PyYAML not installed. Use JSON/TOML or install pyyaml.", file=sys.stderr)
                return {}
            return yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            print(f"[config] unsupported file type: {p.suffix}", file=sys.stderr)
            return {}
    except Exception as e:
        print(f"[config] failed to parse {p}: {e}", file=sys.stderr)
        return {}

def _normalize_symbols(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        if "," in val:
            return [s.strip() for s in val.split(",") if s.strip()]
        return [val.strip()] if val.strip() else []
    return []

def _merge_args_with_config(args: argparse.Namespace, parser: argparse.ArgumentParser, cfg: dict) -> argparse.Namespace:
    merged = vars(args).copy()
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    flat = cfg["backtest"] if "backtest" in cfg and isinstance(cfg["backtest"], dict) else cfg
    for k, v in flat.items():
        if k == "symbols":
            if merged.get("symbols", []) == defaults.get("symbols", []):
                merged["symbols"] = _normalize_symbols(v)
            continue
        if k in merged and merged[k] == defaults.get(k):
            merged[k] = v
    return argparse.Namespace(**merged)

def cmd_data_load(args):
    print(f"[data] loading bars for {args.symbol} from {args.base} ...", flush=True)
    bars = load_bars(args.symbol, base_dir=args.base)
    if bars is None or bars.empty:
        print(f"[data] {args.symbol}: no data", flush=True)
        return
    print(f"[data] {args.symbol}: rows={len(bars)} cols={list(bars.columns)}", flush=True)

def _dump(path: Path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        print(f"[debug] wrote {path}", flush=True)
    except Exception as e:
        print(f"[debug] failed to write {path}: {e}", file=sys.stderr, flush=True)

def cmd_backtest_run(args, parser):
    cfg_file = _load_config_file(args.config)
    if cfg_file:
        args = _merge_args_with_config(args, parser, cfg_file)
        print(f"[config] merged settings", flush=True)

    args.symbols = _normalize_symbols(args.symbols)
    if not args.symbols:
        print("ERROR: no symbols provided. Use --symbols or set 'symbols' in config.", file=sys.stderr, flush=True)
        sys.exit(2)
    print(f"[bt] symbols: {', '.join(args.symbols)}", flush=True)

    print("[bt] preparing settings ...", flush=True)
    time_start = None if args.no_time_window else (None if str(args.time_start) in ("None","null") else args.time_start)
    time_end   = None if args.no_time_window else (None if str(args.time_end) in ("None","null") else args.time_end)

    cfg = Settings(
        periods_per_year=args.periods_per_year,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        cooldown_bars=args.cooldown_bars,
        min_hold_bars=args.min_hold_bars,
        min_conf=None if str(args.min_conf) in ("None","null") else float(args.min_conf),
        top_k_per_day=None if int(args.top_k_per_day) == 0 else int(args.top_k_per_day),
        atr_threshold=None if str(args.atr_threshold) in ("None","null") else float(args.atr_threshold) if args.atr_threshold else None,
        max_spread_bps=None if str(args.max_spread_bps) in ("None","null") else float(args.max_spread_bps) if args.max_spread_bps else None,
        time_start=time_start,
        time_end=time_end,
        use_regime=args.use_regime,
    )

    risk = RiskConfig(
        atr_period=int(args.atr_period),
        atr_stop_mult=None if str(args.atr_stop_mult) in ("None","null") else float(args.atr_stop_mult),
        rr_take=None if str(args.rr_take) in ("None","null") else float(args.rr_take),
        risk_per_trade=None if str(args.risk_per_trade) in ("None","null") else float(args.risk_per_trade),
    )

    result = {}
    for sym in args.symbols:
        print(f"[bt] {sym}: loading bars ...", flush=True)
        bars = load_bars(sym, base_dir=args.data_base)
        if bars is None or bars.empty:
            print(f"[bt] {sym}: no data → skip", flush=True)
            result[sym] = {"error":"no_data"}
            continue

        print(f"[bt] {sym}: loading signals ...", flush=True)
        raw_sig = load_signals(sym, base_dir=args.signals_base)
        n_raw = len(raw_sig)
        ts_info = ""
        if n_raw > 0:
            ts_info = f" [{raw_sig['dt'].min()} .. {raw_sig['dt'].max()}]"
        print(f"[bt] {sym}: signals raw = {n_raw}{ts_info}", flush=True)
        if args.debug_dump_signals:
            _dump(Path(f"out/debug/signals_{sym}.csv"), raw_sig)

        print(f"[bt] {sym}: applying filters ...", flush=True)
        sig = apply_filters(raw_sig,
                            min_conf=cfg.min_conf,
                            top_k_per_day=cfg.top_k_per_day,
                            time_start=cfg.time_start,
                            time_end=cfg.time_end,
                            atr_threshold=cfg.atr_threshold,
                            max_spread_bps=cfg.max_spread_bps,
                            use_regime=cfg.use_regime)
        n_filt = len(sig)
        ts_info2 = ""
        if n_filt > 0:
            ts_info2 = f" [{sig['dt'].min()} .. {sig['dt'].max()}]"
        print(f"[bt] {sym}: signals filtered = {n_filt}{ts_info2}", flush=True)
        if args.debug_dump_signals:
            _dump(Path(f"out/debug/signals_filtered_{sym}.csv"), sig)

        print(f"[bt] {sym}: simulate (fees={cfg.fees_bps}bps, slip={cfg.slippage_bps}bps, "
              f"cooldown={cfg.cooldown_bars}, min_hold={cfg.min_hold_bars}, "
              f"atr_stop={risk.atr_stop_mult}, rr_take={risk.rr_take}, rpt={risk.risk_per_trade}) ...",
              flush=True)
        eq, pnl, fills = simulate(bars, sig, SimConfig(
            fees_bps=cfg.fees_bps,
            slippage_bps=cfg.slippage_bps,
            cooldown_bars=cfg.cooldown_bars,
            min_hold_bars=cfg.min_hold_bars,
            qty=float(args.qty),
            risk=risk,
        ))

        print(f"[bt] {sym}: computing metrics ...", flush=True)
        rep = summarize(eq, pnl, periods_per_year=cfg.periods_per_year)
        rep.update({
            "symbol": sym,
            "fills": len(fills),
            "fees_bps": cfg.fees_bps,
            "slippage_bps": cfg.slippage_bps,
        })
        result[sym] = rep

    print(f"[bt] writing report → {args.out}", flush=True)
    write_json_report(result, args.out)
    print("[bt] done.", flush=True)

def build_parser():
    p = argparse.ArgumentParser(prog="app3", description="Clean trading backtester (v2)")
    sub = p.add_subparsers(dest="cmd")

    # data
    p_data = sub.add_parser("data", help="data utilities")
    p_data_sub = p_data.add_subparsers(dest="subcmd")

    p_dl = p_data_sub.add_parser("load", help="load bars and print info")
    p_dl.add_argument("--symbol", required=True)
    p_dl.add_argument("--base", default="data")
    p_dl.set_defaults(func=cmd_data_load)

    # backtest
    p_bt = sub.add_parser("backtest", help="backtest utilities")
    p_bt_sub = p_bt.add_subparsers(dest="subcmd")

    p_run = p_bt_sub.add_parser("run", help="run backtest and save JSON report")
    p_run.add_argument("--symbols", nargs="*", default=[])
    p_run.add_argument("--data-base", default="data")
    p_run.add_argument("--signals-base", default="signals")
    p_run.add_argument("--out", default="out/reports/app3_summary.json")

    # filters & costs
    p_run.add_argument("--min-conf", type=str, default="0.65")
    p_run.add_argument("--top-k-per-day", type=int, default=3)
    p_run.add_argument("--cooldown-bars", type=int, default=20)
    p_run.add_argument("--min-hold-bars", type=int, default=10)
    p_run.add_argument("--atr-threshold", type=str, default="None")
    p_run.add_argument("--max-spread-bps", type=str, default="5.0")
    p_run.add_argument("--time-start", type=str, default="11:10")
    p_run.add_argument("--time-end", type=str, default="18:30")
    p_run.add_argument("--use-regime", action="store_true")
    p_run.add_argument("--no-time-window", action="store_true", help="ignore time window filtering")

    p_run.add_argument("--fees-bps", type=float, default=1.0)
    p_run.add_argument("--slippage-bps", type=float, default=0.0)
    p_run.add_argument("--periods-per-year", type=float, default=252.0)

    # risk & sizing
    p_run.add_argument("--qty", type=float, default=1.0, help="fixed qty if no risk sizing")
    p_run.add_argument("--atr-period", type=int, default=14)
    p_run.add_argument("--atr-stop-mult", type=str, default="2.0")
    p_run.add_argument("--rr-take", type=str, default="2.0")
    p_run.add_argument("--risk-per-trade", type=str, default="0.005")

    # config file
    p_run.add_argument("--config", type=str, default=None, help="path to JSON/TOML/YAML config; auto-discover app3.config.* if omitted")

    # debugging
    p_run.add_argument("--debug-dump-signals", action="store_true", help="dump raw and filtered signals to out/debug/")

    p_run.set_defaults(func=lambda a: cmd_backtest_run(a, p_run))

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
