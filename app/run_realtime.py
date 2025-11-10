
# app/run_realtime.py
from __future__ import annotations
import argparse, datetime as dt, os
from .data_fetch import fetch_moex_candles_10m
from .utils import ensure_dtindex_utc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["SBER","GAZP","LKOH","GMKN","ROSN"])
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--boards", nargs="+", default=["TQBR","TQTF","TQTD","FQBR"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    end = dt.date.today().isoformat()
    start = (dt.date.today() - dt.timedelta(days=args.days)).isoformat()

    for s in args.symbols:
        df = fetch_moex_candles_10m(s, start=start, end=end, boards=args.boards, verbose=args.verbose)
        df = ensure_dtindex_utc(df)
        out = os.path.join(args.data_dir, f"{s}.csv")
        df.to_csv(out)
        print(f"[rt] {s}: saved {len(df)} rows -> {out}")

if __name__ == "__main__":
    main()
