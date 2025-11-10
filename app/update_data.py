# app/update_data.py
from __future__ import annotations
import os, argparse
from typing import Iterable
from .data_fetch import fetch_moex_candles_10m, fetch_moex_candles_10m_paginated, DEFAULT_BOARDS
from .utils import ensure_dtindex_utc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["SBER", "GAZP", "LKOH", "GMKN", "ROSN"])
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--boards", nargs="+", default=list(DEFAULT_BOARDS))
    ap.add_argument("--interval", type=str, default="10min")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--paginated", action="store_true", help="Use pagination to get more than 500 rows")
    ap.add_argument("--chunked", action="store_true", help="Use monthly chunks for very large periods")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Если запрашиваем большой период, используем специальный скрипт
    if args.chunked and args.start and args.end:
        from .update_data_chunked import load_data_by_months
        load_data_by_months(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir,
            verbose=args.verbose
        )
        return

    for sym in args.symbols:
        if args.paginated and args.start and args.end:
            # Используем пагинацию
            df = fetch_moex_candles_10m_paginated(
                sym,
                start=args.start,
                end=args.end,
                boards=args.boards,
                verbose=args.verbose
            )
        else:
            # Обычная загрузка (макс 500 строк)
            df = fetch_moex_candles_10m(
                sym,
                start=args.start,
                end=args.end,
                boards=args.boards,
                verbose=args.verbose
            )

        df = ensure_dtindex_utc(df)
        out_path = os.path.join(args.data_dir, f"{sym}.csv")
        df.to_csv(out_path)
        print(f"[save] {sym} -> {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()