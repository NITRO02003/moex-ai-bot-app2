
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


TZ = "Europe/Moscow"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research study: timing of strong daily moves vs overnight gap and IMOEX benchmark."
    )
    p.add_argument("--symbols", nargs="+", default=["SBER", "GAZP", "LKOH"])
    p.add_argument("--index-symbol", default="IMOEX")
    p.add_argument("--event-threshold", type=float, default=0.03, help="Absolute daily move threshold from previous close.")
    p.add_argument("--market-ratio-threshold", type=float, default=0.6, help="If |index move| / |stock move| >= threshold, classify as market-driven.")
    p.add_argument("--out-dir", default="research/out")
    return p.parse_args()


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path)

    if "begin" not in df.columns:
        df = pd.read_csv(path, header=None)
        if len(df) == 0:
            raise ValueError(f"Пустой файл: {path}")
        header = [str(x).strip() for x in df.iloc[0].tolist()]
        df = df.iloc[1:].copy()
        df.columns = header

    required = {"begin", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле {path} отсутствуют колонки: {sorted(missing)}")

    df = df[["begin", "open", "high", "low", "close", "volume"]].copy()
    df["begin"] = pd.to_datetime(df["begin"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["begin", "open", "high", "low", "close"]).copy()
    df["dt_msk"] = df["begin"].dt.tz_convert(TZ)
    df["session_date"] = df["dt_msk"].dt.date
    df["time_msk"] = df["dt_msk"].dt.strftime("%H:%M")
    df = df.sort_values("dt_msk").reset_index(drop=True)
    return df


def load_10min(symbol: str) -> pd.DataFrame:
    root = Path(".")
    candidates = [
        root / "processed" / f"{symbol}_10min.csv",
        root / "data" / f"{symbol}_10min.csv",
        root / "processed" / f"{symbol}.csv",
        root / "data" / f"{symbol}.csv",
    ]
    for path in candidates:
        if path.exists():
            return _read_ohlcv_csv(path)
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Не найден 10-минутный файл для {symbol}. Проверены пути:\n{tried}")


def build_sessions(df_10m: pd.DataFrame) -> pd.DataFrame:
    sessions = (
        df_10m.groupby("session_date", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            bars=("close", "size"),
            first_dt=("dt_msk", "first"),
            last_dt=("dt_msk", "last"),
        )
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    sessions["prev_close"] = sessions["close"].shift(1)
    sessions["overnight_gap"] = (sessions["open"] - sessions["prev_close"]) / sessions["prev_close"]
    sessions["day_high_move"] = (sessions["high"] - sessions["prev_close"]) / sessions["prev_close"]
    sessions["day_low_move"] = (sessions["low"] - sessions["prev_close"]) / sessions["prev_close"]
    sessions["day_close_move"] = (sessions["close"] - sessions["prev_close"]) / sessions["prev_close"]
    return sessions


def time_bucket(dt_str: str) -> str:
    hhmm = dt_str[:5]
    if hhmm < "12:00":
        return "before_12"
    if hhmm < "14:00":
        return "12_14"
    return "after_14"


def classify_driver(stock_move_abs: float, index_move_abs: float, threshold: float) -> str:
    if pd.isna(index_move_abs) or stock_move_abs <= 0:
        return "unknown"
    ratio = index_move_abs / stock_move_abs
    if ratio >= threshold:
        return "market_driven"
    return "stock_specific"


def analyze_symbol(symbol: str, idx_df: pd.DataFrame, event_threshold: float, market_ratio_threshold: float) -> pd.DataFrame:
    df = load_10min(symbol)
    sessions = build_sessions(df)
    idx_sessions = build_sessions(idx_df)

    idx_map = idx_sessions.set_index("session_date")[["prev_close", "open", "high", "low", "close"]]

    rows = []

    for row in sessions.itertuples(index=False):
        if pd.isna(row.prev_close):
            continue

        intraday = df[df["session_date"] == row.session_date].reset_index(drop=True)
        if intraday.empty:
            continue

        up_idx = intraday["high"].idxmax()
        down_idx = intraday["low"].idxmin()

        up_row = intraday.iloc[up_idx]
        down_row = intraday.iloc[down_idx]

        up_move = (float(up_row["high"]) - float(row.prev_close)) / float(row.prev_close)
        down_move = (float(down_row["low"]) - float(row.prev_close)) / float(row.prev_close)

        if abs(up_move) >= abs(down_move):
            dominant_dir = "up"
            dominant_move = up_move
            dominant_abs = abs(up_move)
            dominant_time = str(up_row["time_msk"])
            dominant_from_open = (float(up_row["high"]) - float(row.open)) / float(row.open)
        else:
            dominant_dir = "down"
            dominant_move = down_move
            dominant_abs = abs(down_move)
            dominant_time = str(down_row["time_msk"])
            dominant_from_open = (float(down_row["low"]) - float(row.open)) / float(row.open)

        if dominant_abs < event_threshold:
            continue

        overnight_gap_abs = abs(float(row.overnight_gap))
        event_source = "overnight_gap" if overnight_gap_abs >= abs(dominant_from_open) else "intraday_shock"

        idx_prev_close = np.nan
        idx_event_move = np.nan
        idx_close_move = np.nan
        driver_type = "unknown"
        rel_event_move = np.nan
        rel_close_move = np.nan

        if row.session_date in idx_map.index:
            idx_prev_close = float(idx_map.loc[row.session_date, "prev_close"])
            idx_day = idx_df[idx_df["session_date"] == row.session_date].reset_index(drop=True)
            if not idx_day.empty and not pd.isna(idx_prev_close):
                idx_event_cut = idx_day[idx_day["time_msk"] <= dominant_time]
                if not idx_event_cut.empty:
                    idx_event_px = float(idx_event_cut.iloc[-1]["close"])
                    idx_event_move = (idx_event_px - idx_prev_close) / idx_prev_close
                idx_close_px = float(idx_day.iloc[-1]["close"])
                idx_close_move = (idx_close_px - idx_prev_close) / idx_prev_close

                rel_event_move = dominant_move - idx_event_move if not pd.isna(idx_event_move) else np.nan
                rel_close_move = float(row.day_close_move) - idx_close_move if not pd.isna(idx_close_move) else np.nan
                driver_type = classify_driver(dominant_abs, abs(idx_event_move) if not pd.isna(idx_event_move) else np.nan, market_ratio_threshold)

        rows.append(
            {
                "symbol": symbol,
                "session_date": str(row.session_date),
                "dominant_dir": dominant_dir,
                "dominant_move": dominant_move,
                "dominant_abs": dominant_abs,
                "dominant_time": dominant_time,
                "dominant_bucket": time_bucket(dominant_time),
                "overnight_gap": float(row.overnight_gap),
                "overnight_gap_abs": overnight_gap_abs,
                "dominant_from_open": dominant_from_open,
                "event_source": event_source,
                "day_close_move": float(row.day_close_move),
                "idx_event_move": idx_event_move,
                "idx_close_move": idx_close_move,
                "rel_event_move": rel_event_move,
                "rel_close_move": rel_close_move,
                "driver_type": driver_type,
                "bars_in_day": int(len(intraday)),
            }
        )

    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    def share_eq(series, value):
        return float((series == value).mean())

    per_symbol = (
        events.groupby("symbol", as_index=False)
        .agg(
            n=("symbol", "size"),
            dominant_abs_mean=("dominant_abs", "mean"),
            overnight_gap_abs_mean=("overnight_gap_abs", "mean"),
            intraday_shock_share=("event_source", lambda s: share_eq(s, "intraday_shock")),
            overnight_gap_share=("event_source", lambda s: share_eq(s, "overnight_gap")),
            before_12_share=("dominant_bucket", lambda s: share_eq(s, "before_12")),
            bucket_12_14_share=("dominant_bucket", lambda s: share_eq(s, "12_14")),
            after_14_share=("dominant_bucket", lambda s: share_eq(s, "after_14")),
            market_driven_share=("driver_type", lambda s: share_eq(s, "market_driven")),
            stock_specific_share=("driver_type", lambda s: share_eq(s, "stock_specific")),
            rel_event_move_mean=("rel_event_move", "mean"),
            rel_close_move_mean=("rel_close_move", "mean"),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    pooled = pd.DataFrame(
        {
            "symbol": ["POOLED"],
            "n": [len(events)],
            "dominant_abs_mean": [events["dominant_abs"].mean()],
            "overnight_gap_abs_mean": [events["overnight_gap_abs"].mean()],
            "intraday_shock_share": [float((events["event_source"] == "intraday_shock").mean())],
            "overnight_gap_share": [float((events["event_source"] == "overnight_gap").mean())],
            "before_12_share": [float((events["dominant_bucket"] == "before_12").mean())],
            "bucket_12_14_share": [float((events["dominant_bucket"] == "12_14").mean())],
            "after_14_share": [float((events["dominant_bucket"] == "after_14").mean())],
            "market_driven_share": [float((events["driver_type"] == "market_driven").mean())],
            "stock_specific_share": [float((events["driver_type"] == "stock_specific").mean())],
            "rel_event_move_mean": [events["rel_event_move"].mean()],
            "rel_close_move_mean": [events["rel_close_move"].mean()],
        }
    )

    return pd.concat([per_symbol, pooled], ignore_index=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_df = load_10min(args.index_symbol)

    all_events = []
    errors = []

    for symbol in args.symbols:
        try:
            events = analyze_symbol(
                symbol=symbol,
                idx_df=idx_df,
                event_threshold=args.event_threshold,
                market_ratio_threshold=args.market_ratio_threshold,
            )
            if events.empty:
                errors.append(f"{symbol}: нет событий по порогу {args.event_threshold:.2%}")
                print(f"[warn] {symbol}: нет событий")
                continue
            all_events.append(events)
            print(f"[ok] {symbol}: events={len(events)}")
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            print(f"[error] {symbol}: {exc}")

    if not all_events:
        (out_dir / "intraday_event_timing_errors.txt").write_text(
            "\n".join(errors) if errors else "Нет событий.", encoding="utf-8"
        )
        raise SystemExit("Не удалось собрать события. Подробности в research/out/intraday_event_timing_errors.txt")

    events = pd.concat(all_events, ignore_index=True)
    summary = summarize(events)

    events_path = out_dir / "intraday_event_timing_events.csv"
    summary_path = out_dir / "intraday_event_timing_summary.csv"

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    if errors:
        (out_dir / "intraday_event_timing_errors.txt").write_text("\n".join(errors), encoding="utf-8")

    print(f"[done] events -> {events_path}")
    print(f"[done] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
