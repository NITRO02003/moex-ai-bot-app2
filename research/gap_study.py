
import os
from pathlib import Path
import pandas as pd
import numpy as np

SYMBOLS = ["SBER", "GAZP", "LKOH"]
GAP_THRESHOLD = 0.015
TZ = "Europe/Moscow"

ROOT = Path(".")
OUT_DIR = ROOT / "research" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path)

    # Защита на случай, если заголовок попал в первую строку данных.
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
    df = df.sort_values("dt_msk").reset_index(drop=True)
    return df


def load_10min(symbol: str) -> pd.DataFrame:
    candidates = [
        ROOT / "processed" / f"{symbol}_10min.csv",
        ROOT / "data" / f"{symbol}_10min.csv",
        ROOT / "processed" / f"{symbol}.csv",
        ROOT / "data" / f"{symbol}.csv",
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
    sessions["gap"] = (sessions["open"] - sessions["prev_close"]) / sessions["prev_close"]
    return sessions


def analyze_symbol(symbol: str) -> pd.DataFrame:
    df_10m = load_10min(symbol)
    sessions = build_sessions(df_10m)

    event_rows = []

    for row in sessions.itertuples(index=False):
        if pd.isna(row.prev_close):
            continue
        if abs(row.gap) < GAP_THRESHOLD:
            continue

        intraday = df_10m[df_10m["session_date"] == row.session_date].reset_index(drop=True)
        if intraday.empty:
            continue

        open_px = float(row.open)
        prev_close = float(row.prev_close)
        gap = float(row.gap)
        direction = "gap_up" if gap > 0 else "gap_down"

        def close_ret_after(n_bars: int) -> float:
            if len(intraday) <= n_bars:
                return np.nan
            px = float(intraday.iloc[n_bars]["close"])
            return (px - open_px) / open_px

        # Возврат против гэпа: положительный знак означает движение в сторону закрытия гэпа.
        def reversion_ret_after(n_bars: int) -> float:
            raw = close_ret_after(n_bars)
            if pd.isna(raw):
                return np.nan
            return -raw if gap > 0 else raw

        if gap > 0:
            filled_mask = intraday["low"] <= prev_close
            max_reversion = (open_px - intraday["low"].min()) / open_px
        else:
            filled_mask = intraday["high"] >= prev_close
            max_reversion = (intraday["high"].max() - open_px) / open_px

        fill_rate_day = int(bool(filled_mask.any()))
        fill_bar_index = int(filled_mask.idxmax()) if fill_rate_day else np.nan

        event_rows.append(
            {
                "symbol": symbol,
                "session_date": str(row.session_date),
                "gap": gap,
                "gap_abs": abs(gap),
                "direction": direction,
                "bars_in_day": int(len(intraday)),
                "open_px": open_px,
                "prev_close": prev_close,
                "ret_1": close_ret_after(1),
                "ret_3": close_ret_after(3),
                "ret_6": close_ret_after(6),
                "ret_close": (float(intraday.iloc[-1]["close"]) - open_px) / open_px,
                "reversion_1": reversion_ret_after(1),
                "reversion_3": reversion_ret_after(3),
                "reversion_6": reversion_ret_after(6),
                "reversion_close": (-((float(intraday.iloc[-1]["close"]) - open_px) / open_px) if gap > 0 else ((float(intraday.iloc[-1]["close"]) - open_px) / open_px)),
                "filled_same_day": fill_rate_day,
                "fill_bar_index": fill_bar_index,
                "max_reversion": max_reversion,
            }
        )

    return pd.DataFrame(event_rows)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    summary = (
        events.groupby(["symbol", "direction"], as_index=False)
        .agg(
            n=("gap", "size"),
            gap_abs_mean=("gap_abs", "mean"),
            reversion_1_mean=("reversion_1", "mean"),
            reversion_3_mean=("reversion_3", "mean"),
            reversion_6_mean=("reversion_6", "mean"),
            reversion_close_mean=("reversion_close", "mean"),
            reversion_1_median=("reversion_1", "median"),
            reversion_3_median=("reversion_3", "median"),
            reversion_6_median=("reversion_6", "median"),
            reversion_close_median=("reversion_close", "median"),
            positive_rev_1_share=("reversion_1", lambda s: float((s > 0).mean())),
            positive_rev_3_share=("reversion_3", lambda s: float((s > 0).mean())),
            positive_rev_6_share=("reversion_6", lambda s: float((s > 0).mean())),
            positive_rev_close_share=("reversion_close", lambda s: float((s > 0).mean())),
            filled_same_day_share=("filled_same_day", "mean"),
            max_reversion_mean=("max_reversion", "mean"),
        )
        .sort_values(["symbol", "direction"])
        .reset_index(drop=True)
    )

    pooled = (
        events.groupby(["direction"], as_index=False)
        .agg(
            n=("gap", "size"),
            gap_abs_mean=("gap_abs", "mean"),
            reversion_1_mean=("reversion_1", "mean"),
            reversion_3_mean=("reversion_3", "mean"),
            reversion_6_mean=("reversion_6", "mean"),
            reversion_close_mean=("reversion_close", "mean"),
            reversion_1_median=("reversion_1", "median"),
            reversion_3_median=("reversion_3", "median"),
            reversion_6_median=("reversion_6", "median"),
            reversion_close_median=("reversion_close", "median"),
            positive_rev_1_share=("reversion_1", lambda s: float((s > 0).mean())),
            positive_rev_3_share=("reversion_3", lambda s: float((s > 0).mean())),
            positive_rev_6_share=("reversion_6", lambda s: float((s > 0).mean())),
            positive_rev_close_share=("reversion_close", lambda s: float((s > 0).mean())),
            filled_same_day_share=("filled_same_day", "mean"),
            max_reversion_mean=("max_reversion", "mean"),
        )
        .sort_values(["direction"])
        .reset_index(drop=True)
    )
    pooled.insert(0, "symbol", "POOLED")
    return pd.concat([summary, pooled], ignore_index=True)


def main() -> int:
    all_events = []
    errors = []

    for symbol in SYMBOLS:
        try:
            events = analyze_symbol(symbol)
            if events.empty:
                errors.append(f"{symbol}: нет событий gap по текущему порогу {GAP_THRESHOLD:.2%}")
                continue
            all_events.append(events)
            print(f"[ok] {symbol}: events={len(events)}")
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            print(f"[error] {symbol}: {exc}")

    if not all_events:
        errors_path = OUT_DIR / "gap_errors.txt"
        errors_path.write_text("\n".join(errors) if errors else "Нет данных для анализа.", encoding="utf-8")
        raise SystemExit("Не удалось собрать ни одного события. Подробности в research/out/gap_errors.txt")

    events = pd.concat(all_events, ignore_index=True)
    summary = summarize_events(events)

    events_path = OUT_DIR / "gap_events.csv"
    summary_path = OUT_DIR / "gap_summary.csv"
    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    if errors:
        (OUT_DIR / "gap_errors.txt").write_text("\n".join(errors), encoding="utf-8")

    print(f"[done] events -> {events_path}")
    print(f"[done] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
