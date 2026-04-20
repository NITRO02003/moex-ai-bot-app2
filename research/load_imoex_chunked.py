
import argparse
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUT = DATA_DIR / "IMOEX_10min.csv"

BASE_URL = "https://iss.moex.com/iss/engines/stock/markets/index/securities/{symbol}/candles.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Надежная загрузка 10-минутных свечей индекса MOEX через ISS с чанками по годам.")
    p.add_argument("--symbol", default="IMOEX", help="Тикер индекса. По умолчанию IMOEX.")
    p.add_argument("--start-year", type=int, default=2020, help="Первый год выгрузки.")
    p.add_argument("--end-year", type=int, default=2026, help="Последний год выгрузки включительно.")
    p.add_argument("--interval", type=int, default=10, help="Интервал свечей ISS. Для 10 минут оставить 10.")
    p.add_argument("--timeout", type=float, default=20.0, help="Timeout одного HTTP-запроса.")
    p.add_argument("--retries", type=int, default=5, help="Число повторов на один запрос.")
    p.add_argument("--retry-sleep", type=float, default=3.0, help="Пауза между повторами.")
    p.add_argument("--page-sleep", type=float, default=0.2, help="Пауза между страницами ISS.")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Куда сохранить итоговый CSV.")
    p.add_argument("--append", action="store_true", help="Если файл уже есть, добавить новые строки и удалить дубли по begin.")
    return p.parse_args()


def year_ranges(start_year: int, end_year: int) -> list[tuple[str, str]]:
    if end_year < start_year:
        raise ValueError("end-year не может быть меньше start-year")
    ranges: list[tuple[str, str]] = []
    for year in range(start_year, end_year + 1):
        date_from = f"{year}-01-01"
        date_till = f"{year}-12-31"
        ranges.append((date_from, date_till))
    return ranges


def request_json(session: requests.Session, url: str, params: dict, timeout: float, retries: int, retry_sleep: float) -> dict:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            print(f"[retry] attempt={attempt}/{retries} start={params.get('start', 0)} from={params.get('from')} till={params.get('till')} err={exc}")
            time.sleep(retry_sleep)
    raise RuntimeError(f"Не удалось получить ответ ISS после {retries} попыток: {last_error}") from last_error


def fetch_range(session: requests.Session, symbol: str, date_from: str, date_till: str, interval: int, timeout: float, retries: int, retry_sleep: float, page_sleep: float) -> pd.DataFrame:
    url = BASE_URL.format(symbol=symbol)
    all_rows = []
    cols = None
    start = 0

    while True:
        params = {
            "interval": interval,
            "from": date_from,
            "till": date_till,
            "start": start,
        }
        payload = request_json(session, url, params, timeout, retries, retry_sleep)
        candles = payload.get("candles", {})
        data = candles.get("data", [])
        cols = candles.get("columns", cols)

        if not data:
            break

        all_rows.extend(data)
        start += len(data)
        print(f"[page] {symbol} {date_from}..{date_till} rows_total={len(all_rows)} next_start={start}")
        time.sleep(page_sleep)

    if not all_rows:
        return pd.DataFrame()

    if not cols:
        raise RuntimeError("ISS вернул данные без columns")

    df = pd.DataFrame(all_rows, columns=cols)
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "begin" not in df.columns:
        raise ValueError("В ответе ISS отсутствует колонка begin")

    for col in ["open", "close", "high", "low", "value", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["begin"] = pd.to_datetime(df["begin"], utc=True, errors="coerce")
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")

    df = df.dropna(subset=["begin", "open", "high", "low", "close"]).copy()
    df = df.sort_values("begin").drop_duplicates(subset=["begin"]).reset_index(drop=True)
    return df


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "begin" not in df.columns:
        raise ValueError(f"Существующий файл {path} не содержит begin")
    df["begin"] = pd.to_datetime(df["begin"], utc=True, errors="coerce")
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.sort_values("begin").drop_duplicates(subset=["begin"]).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    # ISO format for compatibility with existing research readers.
    df_to_save["begin"] = df_to_save["begin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "end" in df_to_save.columns:
        df_to_save["end"] = df_to_save["end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_to_save.to_csv(out_path, index=False)


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    session = requests.Session()
    session.headers.update({"User-Agent": "moex-ai-bot-research-loader/1.0"})

    chunks = year_ranges(args.start_year, args.end_year)
    frames = []

    for date_from, date_till in chunks:
        print(f"[range] {args.symbol} {date_from}..{date_till}")
        df_chunk = fetch_range(
            session=session,
            symbol=args.symbol,
            date_from=date_from,
            date_till=date_till,
            interval=args.interval,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            page_sleep=args.page_sleep,
        )
        if df_chunk.empty:
            print(f"[warn] empty range {date_from}..{date_till}")
            continue
        df_chunk = normalize(df_chunk)
        print(f"[ok] {date_from}..{date_till} rows={len(df_chunk)}")
        frames.append(df_chunk)

    if not frames:
        raise SystemExit("ISS не вернул ни одной свечи. Проверь тикер, интервал или доступность ISS.")

    result = pd.concat(frames, ignore_index=True)
    result = normalize(result)

    if args.append and out_path.exists():
        existing = load_existing(out_path)
        result = pd.concat([existing, result], ignore_index=True)
        result = normalize(result)

    save_csv(result, out_path)
    print(f"[done] saved rows={len(result)} path={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
