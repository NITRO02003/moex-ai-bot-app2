from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def _interval_to_freq(interval: str) -> str:
    """
    Преобразует человекочитаемый интервал в частоту pandas.

    Мы договорились использовать форматы:
      - '10min', '15min', '30min', ...
      - '1h', '4h', ...

    Они уже совместимы с pandas.resample, поэтому достаточно
    нормализовать регистр и пробелы и НЕ использовать 'T'/'H'.
    """
    return interval.strip().lower()


def _load_raw_symbol(path: str) -> Optional[pd.DataFrame]:
    """
    Загружает сырые данные по тикеру из CSV.

    Ожидает хотя бы одну из колонок времени:
      - 'datetime'
      - 'begin'

    Приводит к индексу по времени и оставляет основные поля:
      - open, high, low, close, volume (если есть).
    """
    if not os.path.exists(path):
        print(f"[process-data] file not found: {path}")
        return None

    df = pd.read_csv(path)

    # ищем колонку времени
    dt_col = None
    if "datetime" in df.columns:
        dt_col = "datetime"
    elif "begin" in df.columns:
        dt_col = "begin"

    if dt_col is None:
        print(f"[process-data] no datetime/begin column in {path}, skip")
        return None

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col)
    df = df.set_index(dt_col)

    # оставляем только нужные колонки, если они есть
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not cols:
        print(f"[process-data] no OHLC/volume columns in {path}, skip")
        return None

    return df[cols]


def _check_aggregated(
    symbol: str,
    interval: str,
    ohlc: pd.DataFrame,
    rows_in: int,
    rows_out: int,
) -> None:
    """
    Быстрая проверка качества агрегированных данных.

    Проверяем:
      - отсортированность по времени,
      - дубликаты временных меток,
      - базовую консистентность OHLC (low <= open/close <= high),
      - отрицательные объёмы,
      - подозрительное отношение rows_out / rows_in.
    """
    issues: List[str] = []

    if rows_out == 0:
        issues.append("empty_output")

    # проверка сортировки и дубликатов
    if not ohlc["begin"].is_monotonic_increasing:
        issues.append("not_sorted")
    if ohlc["begin"].duplicated().any():
        issues.append("duplicate_timestamps")

    bad_low_high = bad_open = bad_close = bad_volume = 0

    # проверка OHLC
    required_ohlc = {"open", "high", "low", "close"}
    if required_ohlc.issubset(ohlc.columns):
        bad_low_high = int((ohlc["low"] > ohlc["high"]).sum())
        bad_open = int(
            ((ohlc["open"] < ohlc["low"]) | (ohlc["open"] > ohlc["high"])).sum()
        )
        bad_close = int(
            ((ohlc["close"] < ohlc["low"]) | (ohlc["close"] > ohlc["high"])).sum()
        )
        if bad_low_high or bad_open or bad_close:
            issues.append("ohlc_inconsistency")

    # проверка объёмов
    if "volume" in ohlc.columns:
        bad_volume = int((ohlc["volume"] < 0).sum())
        if bad_volume:
            issues.append("negative_volume")

    ratio = None
    if rows_in:
        ratio = rows_out / rows_in
        # очень грубый фильтр "совсем странно"
        if ratio > 10:
            issues.append("rows_out/rows_in_ratio_suspicious")

    if issues:
        print(
            f"[process-data-check] {symbol} {interval}: issues={issues}, "
            f"rows_in={rows_in}, rows_out={rows_out}, "
            f"bad_low>high={bad_low_high}, bad_open={bad_open}, "
            f"bad_close={bad_close}, bad_volume={bad_volume}, ratio={ratio}"
        )
    else:
        print(
            f"[process-data-check] {symbol} {interval}: OK, "
            f"rows_in={rows_in}, rows_out={rows_out}"
        )


def _aggregate_symbol(
    symbol: str,
    intervals: List[str],
    input_dir: str,
    output_dir: str,
) -> Dict[str, Dict[str, int]]:
    """
    Агрегирует один тикер по набору интервалов.

    Возвращает словарь вида:
      { '10min': {'rows_in': N, 'rows_out': M}, '30min': {...}, ... }
    """
    in_path = os.path.join(input_dir, f"{symbol}.csv")
    df_raw = _load_raw_symbol(in_path)

    if df_raw is None:
        return {}

    rows_in = len(df_raw)
    summary: Dict[str, Dict[str, int]] = {}

    for interval in intervals:
        freq = _interval_to_freq(interval)

        # OHLC агрегация
        ohlc = df_raw[["open", "high", "low", "close"]].resample(freq).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )

        # объём — сумма
        if "volume" in df_raw.columns:
            vol = df_raw["volume"].resample(freq).sum()
            ohlc["volume"] = vol

        # удаляем полностью пустые бары
        ohlc = ohlc.dropna(how="all")
        rows_out = len(ohlc)

        # приводим индекс к колонке 'begin'
        ohlc = ohlc.reset_index()
        time_col = ohlc.columns[0]
        ohlc = ohlc.rename(columns={time_col: "begin"})

        out_fname = f"{symbol}_{interval}.csv"
        out_path = os.path.join(output_dir, out_fname)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ohlc.to_csv(out_path, index=False)

        print(
            f"[process-data] {symbol} {interval}: "
            f"rows_in={rows_in}, rows_out={rows_out}, saved to {out_path}"
        )

        # встроенная проверка корректности агрегата
        _check_aggregated(symbol, interval, ohlc, rows_in, rows_out)

        summary[interval] = {"rows_in": rows_in, "rows_out": rows_out}

    return summary


def run_data_processing(
    symbols: List[str],
    intervals: List[str],
    input_dir: str,
    output_dir: str,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Основная функция агрегации данных.

    - symbols: список тикеров
    - intervals: список интервалов ('10min', '30min', '1h', ...)
    - input_dir: каталог сырых данных (обычно 'data')
    - output_dir: каталог агрегированных данных (обычно 'processed')
    - out: путь к JSON-отчёту (может быть None)
    """
    print(
        f"[process-data] start, symbols={symbols}, intervals={intervals}, "
        f"input_dir={input_dir}, output_dir={output_dir}"
    )

    summary: Dict[str, Any] = {}

    for sym in symbols:
        sym_summary = _aggregate_symbol(sym, intervals, input_dir, output_dir)
        if sym_summary:
            summary[sym] = sym_summary

    print(f"[process-data] done, symbols_processed={len(summary)}")

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[process-data] summary saved to {out_path}")

    return summary


def main(args) -> Dict[str, Any]:
    """
    Обёртка под CLI (app2.cli process-data).

    Ожидает, что в args есть:
      - symbols
      - intervals
      - input_dir
      - output_dir
      - out (может быть None)
    """
    return run_data_processing(
        symbols=args.symbols,
        intervals=args.intervals,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        out=args.out,
    )
