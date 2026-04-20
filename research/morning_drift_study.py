import argparse
from pathlib import Path
import pandas as pd
import numpy as np

TZ = 'Europe/Moscow'
ROOT = Path('.')
OUT_DIR = ROOT / 'research' / 'out'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Morning drift event study on 10-minute MOEX bars.')
    p.add_argument('--symbols', nargs='+', default=['SBER', 'GAZP', 'LKOH'])
    p.add_argument('--max-gap-abs', type=float, default=0.015, help='Max abs overnight gap to keep session in study.')
    p.add_argument('--start-time', default='10:00')
    p.add_argument('--fade-start', default='12:00')
    p.add_argument('--fade-end', default='14:00')
    p.add_argument('--out-prefix', default='morning_drift')
    return p.parse_args()


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Файл не найден: {path}')
    df = pd.read_csv(path)
    if 'begin' not in df.columns:
        df = pd.read_csv(path, header=None)
        if len(df) == 0:
            raise ValueError(f'Пустой файл: {path}')
        header = [str(x).strip() for x in df.iloc[0].tolist()]
        df = df.iloc[1:].copy()
        df.columns = header
    required = {'begin', 'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'В файле {path} отсутствуют колонки: {sorted(missing)}')
    df = df[['begin', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['begin'] = pd.to_datetime(df['begin'], utc=True, errors='coerce')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['begin', 'open', 'high', 'low', 'close']).copy()
    df['dt_msk'] = df['begin'].dt.tz_convert(TZ)
    df['session_date'] = df['dt_msk'].dt.date
    df['time_str'] = df['dt_msk'].dt.strftime('%H:%M')
    df = df.sort_values('dt_msk').reset_index(drop=True)
    return df


def load_10m(symbol: str) -> pd.DataFrame:
    candidates = [
        ROOT / 'processed' / f'{symbol}_10min.csv',
        ROOT / 'data' / f'{symbol}_10min.csv',
        ROOT / 'processed' / f'{symbol}.csv',
        ROOT / 'data' / f'{symbol}.csv',
    ]
    for p in candidates:
        if p.exists():
            return _read_ohlcv_csv(p)
    tried = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(f'Не найден 10-минутный файл для {symbol}. Проверены пути:\n{tried}')


def build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    sessions = (
        df.groupby('session_date', as_index=False)
        .agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            bars=('close', 'size'),
            first_dt=('dt_msk', 'first'),
            last_dt=('dt_msk', 'last'),
        )
        .sort_values('session_date')
        .reset_index(drop=True)
    )
    sessions['prev_close'] = sessions['close'].shift(1)
    sessions['gap'] = (sessions['open'] - sessions['prev_close']) / sessions['prev_close']
    return sessions


def _close_at_or_after(intraday: pd.DataFrame, hhmm: str):
    hit = intraday[intraday['time_str'] >= hhmm]
    if hit.empty:
        return np.nan
    return float(hit.iloc[0]['close'])


def analyze_symbol(symbol: str, max_gap_abs: float, start_time: str, fade_start: str, fade_end: str) -> pd.DataFrame:
    df = load_10m(symbol)
    sessions = build_sessions(df)
    rows = []

    for row in sessions.itertuples(index=False):
        if pd.isna(row.prev_close):
            continue
        if abs(row.gap) > max_gap_abs:
            continue

        intraday = df[df['session_date'] == row.session_date].reset_index(drop=True)
        intraday = intraday[intraday['time_str'] >= start_time].reset_index(drop=True)
        if intraday.empty:
            continue

        open_px = float(intraday.iloc[0]['open'])
        close_30 = _close_at_or_after(intraday, '10:30')
        close_60 = _close_at_or_after(intraday, '11:00')
        close_90 = _close_at_or_after(intraday, '11:30')
        close_1200 = _close_at_or_after(intraday, fade_start)
        close_1400 = _close_at_or_after(intraday, fade_end)
        close_eod = float(intraday.iloc[-1]['close'])

        def ret(px):
            return np.nan if pd.isna(px) else (px - open_px) / open_px

        morning_up = ret(close_90)
        fade_12_14 = np.nan
        if not pd.isna(close_1200) and not pd.isna(close_1400):
            fade_12_14 = (close_1400 - close_1200) / close_1200

        rows.append({
            'symbol': symbol,
            'session_date': str(row.session_date),
            'overnight_gap': float(row.gap),
            'gap_abs': abs(float(row.gap)),
            'open_px': open_px,
            'ret_30m': ret(close_30),
            'ret_60m': ret(close_60),
            'ret_90m': morning_up,
            'ret_1200': ret(close_1200),
            'ret_1400': ret(close_1400),
            'ret_close': ret(close_eod),
            'fade_12_14': fade_12_14,
            'morning_positive_90m': int((not pd.isna(morning_up)) and (morning_up > 0)),
            'fade_negative_12_14': int((not pd.isna(fade_12_14)) and (fade_12_14 < 0)),
            'pattern_morning_up_then_fade': int((not pd.isna(morning_up)) and (morning_up > 0) and (not pd.isna(fade_12_14)) and (fade_12_14 < 0)),
        })
    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    agg = dict(
        n=('symbol', 'size'),
        gap_abs_mean=('gap_abs', 'mean'),
        ret_30m_mean=('ret_30m', 'mean'),
        ret_60m_mean=('ret_60m', 'mean'),
        ret_90m_mean=('ret_90m', 'mean'),
        ret_1200_mean=('ret_1200', 'mean'),
        ret_1400_mean=('ret_1400', 'mean'),
        ret_close_mean=('ret_close', 'mean'),
        fade_12_14_mean=('fade_12_14', 'mean'),
        morning_positive_90m_share=('morning_positive_90m', 'mean'),
        fade_negative_12_14_share=('fade_negative_12_14', 'mean'),
        pattern_morning_up_then_fade_share=('pattern_morning_up_then_fade', 'mean'),
        ret_90m_median=('ret_90m', 'median'),
        fade_12_14_median=('fade_12_14', 'median'),
        ret_close_median=('ret_close', 'median'),
    )
    by_symbol = events.groupby('symbol', as_index=False).agg(**agg)
    pooled = events.groupby(lambda _: True).agg(**agg).reset_index(drop=True)
    pooled.insert(0, 'symbol', 'POOLED')
    return pd.concat([by_symbol, pooled], ignore_index=True)


def main():
    args = parse_args()
    all_events = []
    errors = []
    for symbol in args.symbols:
        try:
            ev = analyze_symbol(symbol, args.max_gap_abs, args.start_time, args.fade_start, args.fade_end)
            if ev.empty:
                errors.append(f'{symbol}: нет сессий после фильтра max_gap_abs={args.max_gap_abs:.2%}')
                continue
            all_events.append(ev)
            print(f'[ok] {symbol}: sessions={len(ev)}')
        except Exception as exc:
            errors.append(f'{symbol}: {exc}')
            print(f'[error] {symbol}: {exc}')
    if not all_events:
        (OUT_DIR / f'{args.out_prefix}_errors.txt').write_text('\n'.join(errors) if errors else 'Нет данных.', encoding='utf-8')
        raise SystemExit('Не удалось собрать ни одной сессии. См. research/out/*errors.txt')
    events = pd.concat(all_events, ignore_index=True)
    summary = summarize(events)
    events_path = OUT_DIR / f'{args.out_prefix}_events.csv'
    summary_path = OUT_DIR / f'{args.out_prefix}_summary.csv'
    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    if errors:
        (OUT_DIR / f'{args.out_prefix}_errors.txt').write_text('\n'.join(errors), encoding='utf-8')
    print(f'[done] events -> {events_path}')
    print(f'[done] summary -> {summary_path}')


if __name__ == '__main__':
    main()
