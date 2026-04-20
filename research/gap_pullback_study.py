import argparse
from pathlib import Path
import pandas as pd
import numpy as np

TZ = 'Europe/Moscow'
DEFAULT_SYMBOLS = ['SBER', 'GAZP', 'LKOH']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Gap > X% pullback continuation event study')
    parser.add_argument('--symbols', nargs='*', default=DEFAULT_SYMBOLS)
    parser.add_argument('--min-gap-abs', type=float, default=0.05, help='Минимальный абсолютный gap, например 0.05 = 5%%')
    parser.add_argument('--pullback-thresholds', nargs='*', type=float, default=[0.0025, 0.005, 0.01], help='Пороги отката от открытия, например 0.0025 = 0.25%%')
    parser.add_argument('--trigger-search-bars', type=int, default=6, help='Сколько первых 10-мин баров искать откат')
    parser.add_argument('--entry-mode', choices=['next_open'], default='next_open')
    parser.add_argument('--out-dir', default='research/out')
    return parser.parse_args()


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Файл не найден: {path}')

    df = pd.read_csv(path)
    if 'begin' not in df.columns:
        df = pd.read_csv(path, header=None)
        if df.empty:
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
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['begin', 'open', 'high', 'low', 'close']).copy()
    df['dt_msk'] = df['begin'].dt.tz_convert(TZ)
    df['session_date'] = df['dt_msk'].dt.date
    df = df.sort_values('dt_msk').reset_index(drop=True)
    return df


def load_10min(symbol: str) -> pd.DataFrame:
    root = Path('.')
    candidates = [
        root / 'processed' / f'{symbol}_10min.csv',
        root / 'data' / f'{symbol}_10min.csv',
        root / 'processed' / f'{symbol}.csv',
        root / 'data' / f'{symbol}.csv',
    ]
    for path in candidates:
        if path.exists():
            return _read_ohlcv_csv(path)
    tried = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(f'Не найден 10-минутный файл для {symbol}. Проверены пути:\n{tried}')


def build_sessions(df_10m: pd.DataFrame) -> pd.DataFrame:
    sessions = (
        df_10m.groupby('session_date', as_index=False)
        .agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            bars=('close', 'size'),
        )
        .sort_values('session_date')
        .reset_index(drop=True)
    )
    sessions['prev_close'] = sessions['close'].shift(1)
    sessions['gap'] = (sessions['open'] - sessions['prev_close']) / sessions['prev_close']
    return sessions


def _directional_ret(entry_px: float, exit_px: float, gap_direction: str) -> float:
    raw = (exit_px - entry_px) / entry_px
    return raw if gap_direction == 'gap_up' else -raw


def analyze_symbol(symbol: str, min_gap_abs: float, pullback_thresholds: list[float], trigger_search_bars: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_10m = load_10min(symbol)
    sessions = build_sessions(df_10m)

    gap_rows = []
    trigger_rows = []

    for row in sessions.itertuples(index=False):
        if pd.isna(row.prev_close):
            continue
        if abs(row.gap) < min_gap_abs:
            continue

        intraday = df_10m[df_10m['session_date'] == row.session_date].reset_index(drop=True)
        if len(intraday) < 3:
            continue

        gap_direction = 'gap_up' if row.gap > 0 else 'gap_down'
        open_px = float(row.open)
        prev_close = float(row.prev_close)
        search = intraday.iloc[: min(len(intraday), max(1, trigger_search_bars))].copy()

        if gap_direction == 'gap_up':
            max_pullback_search = (open_px - search['low'].min()) / open_px
            day_fill = int((intraday['low'] <= prev_close).any())
        else:
            max_pullback_search = (search['high'].max() - open_px) / open_px
            day_fill = int((intraday['high'] >= prev_close).any())

        gap_rows.append({
            'symbol': symbol,
            'session_date': str(row.session_date),
            'gap': float(row.gap),
            'gap_abs': abs(float(row.gap)),
            'direction': gap_direction,
            'bars_in_day': int(len(intraday)),
            'open_px': open_px,
            'prev_close': prev_close,
            'max_pullback_search': max_pullback_search,
            'filled_same_day': day_fill,
        })

        for threshold in pullback_thresholds:
            trigger_idx = None
            trigger_dt = None
            for i, bar in search.iterrows():
                if gap_direction == 'gap_up':
                    pullback_hit = ((open_px - float(bar['low'])) / open_px) >= threshold
                else:
                    pullback_hit = ((float(bar['high']) - open_px) / open_px) >= threshold
                if pullback_hit:
                    trigger_idx = int(i)
                    trigger_dt = bar['dt_msk']
                    break

            hit = int(trigger_idx is not None)
            entry_idx = None
            entry_px = np.nan
            entry_dt = pd.NaT
            if hit and trigger_idx + 1 < len(intraday):
                entry_idx = trigger_idx + 1
                entry_px = float(intraday.iloc[entry_idx]['open'])
                entry_dt = intraday.iloc[entry_idx]['dt_msk']

            out = {
                'symbol': symbol,
                'session_date': str(row.session_date),
                'gap': float(row.gap),
                'gap_abs': abs(float(row.gap)),
                'direction': gap_direction,
                'pullback_threshold': threshold,
                'trigger_hit': hit,
                'trigger_bar_index': trigger_idx if hit else np.nan,
                'trigger_dt_msk': str(trigger_dt) if hit else '',
                'entry_bar_index': entry_idx if entry_idx is not None else np.nan,
                'entry_dt_msk': str(entry_dt) if entry_idx is not None else '',
                'open_px': open_px,
                'prev_close': prev_close,
                'entry_px': entry_px,
                'filled_same_day': day_fill,
                'max_pullback_search': max_pullback_search,
            }

            if entry_idx is not None:
                out['ret_1'] = _directional_ret(entry_px, float(intraday.iloc[entry_idx]['close']), gap_direction)
                for horizon in [1, 3, 6]:
                    idx = entry_idx + horizon
                    key = f'cont_{horizon}'
                    if idx < len(intraday):
                        out[key] = _directional_ret(entry_px, float(intraday.iloc[idx]['close']), gap_direction)
                    else:
                        out[key] = np.nan
                out['cont_close'] = _directional_ret(entry_px, float(intraday.iloc[-1]['close']), gap_direction)
                if gap_direction == 'gap_up':
                    out['mae_after_entry'] = (entry_px - float(intraday.iloc[entry_idx:]['low'].min())) / entry_px
                    out['mfe_after_entry'] = (float(intraday.iloc[entry_idx:]['high'].max()) - entry_px) / entry_px
                else:
                    out['mae_after_entry'] = (float(intraday.iloc[entry_idx:]['high'].max()) - entry_px) / entry_px
                    out['mfe_after_entry'] = (entry_px - float(intraday.iloc[entry_idx:]['low'].min())) / entry_px
            else:
                out['ret_1'] = np.nan
                out['cont_1'] = np.nan
                out['cont_3'] = np.nan
                out['cont_6'] = np.nan
                out['cont_close'] = np.nan
                out['mae_after_entry'] = np.nan
                out['mfe_after_entry'] = np.nan

            trigger_rows.append(out)

    gaps_df = pd.DataFrame(gap_rows)
    triggers_df = pd.DataFrame(trigger_rows)
    return gaps_df, triggers_df


def summarize_triggers(triggers: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame()

    def _share_pos(s: pd.Series) -> float:
        valid = s.dropna()
        if valid.empty:
            return np.nan
        return float((valid > 0).mean())

    summary = (
        triggers.groupby(['symbol', 'direction', 'pullback_threshold'], as_index=False)
        .agg(
            n_gaps=('gap', 'size'),
            trigger_hit_share=('trigger_hit', 'mean'),
            entry_count=('entry_px', lambda s: int(s.notna().sum())),
            cont_1_mean=('cont_1', 'mean'),
            cont_3_mean=('cont_3', 'mean'),
            cont_6_mean=('cont_6', 'mean'),
            cont_close_mean=('cont_close', 'mean'),
            cont_1_pos_share=('cont_1', _share_pos),
            cont_3_pos_share=('cont_3', _share_pos),
            cont_6_pos_share=('cont_6', _share_pos),
            cont_close_pos_share=('cont_close', _share_pos),
            mae_after_entry_mean=('mae_after_entry', 'mean'),
            mfe_after_entry_mean=('mfe_after_entry', 'mean'),
            filled_same_day_share=('filled_same_day', 'mean'),
            max_pullback_search_mean=('max_pullback_search', 'mean'),
        )
        .sort_values(['symbol', 'direction', 'pullback_threshold'])
        .reset_index(drop=True)
    )

    pooled = (
        triggers.groupby(['direction', 'pullback_threshold'], as_index=False)
        .agg(
            n_gaps=('gap', 'size'),
            trigger_hit_share=('trigger_hit', 'mean'),
            entry_count=('entry_px', lambda s: int(s.notna().sum())),
            cont_1_mean=('cont_1', 'mean'),
            cont_3_mean=('cont_3', 'mean'),
            cont_6_mean=('cont_6', 'mean'),
            cont_close_mean=('cont_close', 'mean'),
            cont_1_pos_share=('cont_1', _share_pos),
            cont_3_pos_share=('cont_3', _share_pos),
            cont_6_pos_share=('cont_6', _share_pos),
            cont_close_pos_share=('cont_close', _share_pos),
            mae_after_entry_mean=('mae_after_entry', 'mean'),
            mfe_after_entry_mean=('mfe_after_entry', 'mean'),
            filled_same_day_share=('filled_same_day', 'mean'),
            max_pullback_search_mean=('max_pullback_search', 'mean'),
        )
        .sort_values(['direction', 'pullback_threshold'])
        .reset_index(drop=True)
    )
    pooled.insert(0, 'symbol', 'POOLED')
    return pd.concat([summary, pooled], ignore_index=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gaps = []
    all_triggers = []
    errors = []

    for symbol in args.symbols:
        try:
            gaps_df, triggers_df = analyze_symbol(
                symbol=symbol,
                min_gap_abs=args.min_gap_abs,
                pullback_thresholds=args.pullback_thresholds,
                trigger_search_bars=args.trigger_search_bars,
            )
            if not gaps_df.empty:
                all_gaps.append(gaps_df)
            if not triggers_df.empty:
                all_triggers.append(triggers_df)
            print(f'[ok] {symbol}: gaps={len(gaps_df)} trigger_rows={len(triggers_df)}')
        except Exception as exc:
            errors.append(f'{symbol}: {exc}')
            print(f'[error] {symbol}: {exc}')

    if not all_gaps:
        (out_dir / 'gap_pullback_errors.txt').write_text('\n'.join(errors) if errors else 'Нет данных.', encoding='utf-8')
        raise SystemExit('Не удалось собрать события. См. research/out/gap_pullback_errors.txt')

    gaps = pd.concat(all_gaps, ignore_index=True)
    triggers = pd.concat(all_triggers, ignore_index=True) if all_triggers else pd.DataFrame()
    summary = summarize_triggers(triggers)

    gaps.to_csv(out_dir / 'gap_big_events.csv', index=False)
    triggers.to_csv(out_dir / 'gap_pullback_events.csv', index=False)
    summary.to_csv(out_dir / 'gap_pullback_summary.csv', index=False)
    if errors:
        (out_dir / 'gap_pullback_errors.txt').write_text('\n'.join(errors), encoding='utf-8')

    print(f'[done] gaps -> {out_dir / "gap_big_events.csv"}')
    print(f'[done] pullback events -> {out_dir / "gap_pullback_events.csv"}')
    print(f'[done] summary -> {out_dir / "gap_pullback_summary.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
