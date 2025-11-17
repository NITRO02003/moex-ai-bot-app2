0. Базовая архитектура и структура проекта

[P1] Привести файловую структуру к стандарту

Убедиться, что:

data/ – только сырые выгрузки с MOEX (1m/10m и т.п.).

processed/ – агрегированные таймфреймы: TICKER_10min.csv, TICKER_30min.csv, TICKER_1h.csv.

out/ – все результаты (*.json, *.csv, отчёты, свипы, forward-test).

Внутри app2/ не должно быть своих data/, out/ и т.п.
→ Пройтись по проекту и удалить/перенести любые app2/data, app2/out, временные каталоги.

[P2] Чистка старого кода

Найти и удалить/архивировать:

старые версии cli_old.py, *_backup.py, дубляжи param_sweep_old.py, rule_backtest_old.py, и т.п.;

неиспользуемые модули из первой версии APP, которые не вызываются из cli.py и не используются в app3.

Свести всё “боевое” ядро к:

cli.py, rule_core.py, rule_strategies.py, rule_backtest.py,
regime_detector.py, regime_rule_backtest.py,
param_sweep.py, data_pipeline.py, forward_test.py,
config.py, config.json, utils.py.

1. CLI (app2/cli.py)

[P1] Гарантировать, что каждая команда реально вызывает код

Для каждой команды:

rule-backtest → cmd_rule_backtest() → rule_backtest.main(args)

regime-rule-backtest → cmd_regime_rule_backtest() → regime_rule_backtest.main(args)

process-data → cmd_process_data() → data_pipeline.main(args) / run_data_processing

forward-test → cmd_forward_test() → forward_test.main(args) / run_forward_test

param-sweep → cmd_param_sweep() → param_sweep.run_sweep(...)

detect-regime → cmd_detect_regime() → regime_detector.detect_regime(...) + regime_distribution

В конце main():

args = parser.parse_args()
if not hasattr(args, "func"):
    parser.print_help()
    return
args.func(args)


[P2] Добавить --n-jobs в CLI (единый стиль)

Для “тяжёлых” команд:

param-sweep

forward-test

при желании – rule-backtest (когда торговля идёт по многим тикерам одновременно)

Везде аргумент:

p_sweep.add_argument("--n-jobs", type=int, default=-1,
                     help="Число процессов (-1 = все ядра)")


cli.py НЕ реализует multiprocessing, он только пробрасывает n_jobs в соответствующий модуль.

2. Конфиг и “магические числа”

[P1] Централизовать всё в config.json

Проверить, что в config.json есть:

defaults.RegimeParams

defaults.TrendParams

defaults.MeanRevParams

defaults.BreakoutParams

defaults.RuleBtParams

sweep.MeanRevParams (уже есть)

в будущем: sweep.TrendParams, sweep.RuleBtParams и т.д.

Все значения типа:

risk_per_trade, atr_len, sl_atr_mult, tp_mult,

trend_thr, min_gap_bars,

high_vol_quantile, trend_threshold
должны браться только из config.

[P2] config.py

Убедиться, что load_config(path="app2/config.json"):

import json
from pathlib import Path

def load_config(path: str = "app2/config.json"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

3. rule_core.py и rule_backtest.py

[P1] Одна реализация ядра бэктеста

В rule_core.py:

@dataclass RuleBtParams с полями:

commission, slippage_bps,
risk_per_trade, atr_len, sl_atr_mult, tp_mult и т.п.

run_rule_symbol(df, params, equity0):

принимает df с колонками как минимум: datetime, close, signal;

строит позицию (position), PnL, equity;

возвращает {"equity_curve": ..., "metrics": {...}}.

Запретить альтернативные run_rule_symbol в других файлах (во rule_backtest.py и др. них должно быть только использование ядра).

[P2] Метрики

В metrics вернуть:

total_return

max_drawdown

calmar

volatility_ann

sharpe_ann

profit_factor

win_rate

trade_count

avg_trade

Все стратегические отчёты (backtest/sweep/forward-test) используют один и тот же формат метрик.

[P2] Логгирование

При запуске бэктеста:

print(f"[rule-backtest] symbol={sym}, strategy={strategy}, equity0={equity0}")
print(f"[rule-backtest] params={params}")


В конце:

print(f"[rule-backtest] symbol={sym}, total_return={metrics['total_return']:.4f}, "
      f"max_dd={metrics['max_drawdown']:.4f}, trades={metrics['trade_count']}")

4. rule_strategies.py

[P1] Согласовать сигнатуры с конфигом и свипом

Проверить dataclass'ы:

TrendParams

MeanRevParams

BreakoutParams

Убедиться, что:

MeanRevParams содержит

rsi_len, rsi_low, rsi_high

bb_len (или boll_window)

bb_k (или boll_mult)

min_gap_bars

Функции:

def generate_trend_signals(df: pd.DataFrame, params: TrendParams) -> pd.Series: ...
def generate_meanrev_signals(df: pd.DataFrame, params: MeanRevParams) -> pd.Series: ...
def generate_breakout_signals(df: pd.DataFrame, params: BreakoutParams) -> pd.Series: ...


либо принимают именно params (объект), либо чётко задокументирован список **kwargs.

[P2] Логгирование в стратегиях

При генерации сигналов можно логировать только в свипе и бэктестах. В самих стратегиях лучше не спамить.
Но:

добавить docstring, описывающий, что именно делает стратегия;

убедиться, что серия signal:

индексирована так же, как df,

принимает значения -1/0/+1.

5. Режимы рынка (regime_detector.py, regime_rule_backtest.py)

[P1] detect_regime и regime_distribution

detect_regime(df, regime_params):

использует:

ATR (относительный к цене),

EMA_fast, EMA_slow,

пороги high_vol_quantile и trend_threshold из RegimeParams.

добавляет колонку regime со значениями trend / range / high_vol.

regime_distribution(regime_series):

возвращает dict с долей (в %) по каждому режиму;

используется и в detect-regime CLI, и в отчётах.

[P1] detect-regime CLI

Работает по processed/{SYM}_{interval}.csv.

Если есть колонка begin – переименовывает в datetime.

Логирует:

print(f"[detect-regime] {sym}: trend={trend:.2f}%, range={range_:.2f}%, high_vol={hv:.2f}%")


Сохраняет общий JSON в out/regime_distribution.json.

[P2] regime_rule_backtest

В run_regime_rule_symbol:

заранее считает regime на всём df;

считает три набора сигналов (trend / meanrev / breakout);

собирает итоговый signal по режимам:

trend → trend-сигналы,

range → meanrev-сигналы,

high_vol → breakout, если включён, иначе 0 (нет позы).

В main(args):

читает данные из processed/,

логирует, какие режимы и процент времени по каждому тикеру,

вызывает run_rule_symbol(df, params, equity0).

6. param_sweep.py (включая multiprocessing)

[P1] Привести run_sweep к стабильной сигнатуре

Используем именно ту сигнатуру, которую ожидает cli.py (мы её уже почти сделали):

def run_sweep(
    strategy: str,
    config_path: str,
    csv_path: str,
    symbols: List[str],
    equity0: float = 1_000_000.0,
    use_breakout_in_high_vol: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    ...


[P1] Логгирование

В начале:

print(f"[sweep] strategy={strategy}, symbols={symbols}, equity0={equity0}, n_jobs={n_jobs}")
print(f"[sweep] config={config_path}, out={csv_path}")


В _run_meanrev_sweep:

print(f"[sweep-meanrev] symbols={symbols}")

print(f"[sweep-meanrev] grid keys={list(grid_cfg.keys())}")

print(f"[sweep-meanrev] total combinations={len(combos)}")

каждые N комбинаций:

if idx % 50 == 0:
    print(f"[sweep-meanrev] combo {idx}/{len(combos)}: {param_set}")


В конце:

print(f"[sweep] done, rows={len(df)}, saved to {csv_path}")


[P1] Причина пустого CSV — явно проверять

Если после прохода rows пуст:

if not rows:
    print("[sweep-meanrev] WARNING: no rows collected (no data or all combos invalid)")


→ это сразу покажет, что свип прошёл “вхолостую”.

[P2] Мультипроцессинг

Добавить обработку n_jobs:

если n_jobs == 1 → текущая логика (однопоточная).

если n_jobs != 1:

использовать ProcessPoolExecutor и параллелить либо по тикерам, либо по парам (sym, param_set) (это нужно аккуратно спроектировать, но правило – делаем).

Важно: при multiprocessing:

избегать глобальных объектов,

следить за сериализуемостью параметров (dataclass ок).

7. forward_test.py

[P2] Сигнатура и вызов

Ядро:

def run_forward_test(
    strategy: str,
    symbols: List[str],
    interval: str,
    train_window: int,
    test_window: int,
    step: int,
    equity0: float,
    out_path: str,
    use_breakout_in_high_vol: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    ...


CLI forward-test пробрасывает все аргументы, включая --n-jobs.

[P2] Логгирование

На старте:

print(f"[forward-test] strategy={strategy}, symbols={symbols}, interval={interval}")
print(f"[forward-test] train={train_window}, test={test_window}, step={step}, n_jobs={n_jobs}")


По каждому символу и окну (хотя бы раз в N окон) логировать прогресс.

В конце:

print(f"[forward-test] done, windows={total_windows}, out={out_path}")


[P3] Мультипроцессинг

n_jobs != 1 → параллелить:

либо по тикерам,

либо по окнам (если данных много).

8. data_pipeline.py и качество данных

[P2] Стандартизировать колонки и пути

На входе из data/:

читаем сырые CSV из data/TICKER.csv.

На выходе в processed/:

гарантируем колонки:

datetime

open, high, low, close, volume

старующий столбец begin → переименовываем в datetime при агрегации.

Везде для чтения данных:

сначала пробуем processed/{TICKER}_{interval}.csv;

если нет — fallback в data/{TICKER}.csv.

[P2] Логгирование

По каждому тикеру:

print(f"[process-data] {sym}: intervals={intervals}, rows_in={len(df_raw)}, rows_out={len(df_resampled)}")

9. Контрольные команды (регулярный smoke test)

[P1] После любых крупных изменений прогонять минимум:

Агрегация:

python -m app2.cli process-data \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX \
  --intervals 10min 30min 1h \
  --input-dir data \
  --output-dir processed \
  --out out/process_summary.json


Режимы:

python -m app2.cli detect-regime \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX \
  --interval 30min \
  --config app2/config.json \
  --out out/regime_distribution.json


Свип (с маленькой сеткой, для скорости):

python -m app2.cli param-sweep \
  --strategy meanrev \
  --config app2/config.json \
  --csv out/sweep_meanrev_smoke.csv \
  --symbols SBER GAZP \
  --equity0 1000000 \
  --n-jobs 1


Forward-test (также маленькое окно):

python -m app2.cli forward-test \
  --strategy trend \
  --symbols SBER \
  --interval 30min \
  --train-window 500 \
  --test-window 100 \
  --step 100 \
  --equity0 1000000 \
  --out out/forward_trend_smoke.json \
  --n-jobs 1
