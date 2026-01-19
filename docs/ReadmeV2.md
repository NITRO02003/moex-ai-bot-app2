# MOEX AI Bot – APP2

Алгоритмический бот для торговли акциями Московской биржи (TQBR) с упором на:
- rule-based стратегии (trend / mean-reversion / breakout),
- детекцию рыночных режимов (trend / range / high_vol),
- строгий риск-менеджмент (ATR-стоп + риск на сделку),
- параметрические свипы и forward-testing,
- подготовку диагностик и датасета для будущей AI-модели.

Проект занимает промежуточное место между первой версией (`app`) и целевой архитектурой (`app3`). В `APP2` мы доводим до ума **ядро**, стандартизируем пайплайн данных и метрики, а затем переносим концепции в `APP3`.

---

## Структура проекта

```text
moex-ai-bot/
  ├── app2/
  │   ├── __init__.py
  │   ├── cli.py                 # единый вход: backtest, regime, sweep, forward-test, process-data, analyze-trades
  │   ├── config.py              # загрузка config.json
  │   ├── config.json            # дефолтные параметры и сетки для свипов
  │   ├── utils.py               # вспомогалки: load_symbols и пр.
  │   ├── data.py                # загрузка котировок с MOEX, save_csv
  │   ├── data_pipeline.py       # агрегация сырых данных -> processed/
  │   ├── rule_strategies.py     # генераторы сигналов: trend / meanrev / breakout
  │   ├── rule_core.py           # ядро бэктеста + риск-менеджмент + bar/trade-логи
  │   ├── rule_backtest.py       # простой backtest по одной стратегии
  │   ├── regime_detector.py     # детектор режимов: trend/range/high_vol
  │   ├── regime_rule_backtest.py# backtest с переключением стратегий по режимам
  │   ├── param_sweep.py         # параметрический свип (с multiprocessing)
  │   ├── forward_test.py        # rolling forward-test (будет «прокачан»)
  │   ├── analysis.py            # диагностический модуль (bar/trade-логи)
  │   └── ...
  ├── data/                      # сырые котировки (1m/10m/15m и т.п.), по тикеру: SBER.csv, GAZP.csv, ...
  ├── processed/                 # агрегированные таймфреймы: TICKER_10min.csv, TICKER_30min.csv, TICKER_1h.csv
  └── out/
      ├── reports/               # отчёты backtest/forward-test/сводки
      ├── sweep_*.csv            # результаты свипов
      ├── forward_*.json         # результаты forward-test
      └── diag_*_*.csv           # diagnostics: bar- и trade-логи для анализа

Важно: **все входные и выходные файлы** живут в корне проекта:  
`data/`, `processed/`, `out/`. Внутри `app2/` **нет своих** `data/` и `out/`.

---

## Основные компоненты

### 1. Пайплайн данных (`data_pipeline.py`)

Назначение: привести сырые CSV к стандартизированным таймфреймам.

- Вход: `data/TICKER.csv`
- Выход: `processed/TICKER_10min.csv`, `processed/TICKER_30min.csv`, `processed/TICKER_1h.csv` и т.п.
- Колонки на выходе:
  - `begin` (время бара),
  - `open`, `high`, `low`, `close`, `volume`.

Агрегация:

- `resample(freq)` с частотами `'10min'`, `'30min'`, `'1h'` (без устаревших `'T'`/`'H'`).
- OHLC:
  - `open` – первый,
  - `high` – максимум,
  - `low` – минимум,
  - `close` – последний.
- `volume` – сумма по окну.

Встроены проверки качества:

- сортировка по времени и отсутствие дубликатов временных отметок,
- базовая консистентность OHLC (low ≤ open/close ≤ high),
- отсутствие отрицательных объёмов,
- грубый контроль `rows_out / rows_in` (слишком большие значения помечаются как подозрительные).

CLI:

```bash
python -m app2.cli process-data \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX NVTK NLMK MTSS TATN CHMF SNGS PIKK PLZL MGNT VKCO OZON \
  --intervals 10min 30min 1h \
  --input-dir data \
  --output-dir processed \
  --out out/process_summary.json
2. Rule-стратегии (rule_strategies.py)
Три базовые стратегии, работающие поверх OHLC-данных:

Трендовая (generate_trend_signals)
Сигналы:

long, если нормированная разница EMA_fast - EMA_slow > trend_thr,

short — зеркально,

между переключениями позиции — минимум min_gap_bars баров.

Параметры (dataclass TrendParams):

ema_fast, ema_slow,

atr_len (для нормировки),

trend_thr,

min_gap_bars.

Mean Reversion (generate_meanrev_signals)
Сигналы:

long, если RSI < rsi_low и цена ниже нижней Bollinger band,

short зеркально (RSI > rsi_high и цена выше верхней Bollinger).

Параметры (MeanRevParams):

rsi_len, rsi_low, rsi_high,

bb_len (в config — boll_window),

bb_k (в config — boll_mult),

min_gap_bars.

Breakout (generate_breakout_signals)
Сигналы:

long при пробое верхней границы канала за channel_len баров;

short при пробое нижней.

Параметры (BreakoutParams):

channel_len,

confirm_bars,

min_gap_bars.

Параметры постепенно ужесточались (больше ATR, экстремальные RSI, длиннее каналы, больше min_gap_bars) для уменьшения числа сделок и шума.

3. Ядро бэктеста и риск-менеджмент (rule_core.py)
Ядро, которое:

рассчитывает ATR,

управляет размером позиции по risk_per_trade и ATR-стопу,

учитывает комиссию и проскальзывание,

ведёт bar-level и trade-level статистику,

считает метрики.

Основные параметры (RuleBtParams):

commission — комиссия, доля от объёма сделки (на круг),

slippage_bps — проскальзывание в б.п. от объёма,

risk_per_trade — доля капитала под риск на сделку (например, 0.0015 = 0.15%),

atr_len — длина окна ATR,

sl_atr_mult — стоп-лосс в ATR,

tp_mult — тейк-профит в ATR,

max_leverage — ограничение плеча по позиции.

Функция:

python
Копировать код
run_rule_symbol(
    df,                 # DataFrame с колонками: datetime/begin, open, high, low, close, signal
    params: RuleBtParams,
    equity0: float = 1_000_000.0,
    collect_bar_stats: bool = False,
    collect_trades: bool = False,
) -> Dict[str, Any]
Возвращает:

equity_curve: pd.Series по времени,

metrics: словарь с ключевыми метриками:

total_return

max_drawdown

calmar

volatility_ann

sharpe_ann

trade_count

win_rate

avg_trade

pnl_sum, pnl_mean, pnl_std

и др. при необходимости,

при collect_bar_stats=True:
bar_stats: pd.DataFrame (одна строка на бар):

datetime

close

signal

position (в штуках или знаке)

equity (M2M-эквити на бар)

drawdown (от пика эквити)

trade_id (ID активной сделки либо NaN)

при collect_trades=True:
trades: pd.DataFrame (одна строка на сделку):

trade_id

entry_dt, exit_dt

entry_price, exit_price

direction (1 = long, -1 = short)

qty

pnl_abs (рубли)

pnl_rel (к equity на входе)

bars_in_trade

max_favorable_excursion

max_adverse_excursion

Особенности:

ATR считается по классическому TR (max(high-low, |high-prev_close|, |low-prev_close|)).

Внутри цикла по барам фиксируются:

стопы/тейки по ATR,

развороты/выход по смене сигнала,

комиссия+slippage по объёму сделки.

Параметр sl_atr_mult задаёт размер стопа от ATR; размер позиции — через risk_per_trade / (ATR * sl_atr_mult) (с ограничением по max_leverage).

Важный фикc:

python
Копировать код
df["returns"] = df["close"].pct_change(fill_method=None).shift(-1)
→ убирает FutureWarning от pandas.

4. Режимы рынка (regime_detector.py, regime_rule_backtest.py)
Определение режима:

считаются:

ATR и относительный ATR к цене;

rolling-квантили ATR (например, 98-й) → порог high_vol;

разница EMA_fast - EMA_slow, нормированная на ATR → трендовость.

Логика:

если ATR > порога → high_vol,

иначе, если нормированная разница EMA > trend_threshold → trend,

иначе → range.

Параметры (RegimeParams) берутся из config.json:

json
Копировать код
"RegimeParams": {
  "high_vol_quantile": 0.98,
  "trend_threshold": 2.5,
  "atr_len": 14,
  "ema_fast": 12,
  "ema_slow": 48
}
Режимный backtest:

заранее считаются сигналы всех трёх стратегий: trend / meanrev / breakout;

режим trend → используется трендовая стратегия;

режим range → mean-reversion;

режим high_vol:

по умолчанию — либо breakout, либо «нет торговли» (в будущих версиях — работа с экстремумами/black swan).

Отдельная CLI-команда для оценки распределения режимов:

bash
Копировать код
python -m app2.cli detect-regime \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX \
  --interval 30min \
  --config app2/config.json \
  --out out/regime_distribution.json
5. Параметрические свипы (param_sweep.py)
Назначение: перебрать сетку параметров для заданной стратегии и тикеров и собрать метрики.

Сейчас реализован:

Mean-reversion свип (strategy="meanrev").

Читает config.json:

json
Копировать код
"sweep": {
  "MeanRevParams": {
    "rsi_low":      [20, 25, 30, 35],
    "rsi_high":     [65, 70, 75, 80],
    "rsi_len":      [10, 14, 20],
    "boll_window":  [14, 20, 30],
    "boll_mult":    [1.5, 2.0, 2.5],
    "min_gap_bars": [10, 20, 30]
  }
}
И дефолты для risk-менеджмента:

json
Копировать код
"defaults": {
  "RuleBtParams": {
    "commission": 0.0005,
    "slippage_bps": 1.0,
    "risk_per_trade": 0.0015,
    "atr_len": 14,
    "sl_atr_mult": 1.5,
    "tp_mult": 3.0
  }
}
Логика:

генерирует все комбинации сетки MeanRevParams;

для каждого тикера один раз загружает processed/TICKER_30min.csv;

для каждой комбинации:

собирает MeanRevParams,

генерирует сигналы,

запускает run_rule_symbol,

извлекает метрики и пишет строку в итоговый DataFrame.

Параллелизация:

параллелим по тикерам, а не по комбинациям,

используется ProcessPoolExecutor,

n_jobs задаётся в CLI.

CLI:

bash
Копировать код
# smoke-test на нескольких тикерах, без multiprocessing
python -m app2.cli param-sweep \
  --strategy meanrev \
  --config app2/config.json \
  --csv out/sweep_meanrev_smoke.csv \
  --symbols GAZP ROSN SBER \
  --equity0 1000000 \
  --n-jobs 1

# полный свип на всех тикерах, с multiprocessing
python -m app2.cli param-sweep \
  --strategy meanrev \
  --config app2/config.json \
  --csv out/sweep_meanrev.csv \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX NVTK NLMK MTSS TATN CHMF SNGS PIKK PLZL MGNT VKCO OZON \
  --equity0 1000000 \
  --n-jobs -1
Результат:

CSV с колонками:

strategy, symbol,

параметры (rsi_low, rsi_high, rsi_len, boll_window, boll_mult, min_gap_bars),

total_return, max_drawdown.

Используется для:

оценки «дружелюбности» тикера к mean-reversion (например, YNDX/LKOH/GMKN vs SBER/GAZP),

выбора «здравых» зон параметров для дальнейшего forward-test и AI.

6. Forward-testing (forward_test.py)
Назначение:

проверить устойчивость стратегии во времени по схеме walk-forward (train-window / test-window / step);

в будущем — ядро для обучения и валидации AI-моделей.

Планируемый интерфейс:

bash
Копировать код
python -m app2.cli forward-test \
  --strategy meanrev \
  --symbols YNDX LKOH GMKN \
  --interval 30min \
  --train-window 1000 \
  --test-window 200 \
  --step 200 \
  --equity0 1000000 \
  --out out/forward_meanrev.json \
  --n-jobs -1
Внутри:

для каждого тикера:

двигаем окно [train, test] по данным,

для каждого окна:

берём фиксированные rule-параметры (из config.json),

генерируем сигналы,

гоняем run_rule_symbol на тестовом куске,

сохраняем метрики окна;

результаты:

список окон по тикеру,

средние метрики по окнам,

в перспективе — вывод по портфелю.

Статус: есть базовая реализация; требуется «прокачка» под единые стандарты:

n_jobs,

стабильный формат метрик,

сохранение per-window результатов для последующего анализа/AI.

7. Диагностика и анализ сделок (analysis.py, команда analyze-trades)
Это ключевая новая часть, основанная на обсуждении в этом чате.

Цель:

наложить график прибыльности и эквити на график цены,

увидеть, где именно возникают:

самые большие просадки (DD),

концентрация убыточных сделок,

понять паттерны (режимы, фичи, время), в которых стратегия умирает,

подготовить датасет для AI-модели (meta-labeling).

Мы теперь умеем из run_rule_symbol получать:

bar-level статистику;

trade-level статистику.

Модуль analysis.py:

функция run_analyze_trades(...):

загружает данные по тикеру и интервалу,

генерирует сигналы нужной стратегии (trend, meanrev, breakout),

вызывает run_rule_symbol(..., collect_bar_stats=True, collect_trades=True),

сохраняет:

out/diag_<prefix>_<SYMBOL>_<strategy>_<interval>_bars.csv,

out/diag_<prefix>_<SYMBOL>_<strategy>_<interval>_trades.csv.

CLI-команда:

bash
Копировать код
python -m app2.cli analyze-trades \
  --strategy meanrev \
  --symbols YNDX LKOH GMKN \
  --interval 30min \
  --equity0 1000000 \
  --config app2/config.json \
  --out-prefix out/diag_meanrev
Результат:

out/diag_meanrev_YNDX_meanrev_30min_bars.csv

out/diag_meanrev_YNDX_meanrev_30min_trades.csv

и аналогично по остальным тикерам.

Дальше эти файлы можно:

визуализировать в Jupyter / BI:

график цены + точки входов/выходов,

подсвеченные зоны большого DD,

scatter-плоты фич vs PnL;

использовать как тренировочный датасет:

фичи: ATR, EMA-спред, RSI, Bollinger, режим, время дня, bars_since_entry и др.;

таргет: бинарный good/bad-trade, будущий PnL, MAE/МFE и т.д.

Конвенции по коду и стандартам
Структура директорий:

только один набор: data/, processed/, out/ в корне;

никаких app2/data и app2/out.

Логгирование:

все CLI-команды логируют ключевые этапы:

старт (параметры),

прогресс (например, каждые N комбинаций/окон),

завершение (кол-во строк/окон, путь к файлу).

Параллелизм:

все тяжёлые команды получают опцию --n-jobs:

-1 — все ядра,

1 — без multiprocessing,

N — конкретное число процессов.

Сейчас это реализовано в param-sweep, будет добавлено в forward-test и при необходимости в другие.

Конфиг и «магические числа»:

все ключевые параметры хранятся в config.json:

defaults.* — базовые значения,

sweep.* — сетки для перебора.

Код не должен «зашивать» числа типа 0.0015, 2.5, 0.98 прямо в функции.

План развития (итерации)
Итерация 1 — Стабилизация ядра (выполнено / в процессе)
 Привести структуру директорий к стандарту (data/, processed/, out/).

 Реализовать и стабилизировать data_pipeline с проверками качества и без устаревших частот ('T', 'H').

 Унифицировать risk-core в rule_core.py:

 RuleBtParams + run_rule_symbol,

 метрики (total_return, max_drawdown, win_rate, profit_factor, ...),

 убрать FutureWarning по pct_change.

 Обновить CLI (cli.py) и привести команды к единому стилю.

 Реализовать параметрический свип mean-reversion:

 чтение сеток из config.json,

 multiprocessing по тикерам,

 выдача CSV с метриками.

Итерация 2 — Расширенный анализ свипов и режимов (частично выполнено)
 Запустить meanrev свип на базовом наборе тикеров.

 Оценить «дружелюбность» тикеров к mean-reversion:

YNDX/LKOH/GMKN — перспективные,

SBER/GAZP — скорее трендовые/сложные для mean-rev.

 Оценить распределение режимов по тикерам (detect-regime).

 Сформировать рекомендуемый набор тикеров:

 mean-rev core: YNDX, LKOH, GMKN (+ ROSN),

 trend core: SBER, GAZP (+ ещё несколько).

 Обновить config.json с пер-тикерными дефолтами параметров (или хотя бы per-strategy baseline).

Итерация 3 — Диагностика сделок и DD (текущая)
 Расширить run_rule_symbol bar/trade-логами.

 Добавить модуль analysis.py и команду analyze-trades в CLI.

 Снять bar/trade-логи для:

 mean-rev на YNDX/LKOH/GMKN,

 trend на SBER/GAZP.

 Провизуализировать:

 цена vs входы/выходы,

 зоны max DD,

 концентрация убыточных сделок по режимам и фичам.

 На основе диагностики донастроить фильтры:

 не торговать mean-rev в high_vol,

 фильтры по ATR (low_vol),

 фильтры по тренду (запрет на вход против «сильного» тренда),

 фильтры по времени (конец сессии, неликвидные периоды).

Итерация 4 — Прокачка forward-testing
 Привести forward_test к стандарту:

 единый интерфейс run_forward_test(...),

 поддержка --n-jobs,

 формат результатов: по окнам + агрегированные метрики.

 Прогнать forward-test для:

 mean-rev в core-тикерах,

 trend в core-тикерах.

 Использовать forward-результаты для:

 фильтрации переобученных параметров из свипа,

 выбора стабильных baseline-настроек.

Итерация 5 — Подготовка AI-датасета и baseline-модель
 На базе bar/trade-логов из analysis.py собрать датасет:

 признаки: ATR, EMA-спрэды, RSI, Bollinger, режим, bars_since_entry, time_of_day и т.д.,

 цели:

 бинарные (good/bad entry),

 регрессионные (future PnL, MAE/МFE),

 multi-class (big_win / small_win / small_loss / big_loss).

 Выбрать baseline-модель:

 CatBoost / LightGBM,

 либо простая нейросетка (PyTorch) — особенно актуально на GTX1080.

 Встроить meta-labeling:

 для каждого rule-сигнала прогнозировать качество входа,

 фильтровать сделки по вероятности плохого исхода.

Итерация 6 — Портфель и инфраструктура
 Портфельное управление:

 аллокация по тикерам/стратегиям (rule-based),

 в перспективе — RL-подход.

 Инфраструктура:

 PostgreSQL для хранения исторических данных и результатов,

 GPU-ускорение для ML-части (GTX1080),

 развёртывание прод-пайплайна (signal → execution).

Перенос контекста в новый чат
Этот README аккумулирует:

текущую структуру проекта,

договорённости по директориям, логгированию и multiprocessing,

результаты обсуждений по свипам и режимам,

дизайн bar/trade-логов и analyze-trades,

план развития по итерациям.

В новом чате достаточно будет:

дать ссылку на репозиторий,

сказать, что «актуальный контекст и план — в app2/README.md»,

и мы сразу продолжим с нужного места (например, с прокачки forward_test или с анализа diagnostics CSV).