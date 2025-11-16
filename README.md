# MOEX APP2+ — ядро rule-/AI-трейдинга по акциям MOEX

Этот репозиторий — рабочее ядро для исследования, тестирования и постепенного вывода в прод стратегий по акциям MOEX на базе:
- простых rule-стратегий (trend / mean-reversion / breakout),
- детектора рыночных режимов (trend / range / high_vol),
- механизма бэктестов, forward-testing и свипа параметров,
- в дальнейшем — AI/ML-фильтров и мета-лейблинга.

---

## 1. Структура проекта

```text
moex-ai-bot/
  app2/
    __init__.py
    cli.py                 # единая точка входа для CLI-команд
    data.py                # загрузка сырых котировок с MOEX
    rule_core.py           # общая логика бэктеста (RuleBtParams, run_rule_symbol)
    rule_strategies.py     # rule-стратегии: trend, meanrev, breakout
    rule_backtest.py       # простые бэктесты одной стратегии
    regime_detector.py     # определение рыночных режимов
    regime_rule_backtest.py# бэктест с учётом режима
    param_sweep.py         # свип параметров (trend/meanrev/etc.)
    data_pipeline.py       # агрегация сырых данных в 10m/30m/1h (processed/)
    forward_test.py        # rolling forward-testing
    utils.py               # вспомогательные функции (load_symbols, save_json, и т.п.)
    config.py              # загрузка config.json
    config.json            # дефолтные параметры и сетки свипа
  data/                    # СЫРЫЕ котировки (например, SBER.csv, GAZP.csv, …)
  processed/               # АГРЕГИРОВАННЫЕ данные: SBER_30min.csv, …
  out/
    reports/               # JSON/CSV отчёты бэктестов/forward-test/свипов


Важное соглашение:
✅ Входные и выходные данные живут только в КОРНЕ проекта:

data/ — сырые данные (то, что скачали с MOEX),

processed/ — агрегированные данные (10m/30m/1h и т.д.),

out/ — любые результаты (equity, отчёты, свипы, forward-test, режимы).

Внутри app2/ не должно быть своих data/, out/ и т.п.

2. Основные CLI-команды

Все команды вызываются через:

python -m app2.cli <command> [аргументы]

2.1. rule-backtest

Бэктест одной rule-стратегии на наборе тикеров.

python -m app2.cli rule-backtest \
  --strategy trend|meanrev|breakout \
  --symbols SBER GAZP LKOH ... \
  --interval 30min \
  --equity0 1000000 \
  --out out/rule_trend.json


Использует:

rule_strategies.generate_*_signals,

rule_core.run_rule_symbol для расчёта equity и метрик.

2.2. regime-rule-backtest

Комбинирование стратегий по рыночным режимам.

python -m app2.cli regime-rule-backtest \
  --symbols GAZP ROSN SBER ... \
  --interval 30min \
  --equity0 1000000 \
  --no-breakout \        # опционально: не использовать breakout в high_vol
  --out out/regime_bt.json


Использует:

regime_detector.detect_regime — режим на каждом баре,

trend в режиме trend,

meanrev в режиме range,

breakout в режиме high_vol (опционально, можно отключить и просто не торговать).

2.3. process-data

Агрегация сырых данных в несколько таймфреймов.

python -m app2.cli process-data \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX ... \
  --intervals 10min 30min 1h \
  --input-dir data \
  --output-dir processed \
  --out out/reports/process_summary.json


Результат:

processed/TICKER_10min.csv

processed/TICKER_30min.csv

processed/TICKER_1h.csv

2.4. forward-test

Rolling forward-testing на отрезках «train_window → test_window».

python -m app2.cli forward-test \
  --strategy trend|meanrev|breakout|regime \
  --symbols GAZP ROSN SBER ... \
  --interval 30min \
  --train-window 1000 \
  --test-window 200 \
  --step 200 \
  --equity0 1000000 \
  --use-breakout-in-high-vol \
  --out out/forward_test_trend.json


Используется для более честной оценки стратегий во времени.

2.5. param-sweep

Свип параметров стратегии (сейчас — в первую очередь meanrev).

python -m app2.cli param-sweep \
  --strategy meanrev \
  --config app2/config.json \
  --csv out/sweep_meanrev.csv \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX \
  --equity0 1000000


Сетки параметров лежат в config.json → sweep.MeanRevParams.

Результат: строка на комбинацию параметров и тикер.

2.6. detect-regime

Оценка долей времени в режимах для каждого тикера.

python -m app2.cli detect-regime \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX ... \
  --interval 30min \
  --config app2/config.json \
  --out out/regime_distribution.json


Результат: JSON по тикерам с процентом времени в trend / range / high_vol.

3. Стандарты написания кода (обязательные требования)

Эти требования фиксируем как правило для всех новых модулей и правок.

3.1. Логгирование (никаких «тихих» скриптов)

Для всех тяжёлых операций (и вообще для CLI-команд):

Обязательно логируем:

Старт команды и ключевые аргументы:

print(f"[sweep] strategy={strategy}, symbols={symbols}, equity0={equity0}")
print(f"[sweep] config={config_path}, out={csv_path}")


Крупные этапы:

загрузка конфига,

размер сетки свипа,

начало/окончание обработки каждого тикера,

прогресс по комбинациям (idx / total),

резюме: сколько строк записано, куда.

Пропуски и ошибки данных:

if df is None:
    print(f"[sweep-meanrev] no data for {sym}, skip")


Финиш:

print(f"[forward-test] done, windows={n_windows}, out={out_path}")


Запрещено:

«Молча» писать или не писать файлы;

проглатывать исключения без хотя бы print с текстом ошибки.

3.2. Мультипроцессинг — по умолчанию ВКЛЮЧЕН

Для любых задач, где есть:

перебор по тикерам,

большой свип параметров,

rolling forward-windows,

мы используем мультипроцессинг / многопоточность по умолчанию:

Добавляем параметр n_jobs (CLI: --n-jobs, дефолт -1 = все ядра).

Если n_jobs != 1, используем:

concurrent.futures.ProcessPoolExecutor

или multiprocessing.Pool.

Для лёгких задач n_jobs=1 явно отключает мультипроцессинг.

Правило:
Новые модули для свипа / forward-test / массовых бэктестов должны сразу поддерживать n_jobs, и CLI должен передавать этот параметр.

3.3. Каталоги и пути

Только корневые data/, processed/, out/.

Внутри app2/ — никаких локальных data/, out/, reports/.

Все пути строим через os.path.join / Path с учётом этих корневых каталогов.

3.4. Конфигурация и «магические числа»

Все параметры стратегий, риск-менеджмента, режимов и сеток свипа — только через config.json.

config.py предоставляет load_config(path); путь по умолчанию — app2/config.json.

Никаких жёстких чисел (0.0015, 2.5, 0.98, 14 и т.п.) внутри кода — только чтение из config.

3.5. Таймзона и таймфреймы

Базовый ориентир — московское время (Europe/Moscow), даже если фактическая торговля пойдёт из других таймзон.

Мульти-таймфрейм:

raw → 1m/5m/10m в data/,

агрегированные 10m/30m/1h/… → в processed/.

При расчётах (особенно forward-test) важно аккуратно работать с датой/временем и не пересекать сессии.

3.6. Обработка ошибок

Любая неожиданная ситуация (нет данных, пустой DataFrame, нет секции в конфиге) — даёт понятное сообщение в логе:

if "MeanRevParams" not in config.get("sweep", {}):
    raise ValueError("В config.json нет секции 'sweep.MeanRevParams'")


Лучше упасть с понятной ошибкой, чем «тихо» вернуть пустой результат.

4. Текущий прогресс
4.1. Данные и инфраструктура

Реализован data.fetch_range + save_csv для выгрузки котировок MOEX.

Собран расширенный набор тикеров (до 18 штук: GAZP, ROSN, SBER, LKOH, GMKN, YNDX, NVTK, NLMK, MTSS, TATN, CHMF, SNGS, PIKK, PLZL, MGNT, VKCO, OZON, …).

Введено разделение:

data/ — сырые данные,

processed/ — агрегированные таймфреймы.

Добавлен process-data CLI для мульти-таймфреймной агрегации (10m, 30m, 1h).

4.2. Rule-стратегии и риск-менеджмент

Реализованы:

generate_trend_signals,

generate_meanrev_signals,

generate_breakout_signals.

Параметры ужесточены (увеличены пороги, min_gap_bars и т.п.), чтобы снизить шум и частоту сделок.

Вынесен общий бэктест в rule_core.py:

RuleBtParams (комиссия, slippage, risk_per_trade, ATR-стоп, TP),

run_rule_symbol(df, params, equity0) — универсальная функция для всех стратегий.

risk_per_trade снижён до 0.15% и ниже; реализован ATR-стоп.

4.3. Режимы рынка

Реализован regime_detector.detect_regime:

режущие пороги по ATR/цене (high_vol_quantile),

нормированная разница EMA_fast/EMA_slow (trend_threshold),

режимы: high_vol, trend, range.

Добавлена CLI-команда detect-regime, которая считает долю времени в каждом режиме для тикеров.

По результатам:

~95% времени рынок находится в range,

~5% — high_vol,

trend — крайне редкий режим.

Это подтвердила агрегация для базового набора тикеров (GAZP, ROSN, SBER, LKOH, GMKN, YNDX).

4.4. Regime-aware бэктест и forward-testing

regime_rule_backtest комбинирует:

trend-стратегию в режиме trend,

mean-reversion в режиме range,

breakout в режиме high_vol (опционально, можно отключить).

Реализован forward_test.py:

rolling forward-test по окнам (train/test/step),

запуск через CLI forward-test.

Выполнены тесты трендовой стратегии на 30m-интервале на расширенном наборе тикеров; результаты далеки от целевых → решено сосредоточиться на улучшении стратегий, особенно meanrev.

4.5. Свип параметров

Реализован базовый свип для trend (раньше).

Настроен и запущен свип для meanrev:

сетка параметров в config.json > sweep.MeanRevParams:

rsi_low, rsi_high,

rsi_len,

boll_window (bb_len),

boll_mult (bb_k),

min_gap_bars,

свип идёт по 6 базовым тикерам (GAZP, ROSN, SBER, LKOH, GMKN, YNDX),

используется run_rule_symbol из rule_core.

5. План по итерациям (обновлённый)
Итерация 1 — Данные и базовая архитектура ✅ (сделано)

Структурировать проект (app2/, data/, processed/, out/).

Реализовать rule-стратегии (trend/meanrev/breakout).

Ввести общий бэктест (rule_core.run_rule_symbol).

Добавить базовый risk-менеджмент (ATR-стоп, риск на сделку).

Реализовать rule-backtest и regime-rule-backtest в CLI.

Настроить агрегацию данных (process-data).

Итерация 2 — Улучшение стратегий с учётом режима (ТЕКУЩАЯ)

Фокус: mean-reversion, т.к. рынок ~95% времени в режиме range.

Задачи:

Свип mean-reversion (in progress):

завершить свип по MeanRevParams,

добавить multiprocessing (--n-jobs, дефолт = все ядра),

анализ результатов:

подобрать лучшие параметры по тикеру,

оценить стабильность (по периодам).

Усиление логики входа/выхода:

фильтр входов:

обязательная комбинация «RSI + Bollinger»,

дополнительный фильтр по SMA (например, meanrev торгует только против локального тренда),

TP/макс. время удержания:

добавить max_hold_bars,

добавить take-profit в RuleBtParams (от ATR или фиксированный RR),

фильтрация по волатильности:

отключать mean-reversion в экстремально низкой/высокой волатильности.

Регимная адаптация:

возможно чуть ослабить пороги trend_threshold и high_vol_quantile, чтобы режим trend появлялся чаще, но без шума,

внедрить жёсткие правила для high_vol:

либо «не торговать»,

либо торговать breakout с очень консервативным risk-менеджментом.

Логгирование и мультипроцессинг:

довести до стандарта все тяжёлые модули:

param_sweep (свип),

forward_test,

process-data,

добавить --n-jobs и прогресс-логи во все соответствующие команды.

Итерация 3 — High_vol, black swan и усиленный risk-layer

Проработать сценарии «black swan»:

hard-стопы по дневной просадке,

пауза в торговле при выходе за пределы (ATR/дневной %).

Довести high_vol режим:

по умолчанию — no trade,

опционально — breakout c малым плечом и жёсткими лимитами.

Forward-test для regime-стратегии:

сравнить performance с обычной trend / meanrev.

Итерация 4 — AI/ML уровень

Мета-лейблинг:

обучить модель (CatBoost/LightGBM и т.п.) на уже отобранных rule-сигналах,

цель — фильтровать худшие входы и поднять winrate/PF.

Event-based модели:

обучение моделей на «больших движениях» (>= K·ATR за N баров).

Интеграция с forward-test и строгой валидацией (walk-forward).

Итерация 5 — Инфраструктура и масштабирование

PostgreSQL как хранилище исторических данных и результатов.

Вынос тяжёлого обучения (AI, большие свипы) в облако.

Единый слой для:

логов,

конфигов,

учёта версий моделей/стратегий.