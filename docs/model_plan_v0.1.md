# Model Plan v0.1 — подробный план развития AI-ядра (APP2+ / APP3 / MTL)

Файл: `docs/model_plan_v0.1.md`  
Связан с:
- `docs/READMEv4.7.md` (текущий миграционный чекпоинт),
- будущим `docs/READMEv4.8.md`,
- будущим `docs/CONTRACTv4.8.md` (архитектурный и data-контракт).

Документ собирает **максимально детализированный план** эволюции проекта:
от текущего rule-based ядра `app2` → к многоцелевой AI-модели (MTL) в `app3`,
с учётом обсуждений и набросков дифф-ТЗ с соавтором.

---

## 0. Организация документации и версионирование

### 0.1. Структура документации

Вся документация живёт в каталоге `docs/`:

- `docs/READMEv4.7.md` — текущий миграционный чекпоинт.
- `docs/READMEv4.8.md` — следующий чекпоинт (после завершения Фазы 0).
- `docs/CONTRACTv4.8.md` — архитектурный **и data-контракт** (Regime → Strategy → AI → Risk).
- `docs/model_plan_v0.1.md` — этот план (будет эволюционировать дальше).

Дополнительные документы по мере развития:

- `docs/dataset_spec_v*.md` — форматы датасетов (Entry / In-Trade).
- `docs/features_spec_v*.md` — спецификации фич (в т.ч. intermarket/sentiment).
- `docs/mtl_arch_v*.md` — архитектура MTL-модели.

Версии связаны между собой:

- `READMEv4.8` ↔ `CONTRACTv4.8` ↔ `model_plan_v0.1`  
  (при изменении контракта или плана — новые версии `v4.9`, `v0.2` и т.д.).

### 0.2. Связь план → код → артефакты

Для каждой фазы план фиксирует:

- **Файлы кода** (`app2/`, `app3/`), которых касается фаза.
- **Артефакты в `out/`**, которые должны появиться (отчёты, датасеты, debug).
- **CLI-команды**, обязательные для sanity-check.
- **Критерии успеха** (минимальные метрики / инварианты).

---

## 1. Большая картинка: фазы

Крупные фазы:

1. **Фаза 0 — Range V3 Baseline**  
   Zero Trades → живой baseline Range V3 на SBER 30m и других тикерах.

2. **Фаза 1 — Архитектура датасетов и таргетов (Regime / Entry / Exit, dynamic Y_exit)**  
   Чёткая структура Entry-/In-Trade-датасетов для MTL-модели.

3. **Фаза 2 — Первый AI-фильтр входа + базовый Regime Detector**  
   XGBoost/CatBoost/MLP как AI-надстройка над Range V3.

4. **Фаза 3 — Adaptive Exit (Head 3, динамический выход)**  
   Модель, принимающая решение «держать/закрывать» на каждом баре.

5. **Фаза 4 — Sentiment и Intermarket Features**  
   Локальный sentiment (Transformer) + intermarket-фичи в пайплайне.

6. **Фаза 5 — Unified MTL Core + Online Learning / Transfer Learning**  
   Объединение Regime/Entry/Exit в одно MTL-ядро, опция дообучения.

---

## 2. Фаза 0 — Range V3 Baseline (Zero Trades → Trades)

**Цель:**  
Сделать из «0 сделок» **стабильную rule-based Range V3**, которая:

- строит сегменты,
- генерирует осмысленные сделки на SBER 30m (и далее на других тикерах),
- не обязательно прибыльна (PF может быть ≤ 1), но:
  - сделки не случайны,
  - поведение соответствует идее торговли диапазона.

### 2.0. Файлы и артефакты

Код:

- `app2/range/range_v3.py`
- `app2/range/features.py` (если понадобится доработка фич для сегментации)
- `app2/range/v3_backtest.py`
- `app2/range/dataset.py` (для проверки снапшотов)
- `app2/range/cli.py` или `app2/cli.py` (для новой отладочной команды)
- `app2/range/config.json` (секция `RangeV3`)

Артефакты:

- `out/range_v3/*_v3seg_debug*.json` — расширенный debug.
- `out/range_v3/*_v3seg_trades*.csv` — сделки Range V3 baseline.
- `out/range_v3/*_v3seg_segments*.csv` — (опционально) лог сегментов.
- `out/range_v3/SEGMENTS_DEBUG_*.csv` — выгрузки для визуализации (см. шаг 0.F).

Документация:

- Обновление статуса в `docs/READMEv4.8.md` после завершения Фазы 0.

---

### 2.1. Шаг 0.A — Починить фундамент индикаторов (ATR/MA/NaN)

**Проблема:**  

- `ATR` и `MA` в текущей реализации на реальных данных зачастую полностью `NaN` из-за:
  - дыр в данных (ночная сессия),
  - `rolling(..., min_periods=window)`.

**Действия:**

1. В `app2/range/range_v3.py`:

   - Функция `_calc_atr(high, low, close, window: int)`:
     - заменить `rolling(window=window, min_periods=window)` → более мягкое:
       - первый диагностический шаг — `min_periods=1`,
       - добавить комментарий, что это сделано из-за дыр в данных, т.к. pandas игнорирует NaN.

   - Функция `_calc_ma(close, window: int)`:
     - аналогично — `min_periods=1` с комментарием.

2. Аналогично смягчить `min_periods` в других вспомогательных MA/ATR, используемых **только в Range V3**, если они приводят к тотальным NaN.

**Sanity-check CLI:**

```bash
python -c "import app2.range.range_v3 as r3; print('range_v3 import OK')"
Критерий успеха:

При диагностике на SBER 30m видно, что atr, ma, slope получают ненулевые значения (не тотально NaN).

2.2. Шаг 0.B — Смягчение config-фильтров (AAA/AA, ADX и т.п.)
После того как фундамент индикаторов заработал:

В app2/range/config.json (секция RangeV3):

Временное смягчение порогов:

min_tests_AAA: уменьшить (например, с 3 до 2).

min_tests_AA: уменьшить (например, с 2 до 1).

adx_low_max: увеличить (например, с 20 до 30).

Цель — получить первые сегменты и сделки, не финальную «боевую» конфигурацию.

CLI (диагностика на одном тикере):

bash
Копировать код
python -m app2.cli range-v3-backtest \
  --symbols SBER \
  --interval 30min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/DEBUG_SBER_30m \
  --tag v3seg_cfgtest
Критерий успеха:

out/range_v3/DEBUG_SBER_30m_v3seg_cfgtest_debug.json:

segments_total > 0.

*_trades_v3seg_cfgtest.csv не пустой.

2.3. Шаг 0.C — Расширенный debug-лог Range V3 (причины отбраковки)
Чтобы понимать, почему сегменты/сигналы отбрасываются:

В range_v3.py:

При фильтрации сегментов логировать причины:

reason: too_small,

reason: too_wide,

reason: breakout_fail,

reason: trend_too_strong,

reason: bad_quality (не AAA/AA по порогам),

и т.п.

В *_v3seg_debug.json добавить:

segments_total

segments_tradable

segments_used

reasons_count: словарь reason -> count.

Это позволит по одному debug-файлу понять, на каком уровне «умирают» кандидаты в сделки.

2.4. Шаг 0.D — Инструмент визуализации сегментов (новая CLI-команда)
Для живой диагностики логики Range V3 одного JSON мало.

Новый инструмент:

Команда:

bash
Копировать код
python -m app2.cli range-debug-segments \
  --symbol SBER \
  --interval 30min \
  --date 2024-01-15 \
  --config-range app2/range/config.json \
  --out out/range_v3/SEGMENTS_DEBUG_SBER_30m_2024-01-15.csv
Функциональность:

Для заданного symbol, interval, date:

прогнать детектор диапазонов на выбранных барах;

сохранить CSV с пометками:

L, U, M уровни,

segment_quality (AAA/AA/A/…),

reason (если сегмент отбрасывается),

флаги потенциальных точек входа/выхода.

Цель:

Возможность быстро загрузить CSV в любую plotting-среду и визуально увидеть:

где строятся (или не строятся) диапазоны,

почему build_range_box_v3 иногда возвращает None.

2.5. Шаг 0.E — Базовый baseline по SBER 30m
Когда фундамент (0.A), смягчение фильтров (0.B) и debug/визуализация (0.C/0.D) в рабочем состоянии:

Снять эталонный baseline на SBER 30m.

CLI:

bash
Копировать код
python -m app2.cli range-v3-backtest \
  --symbols SBER \
  --interval 30min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/BASE_SBER_30m \
  --tag v3seg_base
Критерии успеха:

За выбранный период:

≥ 50 сделок.

В debug:

разумное количество сегментов разных качеств (AAA/AA/A).

Equity-поведение:

без очевидной случайной «пилы»,

сделки реально выглядят как торговля диапазонов (от границ к середине и т.п.).

2.6. Шаг 0.F — Расширение baseline на портфель тикеров
CLI:

bash
Копировать код
python -m app2.cli range-v3-backtest \
  --symbols SBER GAZP LKOH GMKN ROSN \
  --interval 30min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/ALL_30m_BASE \
  --tag v3seg_base
Критерии:

По всем тикерам суммарно:

≥ 50–100 сделок.

Нет тикеров с 0 сделок.

2.7. Результат Фазы 0
В docs/READMEv4.8.md фиксируется:

оживление Range V3 baseline,

статистика по SBER 30m и по портфелю,

текущие параметры RangeV3 в config.json.

Range V3 готов:

к сбору датасетов (Фаза 1),

к дальнейшей AI-надстройке.

3. Фаза 1 — Архитектура датасетов и таргетов (Regime / Entry / Exit)
Цель:
Сформировать чёткую структуру данных для MTL-ядра:

Dataset A: Entry Snapshots — для Heads 1/2 (Regime & Entry).

Dataset B: In-Trade Time-Series — для Head 3 (Exit, bar-level target).

Уже на этом этапе учитываем:

будущие intermarket-фичи,

будущий sentiment,

динамическую природу Y_exit.

3.0. Файлы и артефакты
Код:

app2/range/dataset.py или новый модуль app2/ai/dataset.py

app2/ai/labeling.py — разметка таргетов Y_regime, Y_entry, Y_exit

app2/features.py — источники фич

Артефакты:

out/datasets/range_v3_entry_snapshots_30m.csv

out/datasets/range_v3_intrade_timeseries_30m.csv

Документация:

docs/dataset_spec_v0.1.md — описание колонок Entry/In-Trade.

3.1. Dataset A — Entry Snapshots (для Head 1 и Head 2)
Entry Snapshot — 1 строка на 1 потенциальный вход.

Столбцы (примерный набор):

Идентификаторы:

symbol, dt, interval

Базовые OHLCV:

open, high, low, close, volume

Технические фичи:

ATR / волатильность (atr_*, vol_*),

позиция в диапазоне (band_pos, расстояние до границ),

фичи свечей (bar_body_pct, bar_range_pct, upper_shadow_pct…),

индикаторы тренда/флета (ADX, slope MA и др.).

Сегментные фичи Range V3:

segment_quality (AAA/AA/A),

высота диапазона,

ширина deadzone,

положение точки входа в рамках сегмента.

Режим:

rule-based метка режима (если есть) — как вспомогательная,

в перспективе — таргет Y_regime.

Результат сделки (для разметки Y_entry):

pnl_rel,

bars_in_trade,

max_adverse_excursion (MAE),

max_favorable_excursion (MFE).

Таргеты:

Y_regime — истинный режим (класс) на момент входа (для Head 1).

Y_entry — качество входа (для Head 2):

базовый вариант: бинарный:

1, если pnl_rel ≥ pnl_threshold (например 0.001),

0 иначе;

продвинутый: квантильный:

good / bad по квантилям, середину можно исключать из обучения.

Действие 1.1.A:

Реализовать функцию make_entry_snapshots(...) в dataset.py/ai/dataset.py:

вход: трейд-лог Range V3 baseline,

выход: range_v3_entry_snapshots_30m.csv.

Описать формат в docs/dataset_spec_v0.1.md.

3.2. Dataset B — In-Trade Time-Series (для Head 3, динамический Y_exit)
In-Trade Time-Series — N строк на 1 сделку, где N = количество баров в сделке.

Столбцы:

Всё, что доступно на момент данного бара:

symbol, dt, interval,

текущие OHLCV,

технические и сегментные фичи (как в Entry, но по текущему бару),

Позиционные фичи (по сделке):

bars_held — сколько баров сделка уже живёт,

текущий pnl_rel,

текущие MAE, MFE,

time_since_entry,

возможные derived-фичи (скорость изменения ATR, band_pos и т.п.).

Режим:

сигнал от Head 1 (режим на текущем баре),

или rule-based режим.

Таргет Y_exit:

Динамический, bar-level таргет:

кодирует вероятность рационального раннего выхода:

Y_exit_t = 1, если оптимальным (по выбранной метрике) был бы выход на следующем баре.

Y_exit_t = 0 иначе.

В перспективе для MTL:

Head 3 будет оценивать P_early_exit = вероятность того, что на следующем баре имеет смысл выйти,

это критично для Adaptive SL/TP.

Действие 1.2.B:

В app2/ai/labeling.py реализовать:

label_exit_dynamic(trade) → последовательность Y_exit_t по барам.

3.3. Шаг 1.3.C — Создание In-Trade Dataset (разворачивание trades → баровый ряд)
Это отдельная, явно прописанная итерация (по замечанию):

Функция:

make_intrade_timeseries(trades_log, bars_data, ...) в dataset.py:

вход:

trades.csv (Range V3 baseline),

соответствующий OHLCV-ряд (из processed/),

выход:

range_v3_intrade_timeseries_30m.csv:

по каждой сделке развёрнута бар-уровневая последовательность,

на каждую строку — фичи + Y_exit.

CLI:

bash
Копировать код
python -m app2.cli range-v3-make-datasets \
  --symbols SBER GAZP LKOH GMKN ROSN \
  --interval 30min \
  --out-prefix out/datasets/RANGE_V3_30m
Результаты:

out/datasets/RANGE_V3_30m_entry_snapshots.csv

out/datasets/RANGE_V3_30m_intrade_timeseries.csv

3.4. Шаг 1.D — Контроль качества датасетов
Проверить:

Нет ли утечки будущей информации:

все фичи посчитаны только по данным ≤ текущего бара.

Соответствие размеров:

число строк Entry ≈ число сделок,

число строк In-Trade равно сумме длительностей сделок.

Баланс классов:

распределение Y_entry,

распределение Y_exit_t (не всё 0 или всё 1).

3.5. Результат Фазы 1
Есть два формализованных датасета:

Entry Snapshots (для Heads 1/2),

In-Trade Time-Series (для Head 3).

В docs/dataset_spec_v0.1.md описаны все колонки.

Всё готово для обучения первых моделей (Фаза 2/3).

4. Фаза 2 — Первый AI-фильтр входа и базовый Regime Detector
Цель:
Построить первые рабочие AI-модели:

Regime Detector (Head 1) — классификация режима.

Signal Filter (Head 2) — фильтр входов Range V3.

На этом этапе — табличные модели (XGBoost/CatBoost/MLP).

4.0. Файлы и артефакты
Код:

app3/train_entry_models.py (или подкоманда в app3/cli.py)

app3/config_entry_models.toml или .json

изменения в app3/backtest_ai_entry.py (или app2/range/v3_backtest.py с AI-режимом)

Артефакты:

models/head1_regime_xgb.bin

models/head2_entry_filter_xgb.bin

out/reports/entry_models_metrics.json

out/reports/feature_importance_head2.csv

4.1. Head 1 — Regime Detector
Данные:

Dataset A (Entry Snapshots).

Таргет: Y_regime (классы: trend, range, high_vol, ...).

Модель:

XGBoost/CatBoost:

вход: подмножество feature_cols (см. CONTRACT и dataset_spec_v*.md),

выход: распределение вероятностей по режимам.

Метрики:

Accuracy, F1 по классам,

Confusion matrix,

бизнес-интерпретация ошибок (например, пропуск range vs лишний range в тренде).

4.2. Head 2 — Entry Filter
Данные:

Dataset A (Entry Snapshots).

Таргет: Y_entry.

Модель:

XGBoost/CatBoost/MLP.

Интерпретация:

Feature importance,

частичные зависимости для ключевых фич,

понимание, какие факторы отделяют «хорошие» входы от «плохих».

4.3. Интеграция Head 1/2 в backtest (Points of Interruption)
Контракт прерываний (важно зафиксировать в CONTRACTv4.8.md):

При обработке сигнала в backtest (например, в v3_backtest.py):

python
Копировать код
if AI_REGIME_BLOCK:
    # Head 1 считает, что текущий режим не подходит для Range V3
    # → сигнал игнорируется, возможно закрытие существующей Range-позиции.
elif AI_ENTRY_FILTER_BLOCK:
    # Head 2 считает, что данный вход плохой
    # → сигнал игнорируется.
else:
    # Сигнал проходит в risk-core, открытие сделки
Приоритет:

Сначала Regime Detector (Head 1): имеет право полностью заблокировать стратегию в данном режиме.

Затем Entry Filter (Head 2): фильтрует конкретные сигналы в разрешённом режиме.

CLI пример:

bash
Копировать код
python -m app3 backtest ai-range-v3 \
  --symbols SBER GAZP LKOH GMKN ROSN \
  --interval 30min \
  --config-range app2/range/config.json \
  --entry-model-config app3/config_entry_models.toml \
  --out-prefix out/ai_range_v3/ALL_30m_AI \
  --tag v3_ai_entry
4.4. Оценка эффекта AI-фильтра
Сравнить:

baseline Range V3 (Фаза 0),

AI-Filtered Range V3.

Метрики:

PF (Profit Factor),

WinRate,

Max Drawdown,

число сделок,

стабильность по под-периодам.

Ориентировочный критерий успеха:

на отложенных периодах:

PF ≥ 1.2 (как первый ориентир),

DD не хуже baseline,

WinRate не падает.

5. Фаза 3 — Adaptive Exit (Head 3, динамический выход)
Цель:
Модель Head 3, которая принимает bar-level решение:

держать позицию или закрыть,

тем самым реализуя Adaptive SL/TP.

5.0. Данные
Dataset B (In-Trade Time-Series) из Фазы 1.

Таргет Y_exit_t:

интерпретация: P_early_exit — вероятность того, что на следующем баре выгодно выйти (см. CONTRACT).

5.1. Базовая модель Head 3
Модель:

на первом шаге — Gradient Boosting или MLP,

вход:

позиционные фичи (PnL, MAE, MFE, bars_held, …),

текущие рыночные фичи,

режим (Head 1 / rule-based),

(по мере готовности) sentiment/intermarket.

Выход:

вероятность P_early_exit:

при превышении порога → сигнал на выход,

при низком значении → продолжаем держать.

5.2. Интеграция Head 3 в backtest
В backtest (в AI-режиме):

на каждом баре открытой позиции:

считаем P_early_exit,

если P_early_exit > threshold — закрываем сделку (полностью или частично).

Ограничение:

Head 3 не может нарушать глобальные risk-лимиты:

не увеличивает риск по позиции,

может только уменьшать продолжительность/размер просадки.

5.3. Оценка эффекта Adaptive Exit
Сравнить:

PF, WinRate, MaxDD,

среднюю длину сделок,

хвосты распределения убытков (удалось ли срезать «толстые хвосты»).

6. Фаза 4 — Sentiment и Intermarket Features
Цель:
Обогатить feature space:

Intermarket: индексы, FX, сырьё и т.п.

Sentiment: локальный Transformer (RuBERT/мультиязычный).

6.1. Intermarket Features
Действия:

Выбрать релевантные инструменты:

индекс рынка,

валютный курс,

возможно, фьючерсы.

Разработать app2/intermarket.py:

загрузка и синхронизация данных,

расчёт intermarket-фич (спрэды, относительная сила, корреляции).

Расширить app2/features.py:

добавить intermarket-фичи в общий набор.

6.2. Sentiment Features
Пайплайн:

Сбор текстов (новости, отчёты, события) по тикерам/рынку.

Локальная Transformer-модель:

оценка тональности,

классификация типа события.

Аггрегация во времени:

rolling-окна sentiment,

Δ sentiment,

флаги «важных» событий.

Интеграция:

отдельная таблица с dt + sentiment_*,

join с OHLCV по времени (без lookahead),

добавление в features.py.

6.3. Обновление датасетов
После добавления intermarket/sentiment:

перегенерировать Entry и In-Trade datasets,

обновить docs/features_spec_v*.md,

обновить список feature_cols для AI в dataset_spec_v*.md и CONTRACTv4.8.md (data-контракт).

7. Фаза 5 — Unified MTL Core + Online Learning
Цель:
Перейти от набора отдельных моделей к единому MTL-ядру:

общие слои (shared encoder),

отдельные головы:

Head 1 — Regime,

Head 2 — Entry,

Head 3 — Exit.

7.1. MTL-архитектура
Дизайн:

NN-ядеро (MLP + attention / Transformer),

учёт временного контекста (последовательности баров),

вход:

feature_cols (согласно CONTRACT),

positional / intermarket / sentiment.

Выход:

Head 1: распределение по режимам.

Head 2: вероятность «хорошего» входа.

Head 3: P_early_exit.

7.2. Обучение MTL-модели
Данные:

Entry Dataset A для Heads 1/2,

In-Trade Dataset B для Head 3.

Лосс:

суммарный (взвешенный):

кросс-энтропия по Y_regime,

кросс-энтропия/лог-лосс по Y_entry,

лог-лосс по Y_exit.

7.3. Online Learning и Transfer Learning
Периодическое дообучение на новых данных.

Fine-tuning по тикерам/режимам.

Мониторинг деградации:

если метрики ухудшаются → roll-back модели / ре-трейн.

8. Метрики и цели
Долгосрочные ориентиры:

WinRate ≥ 65%

PF ≥ 2.0

приемлемый MaxDD (оговаривается в CONTRACT/README каждой версии).

Для каждой фазы:

Фаза 0:

Range V3 перестаёт быть «0 trades»,

логика стабильна и осмысленна.

Фаза 2:

на out-of-sample периоде PF ≥ 1.2 при не ухудшающемся DD.

Фаза 3:

Adaptive Exit улучшает хвосты убытков и/или PF.

9. CONTRACTv4.8.md — ключевые элементы data-контракта (резюме)
(Подробно будет расписано в docs/CONTRACTv4.8.md, здесь только выжимка того, что план требует зафиксировать.)

Input для AI:

Явный список feature_cols:

определён в docs/dataset_spec_v*.md,

используется всеми моделями (XGB/NN/MTL),

включает:

технические,

сегментные,

positional,

sentiment,

intermarket фичи.

Output AI (3 головы):

Head 1: распределение по режимам P(regime | features).

Head 2: P(good_entry | features) — для фильтрации входов.

Head 3: P_early_exit — вероятность, что на следующем баре рационально выйти (основа Adaptive SL/TP).

Points of Interruption (приоритет):

В коде (например, v3_backtest.py / app3/backtest_ai_entry.py):

Если AI_REGIME_BLOCK (Head 1 против режима) → сигнал блокируется (и/или позиция закрывается).

Иначе, если AI_ENTRY_FILTER_BLOCK (Head 2 против входа) → сигнал блокируется.

Иначе → сигнал идёт в risk-core.

10. Связь с версиями документов
По мере выполнения:

Фаза 0:

создаётся/обновляется docs/READMEv4.8.md,

создаётся docs/CONTRACTv4.8.md (с учётом data-контракта),

при необходимости — обновляется docs/model_plan_v0.1.md → model_plan_v0.2.md.

Каждое крупное изменение логики/архитектуры:

фиксируется новой версией README/CONTRACT/plan,

старые версии не переписываются, а остаются как история.