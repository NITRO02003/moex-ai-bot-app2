# model_plan_v1.0 - план развития модели и системы

Файл: `docs2/model_plan_v1.0.md`

Это единый рабочий план `docs2`, собранный из `model_plan_v1.0` и `model_plan_v1.1`.
Мелкие уточнения без смены стадии проекта вносятся в текущий файл без новой версии имени.

## 0. Актуальная рамка планирования

- финальная цель - исследовательски честная offline/online торговая система;
- `core` - главный и единственный рабочий контур развития;
- `legacy` - архивный reference, не отдельная дорожная карта;
- песочница - промежуточный этап к online;
- Dataset A truth policy:
  - `Dataset A_research = candidates`
  - `Dataset A_policy = trades`.

## 1. Текущий принцип приоритизации

- сначала стабилизация `core` и его контрактов;
- потом честный research-to-policy контур;
- затем sandbox/online bridge;
- сравнение с legacy не является обязательным этапом roadmap.

## 0. Назначение

Этот план объединяет смысл `model_plan_v0.3` и `model_plan_v0.4`, но раскладывает его в новой системе docs2.
Ни один смысловой блок старых планов не считается потерянным:
- прежние версии лежат verbatim в `docs2/history/model_plan/`;
- здесь они собраны в новую рамку, где:
  - финальная цель - research-project и online-capable система;
  - песочница - промежуточный обязательный этап;
  - core считается главным контуром развития.

## 1. Главная цель

Главная цель проекта:
- построить исследовательски честную, архитектурно устойчивую offline/online торговую систему;
- пройти путь:
  - offline research
  - sandbox
  - online testing
  - mature research platform.

Песочница не является финальной целью.
Это только шаг перед online-тестом и частью общего research-project.

## 2. Что сохраняется из старых планов

Из `model_plan_v0.3` без потери смысла сохраняются:
- большая фазовая картина 0-5;
- идея Dataset A / Dataset B;
- AI-фильтр входа;
- Adaptive Exit;
- Intermarket / Sentiment;
- Unified MTL Core + Online / MLOps-lite;
- UI / orchestrator;
- место агентского режима в общей картине.

Из `model_plan_v0.4` без потери смысла сохраняются:
- разделение на legacy baseline и core_v4;
- dual-engine и freeze legacy;
- Vectorized State Machine;
- Geometry v1 как аналитика;
- Risk Checker v1 и Leakage Validator v1;
- перенос тяжёлых online-фич в будущие фазы.

## 3. Текущая интерпретация фаз

### Фаза A - стабилизация core
Задачи:
- удержать core от превращения в монолит;
- разграничить ownership модулей;
- убрать архитектурный дрейф `core/backtest.py`;
- привести risk config к единому source-of-truth;
- закрепить architecture blueprint и module registry.

### Фаза B - исследовательская честность
Задачи:
- формализовать Dataset A_research и Dataset A_policy;
- закрепить truth policy;
- описать честную split-политику;
- зафиксировать offline-only labels;
- усилить leakage discipline;
- ввести reproducible experiment metadata.

### Фаза C - улучшение модели
Задачи:
- диагностика входов/выходов;
- feature engineering;
- baseline ML;
- CatBoost / дальнейшие модели;
- AI gating и анализ uplift;
- regime-aware и geometry-aware improvements.

### Фаза D - sandbox path
Задачи:
- минимальный execution path;
- sandbox adapter;
- risk and monitoring;
- fail-safe;
- paper/sandbox orchestration.

### Фаза E - online-capable research platform
Задачи:
- online inference;
- online execution;
- broker adapters;
- monitoring;
- advanced alignment;
- intermarket / sentiment / later MTF;
- MLOps-lite / registry / orchestrator;
- mature research loop.

## 4. Что считаем текущим фокусом

Текущий фокус не в "ещё одной фиче", а в трёх направлениях:

1. Архитектурная устойчивость core
2. Исследовательская честность Dataset / ML
3. Подготовка к следующему roadmap без потери смысла прежних документов

## 5. Что прямо не делаем до фиксации docs2

- не распухаем дальше `core/backtest.py`;
- не вводим новые сложные online-модули;
- не плодим новые фичи без data truth policy;
- не позволяем внешним инструментам обходить патч-дисциплину;
- не подменяем исследование "красивыми" офлайн-метриками.

## 6. Поздние фазы и будущие идеи

### 6.1. AI и режимы
Сохраняется идея:
- Regime Detection как фундамент;
- AI-фильтрация входа;
- Adaptive Exit;
- MTL-подход на поздних фазах.

### 6.2. Intermarket / Sentiment / MTF
Эти идеи не отбрасываются.
Они переносятся в поздние фазы и должны войти в blueprint и roadmap как planned-модули.

### 6.3. UI / orchestrator
Смысл из `model_plan_v0.3` сохраняется:
- нужен orchestrator;
- нужен управляемый локальный UI / control layer;
- но это не должно обгонять по приоритету stabilizing core и research honesty.

### 6.4. Агентский режим
Смысл из старых документов сохраняется:
- агентский режим нужен;
- но должен работать только в рамках чёткой архитектуры и политики прав;
- read-only bridge уже допускается как аналитический слой.

## 7. Как этот plan связан с другими docs2

- архитектурная карта живёт в `architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`;
- реестр модулей живёт в `architecture/MODULE_REGISTRY_v1.0.md`;
- правила честного research живут в `policies/RESEARCH_POLICY_v1.0.md`;
- операционный контекст переезда живёт в `operations/MIGRATION_CONTEXT_v1.0.md`.

## 8. Источники без потери смысла

Сохранены:
- `docs2/history/model_plan/model_plan_v0.3_migrated.md`
- `docs2/history/model_plan/model_plan_v0.4_migrated.md`

## 100. Сохранённые legacy-планы из исходного `docs`

Ниже сохранено verbatim-содержимое старых model_plan из исходного `docs`. Это исторический слой, а не активный roadmap.

## Appendix A - preserved `docs/model_plan_v0.3.md`

# Model Plan v0.3 — подробный план развития AI-ядра (APP2+ / APP3 / MTL)

## Изменения в версии v0.3 (по сравнению с v0.2)

- Фаза 0 (Range V3 Baseline):
  - реализовано rolling-ядро Range V3 с честной каузальной логикой на 30m SBER/GAZP/LKOH/GMKN/ROSN;
  - добавлено риск-ядро с параметрами `risk_pct_per_trade`, `sl_pct`, `tp_pct`, `max_bars_in_trade`, `max_consecutive_losses`, `daily_dd_limit_pct`;
  - внедрён robust ATR-фильтр (`atr_window`, `atr_pct_min/max`, `atr_min_valid_*`) и диагностический slope-фильтр (`mask_slope_frac`), который пока не включён в `mask_range`;
  - базовый результат по последним прогонам отрицательный (PF ~0.5–0.6), но сделки есть и риск контролируем — Фаза 0 считается *почти завершённой* с фокусом на донастройке Range V3.
- Добавлен подплан по **риск-менеджменту Range V3** (Шаг 0.I) и его связь с CONTRACTv4.8.
- Зафиксирована роль внешнего инструмента **Qwen** как локального помощника для багфикса.
- Добавлен отдельный раздел (в конце плана) про интеграцию **агентского режима**, согласованный с `docs/CONTRACTv4.8.md`.



Файл: `docs/model_plan_v0.3.md`  
Связан с:
- `docs/READMEv4.7.md` (текущий миграционный чекпоинт),
- будущим `docs/READMEv4.8.md`,
- `docs/CONTRACTv4.8.md` (архитектурный и data-контракт).

Документ фиксирует **детальный план** эволюции проекта:
от текущего rule-based ядра `app2` → к многоцелевой AI-модели (MTL) в `app3`,
с учётом обсуждений, дифф-ТЗ и фидбэка по архитектуре и MLOps.

---

## 0. Организация документации и версионирование

### 0.1. Структура документации

Вся документация живёт в каталоге `docs/`:

- `docs/READMEv4.7.md` — текущий миграционный чекпоинт.
- `docs/READMEv4.8.md` — следующий чекпоинт (после завершения Фазы 0).
- `docs/CONTRACTv4.8.md` — архитектурный **и data-контракт** (Regime → Strategy → AI → Risk).
- `docs/model_plan_v0.2.md` — этот план (актуален для CONTRACTv4.8).

Дополнительно по мере развития:

- `docs/dataset_spec_v0.1.md`, `v0.2` … — форматы датасетов (Entry / In-Trade).
- `docs/features_spec_v0.1.md`, `v0.2` … — спецификации фич (в т.ч. intermarket/sentiment).
- `docs/mtl_arch_v*.md` — архитектура MTL-модели.
- `docs/mlops_plan_v*.md` — мониторинг, forward-test, управление экспериментами.

### 0.2. Связь план → код → артефакты

Для каждой фазы план фиксирует:

- **Файлы кода** (`app2/`, `app3/`), которых касается фаза.
- **Артефакты в `out/`**, которые должны появиться.
- **CLI-команды** для sanity-check.
- **Критерии успеха** (минимальные метрики / инварианты).
- Связь с CONTRACT (что должно быть отражено/обновлено в `CONTRACTv4.8+.md`).

---

## 1. Большая картинка: фазы

1. **Фаза 0 — Range V3 Baseline**  
   Zero Trades → живой baseline Range V3 на SBER 30m и других тикерах.

2. **Фаза 1 — Архитектура датасетов и таргетов (Regime / Entry / Exit)**  
   Entry / In-Trade датасеты, чёткое v1-определение `Y_exit`.

3. **Фаза 2 — Первый AI-фильтр входа + базовый Regime Detector**  
   XGBoost/CatBoost/MLP как AI-надстройка над Range V3. Метрики: PF, WinRate, Sharpe, Calmar.

4. **Фаза 3 — Adaptive Exit (Head 3)**  
   Модель `P_early_exit` (динамический выход), жёстко подчиняющаяся hard SL риск-ядра.

5. **Фаза 4 — Intermarket и Sentiment Features**  
   Сначала intermarket (структурированные ряды), затем отдельный подплан по sentiment (Transformer).

6. **Фаза 5 — Unified MTL Core + Online Learning / MLOps-lite**  
   Объединение Regime/Entry/Exit в одно MTL-ядро, осторожное периодическое дообучение, мониторинг и алертинг.

---

## 2. Фаза 0 — Range V3 Baseline (Zero Trades → Trades)

**Цель:**  
Сделать из «0 сделок» **стабильную rule-based Range V3**, которая:

- строит диапазоны,
- генерирует осмысленные сделки на SBER 30m (и далее на других тикерах),
- не обязательно прибыльна (PF может быть ≤ 1), но:
  - сделки не случайны,
  - логика соответствует идее торговли range.

### 2.0. Файлы и артефакты

Код:

- `app2/range/range_v3.py`
- `app2/range/features.py`
- `app2/range/v3_backtest.py`
- `app2/range/dataset.py`
- `app2/cli.py` (подкоманда `range-v3-backtest` уже есть)  
  + подкоманда `range-debug-segments` (новая)
- `app2/range/config.json` (секция `RangeV3`)

Артефакты:

- `out/range_v3/*_v3seg_debug*.json`
- `out/range_v3/*_v3seg_trades*.csv`
- `out/range_v3/*_v3seg_segments*.csv` (опционально)
- `out/range_v3/SEGMENTS_DEBUG_*.csv` и `.png`

Документация:

- статус Фазы 0 → `docs/READMEv4.8.md`.

---

### 2.1. Шаг 0.A — Фундамент индикаторов (ATR/MA, v1 диагностика)

Проблема:

- `ATR` и `MA` становятся полностью `NaN` из-за дыр (ночная сессия) и `min_periods=window`.

Действия (v1):

- В `range_v3.py`:

  - `_calc_atr(...)`:
    - заменить `rolling(window=window, min_periods=window)` → `min_periods=1`;
    - комментарием отметить, что это **диагностический** режим для разреженных рядов.

  - `_calc_ma(...)`:
    - аналогично `min_periods=1`.

- При необходимости — аналогично для вспомогательных MA/ATR, используемых только в Range V3.

Sanity:

```bash
python -c "import app2.range.range_v3 as r3; print('range_v3 import OK')"
Критерий:

На SBER 30m atr, ma, slope не тотально NaN.

2.1.1. Шаг 0.A.2 — Refine ATR/MA для продакшена
После оживления V3:

реализовать более аккуратный подход:

min_periods = max(1, int(window * k)) (например k=0.5),
или хранить/форвард-филить последнее валидное значение,

описать выбранную схему в docs/features_spec_v0.1.md,

сослаться в docs/CONTRACTv4.8.md (как часть data-контракта).

2.2. Шаг 0.B — Смягчение config-фильтров (AAA/AA, ADX, высота диапазона)
В app2/range/config.json (RangeV3):

диагностически ослабить:

min_tests_AAA: 3 → 2 (пример),

min_tests_AA: 2 → 1,

adx_low_max: 20 → 30,

при необходимости — чутка расширить min_range_height_pct / max_range_height_pct.

CLI:

bash
Копировать код
python -m app2.cli range-v3-backtest \
  --symbols SBER \
  --interval 30min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/DEBUG_SBER_30m \
  --tag v3seg_cfgtest
Критерий:

segments_total > 0 в debug,

есть сделки в *_trades_v3seg_cfgtest.csv.

2.3. Шаг 0.C — Расширенный debug-лог (reason + actual/threshold)
В range_v3.py:

при отбраковке сегмента логировать структуру:

json
Копировать код
{
  "reason": "too_small",
  "actual_height": 0.5,
  "threshold": 1.0
}
в *_v3seg_debug.json:

segments_total,

segments_tradable,

segments_used,

reasons_count (reason → count),

по важным причинам — агрегированные actual/threshold.

Цель: быстро понимать, «где глохнет» пайплайн.

2.4. Шаг 0.D — Визуализация сегментов (CSV + PNG)
Новая команда:

bash
Копировать код
python -m app2.cli range-debug-segments \
  --symbol SBER \
  --interval 30min \
  --date 2024-01-15 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/SEGMENTS_DEBUG_SBER_30m_2024-01-15
Функциональность:

CSV с:

L, U, M,

segment_quality,

reason (если отбраковка),

точками потенциальных входов/выходов.

Опционально: отрисовка через matplotlib в .png (цены + диапазоны + сделки).

2.5. Шаг 0.E — Baseline по SBER 30m
bash
Копировать код
python -m app2.cli range-v3-backtest \
  --symbols SBER \
  --interval 30min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/BASE_SBER_30m \
  --tag v3seg_base
Критерии:

≥ 50 сделок за период,

сегменты разных качеств (AAA/AA/A),

визуально: сделки действительно торгуют диапазоны (а не шум).

2.6. Шаг 0.F — Baseline на портфеле тикеров
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

суммарно ≥ 50–100 сделок,

ни один тикер не «немой».

2.7. Результат Фазы 0
В docs/READMEv4.8.md:

зафиксировать оживление Range V3,

базовые метрики (PF, WinRate, MaxDD) по SBER/портфелю,

ссылку на debug-инструменты (range-debug-segments),

ссылку на CONTRACTv4.8 (инварианты не нарушены).

Range V3 готов к сбору датасетов (Фаза 1) и AI-фильтрации.

**Фиксация завершения (2026-01-19):**
- выполнены прогоны baseline:
  - `out/range_v3/BASE_SBER_30m_*`
  - `out/range_v3/ALL_30m_BASE_*`
- добавлен и выполнен `range-debug-segments`:
  - `out/range_v3/SEGMENTS_DEBUG_SBER_30m_2024-01-15_*`
- READMEv4.8 будет обновлён в отдельной итерации (по лимиту контекста).

### 2.8. Шаг 0.G — Переосмысление Range V3: Offline vs Rolling

По результатам диагностики стало ясно, что текущая реализация Range V3:

- использует оффлайн-сегментацию,
- строит боксы (L/U) по **всему сегменту**, включая будущие бары,
- и поэтому даёт «невозможные» статданные (PF ~ 5, WinRate > 80%) с точки зрения real-time.

Решение:

1. **Offline Range V3** (текущий пайплайн `detect_range_segments_v3` + `build_range_box_v3` + `generate_signals_v3_for_segment`) официально переквалифицируется в:

   > *Labeler для идеальных диапазонов и сделок.*

   - Используется **только** для:
     - диагностики,
     - разметки датасетов (Фаза 1: `Y_regime`, `Y_entry`, “идеальные range-эпизоды”).
   - Не используется как честный backtest / forward-test.

2. Для честного baseline вводится отдельная rolling-реализация Range V3 (см. Шаг 0.H), которая:

   - использует только исторические данные (rolling + `shift(1)`),
   - не строит боксы по будущим барам,
   - интегрируется с существующим риск-ядром.

### 2.9. Шаг 0.H — Honest / Rolling Range V3 baseline

Цель:

- Получить честный baseline Range V3:
  - без look-ahead,
  - с caузальными признаками (rolling + `shift(1)`),
  - с теми же risk-параметрами, что и offline-версия.

Действия (см. diff-ТЗ):

1. В `app2/range/range_v3.py`:
   - переименовать текущую offline-функцию в `run_range_v3_offline_for_symbol` и явно отметить её как Labeler,
   - реализовать новую функцию `run_range_v3_for_symbol` с rolling-логикой:
     - L/U/M по окну прошлых баров,
     - фильтр высоты диапазона (`min_range_height_pct` / `max_range_height_pct`),
     - фильтр наклона MA,
     - entry/exit-логика Range V3 поверх rolling-боксов.

2. В `config.json` (`RangeV3.params`):
   - добавить параметры:
     - `mode: "rolling"` (для фиксации),
     - опционально `slope_k: 0.0005`.

3. В `app2/cli.py`:
   - гарантировать, что `range-v3-backtest` использует **rolling**-реализацию.

Критерии успеха:

- Число сделок и метрики PF/WinRate/MaxDD находятся на реалистичных уровнях (без «фантастических» значений),
- Максимальная просадка и профиль PnL согласуются между per-symbol и портфельной статистикой,
- Offline-Labeler остаётся доступен для Фазы 1 (dataset/labeling), но не используется для честного backtest.


3. Фаза 1 — Архитектура датасетов и таргетов (Regime / Entry / Exit)
Цель:
Сформировать чёткую структуру данных под MTL-ядро:

Dataset A: Entry Snapsho

### 2.10. Шаг 0.I — Настройка порогов риск-менеджмента для Range V3

Проблема:

- В текущем rolling-бейзлайне Range V3 первые несколько сделок могут давать крупные убытки при относительно небольших тейках.
- Жёсткий `max_consecutive_losses = 3` и `daily_dd_limit_pct = 0.02` приводят к тому, что весь тест раннего периода обрывается после нескольких неудачных трейдов, и итоговая статистика отражает поведение «первых пяти сделок», а не всей стратегии.
- Соотношение «средний убыток / средняя прибыль» по первым сделкам может достигать 50:1, что говорит о необходимости явной настройки SL/TP и риск-порогов.

Решение (на уровне плана):

1. В `app2/range/config.json` для блока `RangeV3.params` ввести и зафиксировать в CONTRACT набор ключевых риск-параметров:
   - `risk_pct_per_trade` — доля капитала на сделку;
   - `max_bars_in_trade` — ограничение на длительность сделки;
   - `sl_atr_mult` / `tp_atr_mult` для Range V0 и эквивалентные параметры/эвристики для Range V3 (стоп и тейк в ATR/высоте диапазона);
   - `max_consecutive_losses` — максимальное количество подряд убыточных сделок до выключения стратегии;
   - `daily_dd_limit_pct` — дневной лимит просадки.

2. Определить минимум два профиля настроек для Range V3:
   - **prod_baseline** — боевой профиль (консервативные значения по риску, реальные ограничения по серии убытков и дневной просадке);
   - **diag_fullrun** — диагностический профиль:
     - сильно ослабленные или отключённые `max_consecutive_losses` и `daily_dd_limit_pct`,
     - сохранённая логика стопов/тейков,
     - используется только для честной оценки формы распределения PnL и поведения стратегии на всём периоде.

3. В Фазе 0:
   - не подбирать риск-параметры под красоту equity-кривой,
   - а проверить, что:
     - профили prod_baseline и diag_fullrun дают схожую картину по распределению сделок,
     - разница в метриках объясняется только ограничением риска, а не изменением core-логики стратегий.

4. В дальнейшем (Фаза 2+):
   - завести механизм профилей риск-менеджмента (conservative / baseline / aggressive) с фиксированными в CONTRACT ограничениями по:
     - максимальному per-trade риску,
     - максимальной дневной просадке,
     - максимальной серии убытков.
ts — Heads 1/2 (Y_regime, Y_entry).

Dataset B: In-Trade Time-Series — Head 3 (Y_exit_t v1 = P_early_exit).

Встроить в пайплайн проверку data leakage.

3.0. Файлы и артефакты
Код:

app2/ai/dataset.py (можно начинать с расширения app2/range/dataset.py)

app2/ai/labeling.py — функции разметки таргетов

app2/features.py — источники фич

Артефакты:

out/datasets/range_v3_entry_snapshots_30m.csv

out/datasets/range_v3_intrade_timeseries_30m.csv

Документация:

docs/dataset_spec_v0.1.md — описания колонок A и B.

docs/features_spec_v0.1.md — описание базовых feature_cols.

Ссылка из docs/CONTRACTv4.8.md на feature_cols и таргеты.

3.1. Dataset A — Entry Snapshots (Head 1 & Head 2)
Entry Snapshot — 1 строка на потенциальный вход.

Примерные столбцы:

symbol, dt, interval

OHLCV: open, high, low, close, volume

технические фичи: ATR/вола, band_pos, bar_body_pct, bar_range_pct, ADX, slope MA и т.п.

сегментные фичи RangeV3: segment_quality, высота, ширина, позиция входа в боковике

режим (rule-based) — вспомогательная метка

результат сделки:

pnl_rel, bars_in_trade, MAE, MFE, ...

Таргеты:

Y_regime — режим на момент входа (по rule-based/простым правилам; baseline для Head1).

Y_entry — качество входа для Head2 (v1):

простой бинарный вариант:

1, если pnl_rel >= pnl_threshold (например, 0.001),

0 иначе;

далее можно перейти к квантильной разметке (top/bottom quantiles).

Функция:

make_entry_snapshots(...) → range_v3_entry_snapshots_30m.csv.

3.2. Dataset B — In-Trade Time-Series (Head 3, динамический Y_exit v1)
In-Trade — N строк на сделку (одна на каждый бар пока сделка открыта).

Столбцы:

всё, что доступно на баре:

symbol, dt, interval, OHLCV,

технические и сегментные фичи,

позиционные фичи:

bars_held,

текущий pnl_rel,

текущие MAE, MFE,

time_since_entry,

контекст входа (важное добавление):

подмножество entry-фич (например, тип сегмента, начальный ATR, band_pos на входе и т.п.),

чтобы Head3 понимал «в какую сделку мы вписались».

Таргет Y_exit_t v1:

интерпретация: вероятность рационального раннего выхода (P_early_exit):

v1-определение (простое, но осмысленное):

Y_exit_t = 1, если на следующем баре цена коснулась бы условного трейлинг-стопа или тейк-профита, рассчитанных по адаптивному правилу (например, entry_price ± k * ATR), и этот выход улучшает PnL по сравнению с фактическим продолжением сделки.

всё остальное: Y_exit_t = 0.

Это v1-определение фиксируется в docs/CONTRACTv4.8.md.
В следующих версиях (v2/v3) критерий «оптимальности» может быть усложнён, но строго с новой версией CONTRACT.

Функции:

label_exit_dynamic(trade) (в ai/labeling.py) → последовательность Y_exit_t.

make_intrade_timeseries(...) (в ai/dataset.py) → сбор In-Trade ряда.

Результат:

out/datasets/range_v3_intrade_timeseries_30m.csv.

3.3. Шаг 1.C — CLI для генерации датасетов
Команда:

bash
Копировать код
python -m app2.cli range-v3-make-datasets \
  --symbols SBER GAZP LKOH GMKN ROSN \
  --interval 30min \
  --out-prefix out/datasets/RANGE_V3_30m
Результат:

RANGE_V3_30m_entry_snapshots.csv

RANGE_V3_30m_intrade_timeseries.csv

3.4. Шаг 1.D — Валидатор data leakage
Добавить функцию-валидатор в ai/dataset.py:

проверяет, что:

никакая фича не использует high/low/close будущих баров,

при join intermarket/sentiment — используется только информация с dt' <= dt текущего бара.

Эта функция должна запускаться из CLI (например, флаг --validate) и логировать любые подозрительные фичи.

3.5. Шаг 1.E — Контроль балансировки и размеров
Проверить:

размер Entry vs количество сделок,

размер In-Trade vs суммарная длительность сделок,

баланс классов Y_entry и Y_exit_t.

3.6. Результат Фазы 1
Есть два специфицированных датасета,

описания в docs/dataset_spec_v0.1.md + ссылка в CONTRACT,

Y_exit v1 формально определён,

есть leakage-валидатор.

4. Фаза 2 — Первый AI-фильтр входа + базовый Regime Detector
Цель:
Построить первые практичные модели:

Head 1 — Regime Detector,

Head 2 — Entry Filter,

и интегрировать их в backtest с приоритетом Head1 → Head2 → Risk Core.

4.0. Файлы и артефакты
Код:

app3/train_entry_models.py (или подкоманда app3/cli.py),

app3/backtest_ai_entry.py (или AI-режим в существующем backtest),

app3/config_entry_models.toml.

Артефакты:

models/head1_regime_xgb.bin ( или аналог),

models/head2_entry_filter_xgb.bin,

out/reports/entry_models_metrics.json,

out/reports/feature_importance_head2.csv.

Метрики для сравнения:

PF, WinRate,

Sharpe Ratio, Calmar Ratio, MaxDD.

4.1. Head 1 — Regime Detector
Данные:

Entry Dataset A (Y_regime),

на старте Y_regime размечается простыми rule-based правилами (ADX, вола, и т.п.).

Модель:

XGBoost/CatBoost (табличные данные),

вход: подмножество feature_cols,

выход: вектор вероятностей по режимам.

4.2. Head 2 — Entry Filter
Данные:

Entry Dataset A (Y_entry).

Модель:

XGBoost/CatBoost/MLP.

Задача:

предсказывать P(good_entry),

использовать как фильтр для Range V3.

4.3. Интеграция в backtest (Points of Interruption v1)
Логика (подробно в CONTRACT):

python
Копировать код
# Псевдокод
regime_probs = Head1(features)
if regime_probs['range'] < regime_min_prob:
    AI_REGIME_BLOCK = True
else:
    AI_REGIME_BLOCK = False

entry_prob = Head2(features)  # P(good_entry)
AI_ENTRY_FILTER_BLOCK = entry_prob < entry_min_prob

if AI_REGIME_BLOCK:
    # блокируем сигнал, при необходимости закрываем open Range-сделки
    pass
elif AI_ENTRY_FILTER_BLOCK:
    # блокируем именно этот вход
    pass
else:
    # сигнал идёт в risk-core (rule-based открытие)
    open_position()
CLI:

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
Сравниваем baseline (Ф0) vs AI-режим (Ф2):

PF,

WinRate,

Sharpe,

Calmar,

MaxDD,

число сделок.

Ориентир Ф2:

PF ≥ 1.2 на out-of-sample при не ухудшенном DD.

5. Фаза 3 — Adaptive Exit (Head 3, динамический выход)
Цель:
Модель Head 3, дающая P_early_exit на каждом баре сделки.

5.0. Данные
In-Trade Dataset B (Y_exit_t v1),

контекст входа включён.

5.1. Модель Head 3 (v1)
Модель: Gradient Boosting / MLP (последовательный контекст можно сначала игнорировать).

Вход: текущие фичи + позиционные + контекст входа.

Выход: P_early_exit.

5.2. Интеграция с Risk Core
Контракт (подробно в CONTRACT):

Head 3 может инициировать досрочный выход,

никогда не может:

отменять или отодвигать hard SL,

увеличивать риск позиции.

Псевдокод:

python
Копировать код
if P_early_exit > early_exit_threshold:
    close_position()  # полностью или частично согласно конфигу
# hard SL/TP от risk-core остаётся железным
5.3. Метрики эффекта Head 3
PF, WinRate,

MaxDD,

распределение PnL (особенно хвосты убытков),

средняя и медианная длительность сделок.

6. Фаза 4 — Intermarket и Sentiment
Приоритет:

Сначала intermarket (структурированные числовые ряды).

Затем sentiment как отдельный подплан (связанный с Transformer, сбором текстов и fine-tuning).

6.1. Intermarket
Действия:

определить список intermarket-инструментов (индексы, FX, сырьё),

модуль app2/intermarket.py:

загрузка/синхронизация,

расчёт фич (спрэды, относительная сила, корреляции),

интеграция в features.py,

обновление features_spec_v*.md и dataset_spec_v*.md.

6.2. Sentiment
Отдельный подплан (в будущем docs/sentiment_plan_v*.md):

сбор новостей/отчётов,

локальный Transformer (RuBERT/мультиязычный),

агрегирование во времени (rolling sentiment, Δ sentiment, флаги событий),

интеграция без lookahead.

6.3. Обновление датасетов
После добавления intermarket/sentiment:

перегенерировать Entry / In-Trade datasets,

обновить CONTRACT (новая версия, если меняется структура feature_cols).

7. Фаза 5 — Unified MTL Core + Online Learning / MLOps-lite
Цель:
Объединить опыт отдельных моделей в MTL-ядро и выстроить лёгкий MLOps вокруг.

7.1. Unified MTL
Общий encoder,

головы:

Head 1: Regime,

Head 2: Entry,

Head 3: Exit.

Постепенность:

сначала общий encoder для Head1/Head2 (Entry Dataset),

Head3 подключать позже (In-Trade).

7.2. Online Learning
Политика:

начинать с периодического re-training (по расписанию) на новых данных,

не делать онлайн-обучение «на каждом баре»,

при необходимости → отдельный план с replay-buffer, регуляризацией и жёстким мониторингом.

7.3. MLOps-lite: мониторинг, алертинг, эксперименты
Уже с Фазы 2:

завести простой трекинг экспериментов:

хотя бы CSV/JSON-лог с:

конфигом,

датой,

метриками,

комментариями,

мониторинг:

data drift: распределения ключевых фич по времени,

concept drift: падение метрик на скользящем окне,

алертинг:

хотя бы в лог/файл,

потенциально в Telegram/почту.

Перед переходом в следующие фазы:

предусмотреть walk-forward / paper trading как условие продвижения (например, из Ф2 → Ф3, Ф3 → продакшен).

8. Метрики и цели
Долгосрочные ориентиры:

WinRate ≥ 65%,

PF ≥ 2.0,

приемлемый MaxDD (фиксируется в CONTRACT и README конкретной версии).

Фазовые цели:

Ф0: Range V3 ≠ 0 trades, логика осмысленна.

Ф1: консистентные датасеты без leakage.

Ф2: AI-фильтр улучшает PF/Sharpe/Calmar vs baseline.

Ф3: Adaptive Exit срезает хвосты убытков/улучшает PF без ухудшения DD.

Ф4–Ф5: intermarket/sentiment и MTL стабильно улучшают риск-профиль стратегии.

## 9. Фаза UI0 — локальный UI и оркестратор

**Цель:**  
Сделать минимальный, но полезный интерфейс для работы с моделью, который:

- отображает результаты (дашборды по backtest/forward/AI-моделям),
- позволяет **запускать** функционал модели:
  - backtest / sandbox / live-торговлю,
  - подключение к API брокера,
  - остановку сессий,
- работает локально и/или через Telegram, не нарушая архитектурных и риск-инвариантов.

Фаза UI0 **идёт после основных фаз по логике и моделям** (Ф0–Ф5), чтобы не отвлекаться от оживления и стабилизации стратегии.

### 9.1. Архитектура управления

Выделяются два уровня:

1. **Оркестратор / API-слой**  
   - инструмент: **FastAPI** (или аналогичный лёгкий HTTP-фреймворк на Python),
   - отвечает за:
     - эндпоинты вида:
       - `POST /backtest/start`
       - `POST /sandbox/start`
       - `POST /live/start`, `POST /live/stop`
       - `POST /broker/connect`
     - запуск CLI-команд:
       - `python -m app2.cli ...`
       - `python -m app3 ...`
     - ведение простого журнала задач и статусов (в `out/logs/` или небольшой БД).

   - длинные задания:
     - выполняются как фоновые задачи (BackgroundTasks/отдельный процесс),
     - **не** переносят логику внутрь API, а только оркестрируют вызовы существующих модулей.

2. **Клиентские интерфейсы**  
   Используют **один и тот же FastAPI**, не дублируя бизнес-логику.

   - **Локальный UI (пульт управления)**:
     - инструмент: **Streamlit**,
     - назначение:
       - отображение дашбордов по файлам из `out/` (equity, trades, метрики),
       - формы и кнопки:
         - «Запустить backtest»,
         - «Запустить sandbox/forward»,
         - «Старт/Стоп live» и т.п.,
       - под капотом: отправка HTTP-запросов к FastAPI и опрос статусов задач.

   - **Telegram-бот**:
     - инструмент: `aiogram` или `python-telegram-bot`,
     - функции:
       - команды `/status`, `/backtest`, `/start_live`, `/stop_live` и т.п.,
       - пересылка этих команд в FastAPI (POST/GET),
       - отображение пользователю краткого статуса / последних метрик.

### 9.2. Соблюдение производственных правил

- Оркестратор и UI:

  - **не создают** новых `out/` внутри `app2/`/`app3/`,
  - читают и пишут результаты только в корневой `out/`,
  - используют только официальные интерфейсы:
    - CLI-команды (`python -m app2.cli ...`, `python -m app3 ...`),
    - либо тонкие обёртки над ними.

- Все действия UI/Telegram через FastAPI **подчиняются CONTRACT**:

  - не могут обходить Risk Core,
  - не могут менять риск-лимиты,
  - не могут запускать «сомнительные» режимы в обход конфигов.

### 9.3. Положение в дорожной карте

- Фаза UI0 запускается **после**:

  - оживления и стабилизации Range V3 (Ф0),
  - появления консистентных датасетов и базовых моделей (Ф1–Ф3),
  - когда уже есть что смотреть и чем управлять.

- Детализированная структура:

  - файлы `api/main.py`, `ui/app.py`,
  - конкретные эндпоинты и команды Streamlit/Telegram

  будет описана в отдельном документе (например, `docs/ui_orchestration_plan_v0.1.md`) **после завершения критичных фаз по стратегии**.

---

## 6. Агентский режим — место в плане

### 6.1. Зачем нужен агентский режим

Агентский режим рассматривается как инструмент для:
- автоматизации рутинных экспериментов (серии бэктестов, свипы параметров),
- построения отчётов и мониторинга деградации стратегий,
- постепенной подготовки к полуавтономной работе `app3` в песочнице брокера.

Он **не заменяет** ручной режим разработки и не имеет права нарушать CONTRACT.

### 6.2. Привязка к фазам плана

- **До завершения Фазы 0 и 1**:
  - агентский режим *не* включается;
  - все изменения делаются вручную через дифф-ТЗ и патчи.
- **После Фазы 1 (готовы Entry / In-Trade датасеты, нет data leakage)**:
  - агент может запускать фиксированные сценарии бэктеста и собирать отчёты по Range/Trend/MeanRev.
- **После Фазы 2–3 (есть AI-ядро и Adaptive Exit)**:
  - агент получает ограниченный доступ к:
    - свипу гиперпараметров,
    - переобучению моделей,
    - анализу форвард-тестов.

Во всех случаях:
- список разрешённых действий и лимиты шагов фиксируются в `docs/CONTRACTv*.md`,
- каждое изменение кода, сделанное агентом, должно логироваться и подтверждаться человеком.

### 6.3. Связанные документы

- Детальный контракт по агентскому режиму: `docs/CONTRACTv4.8.md`, раздел «ТЗ для агентского режима».
- Общая архитектура AI-ядра и MTL: текущий документ `docs/model_plan_v0.3.md`.


---

## Appendix B - preserved `docs/model_plan_v0.4.md`

# Model Plan v0.4 — план развития Range ядра (APP2+ / core_v4 / online-модель)

Файл: `docs/model_plan_v0.4.md`  

## Изменения в версии v0.4 (по сравнению с v0.3)

- Фаза 0 разделена на:
  - **Фазу 0.1 — Legacy Baseline** (существующий Range V3 rolling + риск-ядро, PF<1, но честный baseline);
  - **Фазу 0.2 — Range core_v4 (offline)**.
- Чётко разведены зоны ответственности:
  - `app2` — текущий offline-core и legacy;
  - `app3` — будущая online-модель (event-driven, microstructure, multi-TF и т.п.).
- Предложенный коллегой «большой план» переработан:
  - сложные элементы (microstructure, полный event-driven, AAA/AA/A как боевой фильтр, сложные leakage-проверки)
    перенесены в будущие фазы **online-модели**;
  - в Phase 0.2 остаётся реалистичный по объёму набор задач для core_v4.
- Добавлены уточнения, согласованные с v0.3:
  - улучшения входов относятся к завершению Phase 0 (slope-гейтинг, симметрия long/short);
  - Dataset A должен строиться по **кандидатам входа**, а не только по фактическим сделкам;
  - эффективность Entry‑AI оценивается по PF/Calmar **после фильтрации входов**, а не по F1.
- Зафиксирован фокус: базовый TF = 10min, старшие TF (30min/1h) — отдельный трек настройки.
- Обновлена роль LLM-инструментов:
  - Qwen — зафиксирован как локальный багфиксер/экспериментатор;
  - ChatGPT-агент — как инструмент будущих фаз с жёсткими ограничениями (см. CONTRACTv4.9).

---

## 0. Организация документации

Актуальные файлы:

- `docs/READMEv4.10.md` — текущий статус и старт Phase 1 (датасеты);
- `docs/CONTRACTv4.10.md` — актуальный контракт (core_v4 + Dataset A/B);
- `docs/model_plan_v0.4.md` — этот план.

Все предыдущие версии (`READMEv4.7`, `READMEv4.8`, `READMEv4.9`, `CONTRACTv4.8`, `CONTRACTv4.9`, `model_plan_v0.2`, `v0.3`) остаются как история.

---

## 1. Фаза 0.1 — Legacy Baseline (завершена)

Цель: иметь честный, стабильный baseline для Range V3, пусть и убыточный.

Состояние:

- rolling Range V3 реализован для 30m SBER/GAZP/LKOH/GMKN/ROSN;
- риск-ядро реализовано с параметрами:
  - `risk_pct_per_trade`, `sl_pct`, `tp_pct`,
  - `max_bars_in_trade`, `max_consecutive_losses`, `daily_dd_limit_pct`;
- введён robust ATR-фильтр и диагностический slope-фильтр;
- стратегии дают сделки, но PF портфеля < 1, WinRate ~35–40%;
- Qwen подключён как вспомогательный инструмент.

Фаза 0.1 считается **завершённой**, когда:

- legacy Range V3 (в виде `range_v3_legacy.py`) стабильно воспроизводит эти результаты;
- формально описан контракт (CONTRACTv4.9) и минимальные правила leakage/risk.

**Фиксация завершения (2026-01-19):**
- прогон legacy baseline выполнен на SBER 30m и портфеле SBER/GAZP/LKOH/GMKN/ROSN;
- артефакты сохранены в `out/range_v3/BASE_SBER_30m_*` и `out/range_v3/ALL_30m_BASE_*`;
- диагностический инструмент `range-debug-segments` добавлен и использован.

---

## 2. Фаза 0.2 — Range core_v4 (offline)

Цель: выделить аккуратное модульное офлайн-ядро Range, подготовленное к будущему online, но без избыточной сложности.

### 2.1. Подфаза 0.2.A — Dual-engine и freeze legacy

- Переименовать текущие файлы:
  - `range_v3.py` → `range_v3_legacy.py`,
  - `v3_backtest.py` → `v3_backtest_legacy.py`.
- В `app2/cli.py` добавить параметр `--engine legacy|core` для `range-v3-backtest`.
- Зафиксировать эталонный набор результатов legacy:
  - `*_stats.json`, `*_trades.csv` для 30m SBER (и набора тикеров SBER/GAZP/LKOH/GMKN/ROSN).
- Любые новые идеи по Range реализуются в core_v4, а legacy трогаем только для bugfix/диагностики.

### 2.2. Подфаза 0.2.B — Vectorized State Machine v1

Задача: перенести текущую (даже убыточную) логику Range в новый core в виде прозрачного векторизованного автомата.

- Создать в `app2/range/core/` модуль `state_machine.py`:
  - определить набор состояний (например, `FLAT`, `RANGE_CANDIDATE`, `RANGE_ACTIVE`, `IN_TRADE`);
  - реализовать переходы в виде операций над колонками DataFrame (без циклов по барам).
- На этом этапе **не улучшаем стратегию**, только переносим:
  - условие появления диапазона,
  - критерии входа/выхода,
  - гейтинг по height/ATR и др.
- Критерий успеха:
  - PF/WinRate core_v4 ≈ legacy на том же датасете (допускается небольшая разница, но без явных регрессий);
  - различия документируются (отличия из-за, например, более жёсткой модели исполнения).

### 2.2.C — Entry robustness patch (до Фазы 2)

Цель: довести baseline‑входы до устойчивого уровня (не обязательно прибыльного),
чтобы AI‑фильтр входа строился поверх **осмысленного** кандидата.

Действия:
- включить slope‑фильтр в `mask_range` (сейчас он только в диагностике);
- добавить симметрию входов (short‑сигналы по зоне U);
- прогонять full‑run профиль риска для честной формы распределений.
 - фиксировать базовый TF = 10min; 30min/1h не блокируют Phase 2, идут в отдельной настройке.

Критерии:
- нет «немых» тикеров (сделки есть у каждого);
- PF/WinRate/Calmar на полном периоде реалистичны и воспроизводимы;
- baseline‑входы стабилизированы для Phase 2 (Entry‑AI).

Тюнинг старших TF (30min/1h), если потребуется:
- отдельный прогон после фиксации 10min;
- параметры фокуса: `slope_k`, `entry_zone_alpha`, `min_confirmations`,
  `lock_bars_after_breakout`, `atr_pct_min/max`, `min_range_height_pct/max_range_height_pct`.

### 2.3. Подфаза 0.2.C — Geometry v1 (аналитика, не фильтр)

Задача: формализовать геометрию диапазонов, но сначала использовать её только как аналитический слой.

- Создать `app2/range/core/geometry.py`:
  - функции для расчёта L/U/H/M;
  - примитивная классификация диапазонов (например, A / INVALID);
  - опционально — первичная AAA/AA/A-классификация как **feature**, а не как фильтр.
- Подключить geometry к state machine так, чтобы:
  - мог записываться тип диапазона в `debug_info`,
  - можно было строить отчёты PF/WinRate по классам диапазона.
- На этом этапе **не использовать AAA/AA/A для фильтрации сделок** — только анализ.

### 2.4. Подфаза 0.2.D — Risk Checker v1 и Leakage Validator v1 (фиксируем)

Задача: укрепить доверие к бэктесту и контролю рисков.

- Реализовать `risk.py` для core_v4:
  - вход по open_{t+1};
  - пессимистичный выбор между SL/TP на одном баре (всегда SL);
  - простая обработка gap’ов без моделирования стакана.
- Реализовать `leakage_validator.py` (можно в `app2/tools/` или аналогичном месте):
  - проверка на отсутствие `shift(-k)` в rolling-части;
  - базовая диагностика подозрительных future-колонок.
- Добавить CLI-режимы:
  - `python -m app2.cli range-core-backtest --engine core ...`;
  - `python -m app2.cli leakage-check ...`.

**Фиксация v0.2.D (текущее состояние):**
- `core/risk.py` вынесен в отдельный модуль и покрыт тестами edge-case (gap/SL/TP, daily DD, consecutive losses);
- добавлены модульные блоки для расчётов диапазона в `core/blocks.py`;
- `leakage_validator.py` усилен (severity, allowlist, auto-exclude), доступен через CLI.

Фаза 0.2 считается **завершённой**, когда:

- существует рабочий core_v4, дающий осмысленные результаты и понятный debug;
- есть минимальный, но полезный leakage-validator;
- risk core реализован и согласован с CONTRACTv4.10.

---

## 2.5. Фаза 1 (датасеты) — статус и рекомендации

Сделано:
- Dataset A (Entry Snapshots) сгенерирован из Range V3 baseline.
- Dataset B (In-Trade Time-Series) реализован (генератор + CLI).
- `y_exit` v1 зафиксирован в контракте:
  - next-bar `pnl_rel` против финального `trade_pnl_rel` + порог.
- Сформирован отчёт по `y_exit`:
  - `out/range_v3/ALL_30m_BASE_intrade_y_exit_report.json`
  - `out/range_v3/ALL_30m_BASE_intrade_y_exit_by_symbol.csv`
- Leakage-validator прогнан:
  - `out/leakage_report.json`

Рекомендации:
- Проверить долю `y_exit=1`, распределения `bars_held`, `pnl_rel`.
- Зафиксировать выводы по `out/leakage_report.json` (ожидаемое `shift(-1)` в `dataset.py` для offline-лейбла).
- Следующий шаг: расширить Dataset A до **кандидатов входа** (касание зон/условий),
  чтобы избежать selection bias и обеспечить устойчивый Head2‑Entry.
- Y_entry привязать к forward‑горизонту (MAE/MFE/return за N баров), а не к факту сделки.
  - Базовый режим: **B3 (mfe/mae)** — риск‑aware лейбл (MFE>=X, MAE>=‑Y).
  - Резервный режим: **B4 (quantile)** — включать на тренд‑участках при сильных сигналах.

---

## 3. Фазы 2–5 (как в v0.3, дорожная карта без изменений)

Фазы ниже остаются **идентичными v0.3**, v0.4 лишь уточняет Phase 0–1.

2) **Фаза 2 — Первый AI‑фильтр входа + Regime Detector**  
- Head1/Head2 строятся на Dataset A и фильтруют входы Range V3.  
- Метрики сравнения: PF/WinRate/Calmar/MaxDD **после gating**.

3) **Фаза 3 — Adaptive Exit (Head 3)**  
- Модель P_early_exit на Dataset B, подчинена hard‑risk правилам.

4) **Фаза 4 — Intermarket/Sentiment фичи**  
- Добавляются после стабилизации входов и exit‑логики.

5) **Фаза 5 — Unified MTL Core + MLOps‑lite**  
- Объединение Head1/2/3, мониторинг, walk‑forward.

## 4. Фаза 1+ — Online-модель и расширенные фичи (планы на будущее)

Всё, что в текущем обсуждении было признано «тяжёлым для текущего момента»,
переносится в будущие фазы, связанные с online-моделью (скорее в `app3`):

- **Event-driven engine**:
  - очереди событий, подписчики, realtime state;
  - интеграция с брокером и холодной резервной логикой.

- **Microstructure-aware риск**:
  - учёт bid/ask спреда, глубины стакана;
  - отдельные фичи для «выносов» стопов и slippage.

- **Multi-timeframe Alignment Layer**:
  - аккуратное совмещение 30m / 1h / 4h / daily баров;
  - строгие правила каузальности при cross-TF-агрегации.

- **Geometry AAA/AA/A как боевой фильтр**:
  - использование класса диапазона для включения/отключения стратегии;
  - фильтрация «плохих» структур до генерации сигналов.

- **Leakage Validator v2/v3**:
  - более умный анализ графа зависимостей,
  - автоматизация проверки новых пайплайнов.

- **LLM-агентский режим**:
  - автоматические последовательности бэктестов,
  - автогенерация отчётов,
  - контролируемый рефакторинг.

Запуск этих фич возможен только после того, как:

- core_v4 демонстрирует устойчивый PF≥1 на offline-тестах;
- leakage-validator v1 не находит критичных проблем;
- риск-ядро признано стабильным.

---

Этот план v0.4 задаёт реальный, исполнимый маршрут от текущего убыточного Range baseline к
модульному core_v4 и далее к online-модели, не бросаясь сразу в избыточную архитектуру.

