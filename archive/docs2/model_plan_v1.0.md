# model_plan_v1.0 - план развития модели и системы

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
