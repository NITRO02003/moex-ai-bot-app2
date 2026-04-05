# READMEv1.0 - MOEX APP2+ / docs2 source of truth

Файл: `docs2/READMEv1.0.md`

Этот файл открывают первым.
Он объединяет смысл прежних README v4.7-v4.10 и вводит новую систему `docs2/`.

## 1. Назначение docs2

`docs2/` - новая система документации проекта.
Её задача:

- не потерять смысл прежних README / CONTRACT / model_plan;
- разложить документы по ролям;
- дать стабильный контекст для нового чата и для внешних аналитиков;
- зафиксировать, что проект уже делает боевую модель, а не только "строит каркас".

Все прежние документы сохранены без потери смысла в `docs2/history/`.

## 2. Текущий статус проекта

Текущий реальный статус по состоянию на миграцию в `docs2`:

- legacy Range V3 существует и сохраняется как reference / sanity-check;
- core в `app2/range/core/` уже является главным контуром развития;
- Dataset / ML pipeline уже начат и считается частью боевой архитектуры;
- песочница рассматривается как обязательная промежуточная стадия перед online-тестом;
- главная финальная цель проекта - исследовательски честная offline/online торговая платформа, а не просто быстрый запуск в песочнице.

Ключевая коррекция относительно старых формулировок:
- мы уже не "строим каркас";
- мы уже строим боевую модель, но с незавершённым разведением зон ответственности.

## 3. Что перенесено из прежних README

Из README v4.7-v4.10 в `docs2` сохранены следующие смысловые блоки:

1. Хронология перехода:
   - от zero-trades и диагностики Range V3;
   - к rolling baseline;
   - к risk core;
   - к freeze legacy + dual-engine;
   - к core_v4;
   - к старту Phase 1 с Dataset A/B, baseline ML, CatBoost, feature sweeps.

2. Практические правила:
   - патчи только архивом и только изменённые файлы;
   - сомнение в актуальности файла = стоп правки;
   - все артефакты только в корневой `out/`;
   - self-check обязателен;
   - не копировать большие куски кода в чат без запроса;
   - спорные архитектурные упрощения только через обсуждение.

3. Правила структуры проекта:
   - `data/` - сырые данные;
   - `processed/` - агрегированные бары;
   - `out/` - все результаты;
   - `app2/` - основной боевой контур;
   - `app3/` - будущая online / модульная ветка.

4. Роль LLM-инструментов:
   - ChatGPT - для анализа, планирования, контрактов и патчей;
   - Qwen - локальный reviewer / debugger;
   - `agent_hands` - read-only bridge к локальной модели.

5. Текущий исследовательский фокус:
   - core важнее legacy;
   - архитектурная устойчивость важнее локального удобства;
   - исследовательская честность важнее красивых офлайн-метрик;
   - песочница - только этап пути к online.

## 4. Source of truth в docs2

Читать в таком порядке:

1. `docs2/READMEv1.0.md`
2. `docs2/DOC_INDEX_v1.0.md`
3. `docs2/architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
4. `docs2/CONTRACTv1.0.md`
5. `docs2/model_plan_v1.0.md`
6. `docs2/policies/RESEARCH_POLICY_v1.0.md`
7. `docs2/policies/LLM_WORKFLOW_v1.0.md`
8. `docs2/operations/MIGRATION_CONTEXT_v1.0.md`

## 5. Что лежит в docs2/history

В `docs2/history/` лежат verbatim-копии ключевых документов старой системы:

- `history/readme/READMEv4.7_migrated.md`
- `history/readme/READMEv4.8_migrated.md`
- `history/readme/READMEv4.9_migrated.md`
- `history/readme/READMEv4.10_migrated.md`
- `history/contract/CONTRACTv4.8_migrated.md`
- `history/contract/CONTRACTv4.9_migrated.md`
- `history/contract/CONTRACTv4.10_migrated.md`
- `history/model_plan/model_plan_v0.3_migrated.md`
- `history/model_plan/model_plan_v0.4_migrated.md`

Они нужны, чтобы ни один смысловой пункт не потерялся при переходе на новую структуру.

## 6. Где мы сейчас по факту

Если смотреть не по старым формулировкам, а по реальному `app2`, то у проекта уже три слоя:

1. Legacy lane
2. Core lane
3. Dataset / ML lane

Это значит, что ближайшая задача - не "добавить ещё одну фичу", а:
- зафиксировать целевую архитектуру;
- закрепить зоны ответственности модулей;
- определить исследовательски честную постановку Dataset A/B;
- и только потом переписывать roadmap и контракты под новую систему.

## 7. Что дальше

Следующие документы, которые становятся обязательными:

- `architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
- `architecture/MODULE_REGISTRY_v1.0.md`
- `policies/RESEARCH_POLICY_v1.0.md`
- `policies/LLM_WORKFLOW_v1.0.md`
- `operations/MIGRATION_CONTEXT_v1.0.md`

После этого:
- обновляется roadmap;
- обновляется контракт;
- затем идут кодовые патчи.

## 8. Что не хранится в README

В этом файле не храним:
- детальный перечень модулей;
- полный data contract Dataset A/B;
- правила leakage;
- детальный план по фазам;
- операционные notes по локальному окружению;
- длинные отчёты по экспериментам.

Для этого есть другие документы `docs2/`.
