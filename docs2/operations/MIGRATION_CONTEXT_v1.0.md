# MIGRATION_CONTEXT_v1.0 - контекст для переезда в новый чат

## 0. Назначение

Этот файл нужен, чтобы новый чат сразу понял:
- где проект находится сейчас;
- какие документы считать source of truth;
- какие зоны кода активны;
- какие риски критичны;
- какой пакет файлов обязателен для миграции.

## 1. Что читать первым

Обязательный порядок чтения:
1. `docs2/READMEv1.0.md`
2. `docs2/DOC_INDEX_v1.0.md`
3. `docs2/policies/PROJECT_RULES_v1.0.md`
4. `docs2/CONTRACTv1.0.md`
5. `docs2/model_plan_v1.0.md`
6. `docs2/policies/LLM_WORKFLOW_v1.0.md`
7. `docs2/architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
8. `docs2/architecture/MODULE_REGISTRY_v1.0.md`

## 2. Current state snapshot

Текущий общий статус:
- `core` - главный и единственный рабочий контур развития;
- `legacy` - архивный reference only;
- dataset/ML pipeline уже начат и является частью боевой архитектуры;
- project goal = исследовательски честная offline/online система;
- песочница не является главной целью.

## 3. Active code zones

На текущем срезе активны:
- `app2/cli.py`
- `app2/range/config.json`
- `app2/range/core/*`
- `app2/range/dataset.py`
- `app2/range/make_datasets.py`
- `app2/range/baseline_ml.py`
- `app2/range/catboost_train.py`
- `app2/tools/leakage_validator.py`

Условная read-only зона:
- `agent_hands.py`, только если файл приложен в текущем пакете.

## 4. Current critical risks

1. перегруженный `core/backtest.py`
2. risk config drift между `params` и `risk_profiles`
3. ambiguity Dataset A truth
4. leakage risk around offline labels and dataset joins
5. скрытый architectural drift между core modules
6. потеря контекста при миграции, если не читать rules first

## 5. Current research decisions

Принятые решения:
- `core` важнее legacy;
- legacy не тянет на себя отдельный roadmap;
- Dataset A делится на `A_research` и `A_policy`;
- замечания внешних аналитиков включаются в план, если с ними согласились;
- песочница - только шаг на пути к online;
- agent_hands остаётся read-only.

## 6. Required artifact pack

Для переезда прикладывать:
- актуальные docs2 source-of-truth файлы;
- `app2/` активные зоны;
- `app2/range/config.json`;
- минимальный набор `out/` артефактов по последним прогонам;
- при необходимости notes от внешних reviewers.

## 7. Что нельзя забывать новому чату

- проект уже делает боевую модель, а не только scaffold;
- sandbox не является главной целью;
- legacy больше не основной объект развития;
- docs2 - новая система source of truth;
- любые дальнейшие roadmap и патчи должны опираться на docs2, а не на старые ощущения о проекте.
