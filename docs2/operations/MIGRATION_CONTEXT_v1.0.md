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
3. `docs2/architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
4. `docs2/CONTRACTv1.0.md`
5. `docs2/model_plan_v1.0.md`
6. `docs2/policies/RESEARCH_POLICY_v1.0.md`
7. `docs2/policies/LLM_WORKFLOW_v1.0.md`

## 2. Current state snapshot

Текущий общий статус:
- legacy остаётся только как reference;
- core является главным объектом развития;
- dataset/ML pipeline и robustness-инструменты уже существуют, но не доказали наличие edge;
- project goal = research-project с offline, sandbox и online стадиями;
- песочница не является главной целью;
- ближайший следующий шаг - stabilization data contract для core-контура: `data/` и `processed/` должны хранить только реальные бары без synthetic empty slots; только после этого допустим повторный forensic по active core.

## 3. Active code zones

На текущем срезе активны:
- `app2/cli.py`
- `app2/range/core/*`
- `app2/range/dataset.py`
- `app2/range/make_datasets.py`
- `app2/range/baseline_ml.py`
- `app2/range/catboost_train.py`
- `app2/tools/leakage_validator.py`
Дополнительные утилиты, появившиеся в рамках стабилизации core и оценки фильтров:
 - `app2/range/core/baseline_core.py` — формирует эталонный baseline на наборе тикеров;
 - `app2/range/core/evaluate_gating.py` — сравнивает baseline и запуск с AI‑фильтром, рассчитывает покрытие и delta по метрикам;
 - `app2/range/core/summary_core.py` — собирает сводную таблицу из нескольких агрегированных статистик;
 - `baseline_symbols.txt` — динамический список тикеров для baseline (можно редактировать без изменения кода).

Legacy зона:
- `app2/range/range_v3_legacy.py`
- `app2/range/v3_backtest_legacy.py`

Локальные LLM-агенты / bridge-слои:
- конкретная реализация может меняться;
- любой такой агент по умолчанию считается read-only аналитическим инструментом.

## 4. Current critical risks

1. перегруженный `core/backtest.py`
2. risk config drift между `params` и `risk_profiles`
3. ambiguity Dataset A truth
4. leakage risk around offline labels and dataset joins
5. скрытый architectural drift между core modules
6. отрицательный baseline - если эталонная стратегия убыточна, оценка AI-фильтра будет вводить в заблуждение; baseline должен быть выше среднего, иначе фильтр не имеет смысла.
7. возможно отсутствие edge в самой логике Range V3; в таком случае дальнейшие надстройки только маскируют проблему и должны быть заморожены.

## 5. Current research decisions

Принятые решения:
- core важнее legacy;
- Dataset A делится на `A_research` и `A_policy`;
- замечания внешних аналитиков включаются в план, если мы с ними согласны;
- песочница - только шаг на пути к online;
- локальные LLM-агенты рассматриваются как вспомогательные инструменты и не считаются source of truth.

Дополнительно принято:
- baseline стратегии должен демонстрировать положительное ожидание перед тем, как сравнивать его с AI-фильтром;
- для построения baseline используется динамический список тикеров из `baseline_symbols.txt`;
- truth-policy и dataset_kind обязательно фиксируются в meta-манифесте каждого dataset;
- все результаты baseline и forensic-диагностики сохраняются в `out/`;
- до завершения forensic-цикла запрещены новые модели, calibration, Regime Engine, sandbox и online-движение;
- если найден живой сегмент, сначала вводится тупой rule-filter, и только потом допускается ML-селектор;
- если живой сегмент не найден, ветка Range V3 замораживается и дальнейшая работа переносится на новую стратегическую гипотезу.


Текущий порядок работ:
1. сначала forensic-отчёт и разрезы по существующим артефактам baseline;
2. затем проверка временной устойчивости найденных сегментов (`early` / `mid` / `late`);
3. после этого одно из решений: `rule-filter`, `exit redesign` или `freeze Range V3`;
4. только при наличии живого rule-based сегмента допускается новый ML-пакет;
5. sandbox/online не являются ближайшим шагом до завершения этого цикла.

## 6. Required artifact pack

Для переезда прикладывать:
- актуальные docs2 source-of-truth файлы;
- `app2/` активные зоны;
- `app2/range/config.json`;
- минимальный набор `out/` артефактов по последним прогоном;
- при необходимости notes от внешних reviewers.

При наличии forensic-цикла по baseline:
- файл `baseline_symbols.txt` (если использовался нестандартный список);
- агрегированные метрики baseline (`*_stats.json`, `*_trades.csv`, `*_per_symbol_stats.csv`);
- результаты robustness-инструментов (`*_wfa.csv`, `*_cv.csv` или `*.json`, `*_bootstrap.json`, `*_latency.json`);
- forensic-артефакты (`*_forensic_*.csv`, `*_forensic_*.json`) по символам, exit reasons, хвостам PnL и stability check;
- если был модельный прогон - `range-v3_gating_eval.json` с покрытием и delta по метрикам.

## 7. Что нельзя забывать новому чату

- проект уже делает боевую модель, а не только scaffold;
- sandbox не является главной целью;
- legacy больше не основной объект развития;
- docs2 - новая система source of truth;
- любые дальнейшие roadmap и патчи должны опираться на docs2, а не на старые ощущения о проекте.



## === UPDATED MIGRATION STATE ===

Current Phase: Phase 2 - Diagnostic Range Isolation

Previous:
- Phase 0: Data Contract Stabilization (DONE)
- Phase 1: Core Revalidation (DONE)

Objective:
Validate range-core in isolated regime.

Execution Protocol:
1. baseline
2. mask gate
3. state machine
4. pooled comparison

Decision:
- viable → Phase 3
- not viable → redesign core
