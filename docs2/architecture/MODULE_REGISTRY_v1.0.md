# MODULE_REGISTRY_v1.0 - реестр модулей и зон ответственности

## 0. Правило чтения

Этот реестр - не код и не roadmap.
Он нужен, чтобы понимать:
- какие модули уже есть;
- какие planned;
- какие active;
- какие partial;
- какие legacy;
- какие обязанности модулю запрещены.

Статусы:
- `legacy`
- `active`
- `partial`
- `planned`
- `deprecated`

## 1. Формат записи

Для каждого модуля фиксируем:
- module_path
- layer
- status
- responsibility
- depends_on
- must_not_do
- notes

## 2. Legacy modules

### app2/range/range_v3_legacy.py
- layer: legacy
- status: legacy
- responsibility: reference implementation Range V3
- depends_on: config / old range logic
- must_not_do: не становиться местом новых архитектурных решений
- notes: используется для sanity-check и архивного baseline

### app2/range/v3_backtest_legacy.py
- layer: legacy
- status: legacy
- responsibility: legacy backtest orchestration
- depends_on: range_v3_legacy
- must_not_do: не должен быть драйвером нового core

## 3. Core modules

### app2/range/core/engine.py
- layer: core
- status: active
- responsibility: вход в core signal path, подготовка параметров, вызов state machine
- depends_on: state_machine
- must_not_do: не считать метрики, не выполнять risk logic внутри себя

### app2/range/core/state_machine.py
- layer: core
- status: active
- responsibility: генерация сигналов и debug-info
- depends_on: blocks, geometry
- must_not_do: не считать portfolio metrics, не заниматься сериализацией

### app2/range/core/blocks.py
- layer: core
- status: active
- responsibility: низкоуровневые строительные блоки логики signal path
- depends_on: core helpers
- must_not_do: не становиться вторым backtest

### app2/range/core/geometry.py
- layer: core / analytics
- status: partial
- responsibility: геометрия диапазонов, geo_* признаки, аналитический контекст
- depends_on: OHLCV + params
- must_not_do: не управлять execution напрямую
- notes: требует выравнивания параметрического контракта окна

### app2/range/core/risk.py
- layer: risk
- status: active
- responsibility: sizing, SL/TP, risk state, circuit breaker, weekend policy
- depends_on: config / RiskState
- must_not_do: не делать feature engineering, не считать dataset labels

### app2/range/core/metrics.py
- layer: metrics
- status: active
- responsibility: trade/symbol-level metrics
- depends_on: trades
- must_not_do: не влиять на решения торговли

### app2/range/core/portfolio.py
- layer: metrics
- status: active
- responsibility: portfolio aggregates
- depends_on: trades / metrics
- must_not_do: не влиять на execution

### app2/range/core/stats.py
- layer: metrics
- status: active
- responsibility: общие статистические helper-функции
- depends_on: numeric arrays / pnls
- must_not_do: не становиться дублёром metrics.py

### app2/range/core/backtest.py
- layer: orchestration
- status: active, overloaded
- responsibility: orchestrate data loading, core run, risk, metrics, serialization
- depends_on: engine, risk, metrics, portfolio, stats
- must_not_do: не должен становиться центром feature engineering и всей business logic
- notes: главный structural risk текущего состояния

### app2/range/core/sweep.py
- layer: research tooling
- status: active
- responsibility: sweeps и массовые прогоны core
- depends_on: core backtest
- must_not_do: не подменять собой orchestrator долгосрочно

## 4. Range / dataset / ML modules

### app2/range/dataset.py
- layer: dataset
- status: active
- responsibility: построение структур Dataset A/B
- depends_on: trades, snapshots, features
- must_not_do: не смешивать features и future-only labels

### app2/range/make_datasets.py
- layer: dataset orchestration
- status: active
- responsibility: CLI generation Dataset A/B
- depends_on: dataset.py, out artifacts
- must_not_do: не скрывать entry_mode и label-type metadata

### app2/range/baseline_ml.py
- layer: ml
- status: active
- responsibility: baseline ML experiments
- depends_on: datasets
- must_not_do: не использовать незафиксированные splits

### app2/range/catboost_train.py
- layer: ml
- status: active
- responsibility: CatBoost experiments / training
- depends_on: datasets
- must_not_do: не обходить research policy по splits и metadata

### app2/range/analysis.py
- layer: research diagnostics
- status: active
- responsibility: анализ range артефактов
- depends_on: out artifacts
- must_not_do: не дублировать core logic

### app2/range/backtest.py
- layer: older range
- status: partial / legacy-adjacent
- responsibility: older range backtest path
- depends_on: older range modules
- must_not_do: не конкурировать с core как основной target architecture

### app2/range/batch.py
- layer: orchestration / tooling
- status: active
- responsibility: batch execution
- depends_on: cli / backtests
- must_not_do: не хранить исследовательскую истину

## 5. Tooling and platform modules

### app2/tools/leakage_validator.py
- layer: tooling
- status: active
- responsibility: heuristic leakage guardrail
- depends_on: source scanning
- must_not_do: не считаться формальным доказательством отсутствия leakage

### app2/cli.py
- layer: orchestration surface
- status: active
- responsibility: единая командная поверхность
- depends_on: почти все верхнеуровневые команды
- must_not_do: не превращаться в место бизнес-логики

### agent_hands.py
- layer: external tooling bridge
- status: active, read-only
- responsibility: связь с локальной моделью для анализа
- depends_on: lm studio bridge
- must_not_do: не писать в проект, не обходить patch workflow

## 6. Planned modules

### app2/range/labels.py
- layer: dataset / labels
- status: planned
- responsibility: явное хранение label logic
- notes: нужен для разведения offline-only и online-compatible labels

### app2/range/inference.py
- layer: ml inference
- status: planned
- responsibility: production-like inference path для моделей
- notes: не должен быть размазан по backtest.py

### app2/range/model_registry.py
- layer: ml infra
- status: planned
- responsibility: registry моделей, метаданных, версий, thresholds

### app2/pipelines/run_experiment.py
- layer: orchestration
- status: planned
- responsibility: единый orchestrator research цепочки

### app2/pipelines/run_training.py
- layer: orchestration
- status: planned
- responsibility: training pipeline runner

### app2/pipelines/run_backtest.py
- layer: orchestration
- status: planned
- responsibility: backtest pipeline runner

### app2/pipelines/run_sandbox.py
- layer: orchestration
- status: planned
- responsibility: sandbox pipeline runner

## 7. Правило актуализации

Любой новый модуль:
1. получает статус;
2. получает responsibility;
3. получает список `must_not_do`;
4. добавляется сюда до или одновременно с кодовым патчем.
