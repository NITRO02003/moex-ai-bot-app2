# ARCHITECTURE_BLUEPRINT_v1.0 - целевая архитектура системы

## 0. Назначение

Этот документ описывает:
- текущую фактическую архитектуру `app2`;
- целевую архитектуру поздних фаз;
- взаимодействие текущих и будущих модулей;
- зоны ответственности;
- запрещённые зависимости.

Цель документа:
- удержать core от повторного превращения в монолит;
- заранее описать финальные слои и модули до поздних фаз;
- упростить миграцию чата и работу внешних reviewers.

## 1. Базовый принцип

Проект уже не "строит каркас".
Проект уже строит боевую модель.

Отсюда следуют правила:
- legacy - reference;
- core - главный контур развития;
- dataset/ML - часть боевой архитектуры, а не отдельная "поздняя мечта";
- online layer и sandbox path проектируются заранее, но реализуются по приоритету.

## 2. Текущее состояние по слоям

### 2.1. Legacy lane
Текущие модули:
- `app2/range/range_v3_legacy.py`
- `app2/range/v3_backtest_legacy.py`

Назначение:
- reference;
- sanity-check;
- регресс-контроль;
- архивный baseline.

### 2.2. Core lane
Текущие модули:
- `app2/range/core/__init__.py`
- `app2/range/core/backtest.py`
- `app2/range/core/blocks.py`
- `app2/range/core/engine.py`
- `app2/range/core/geometry.py`
- `app2/range/core/metrics.py`
- `app2/range/core/portfolio.py`
- `app2/range/core/risk.py`
- `app2/range/core/state_machine.py`
- `app2/range/core/stats.py`
- `app2/range/core/sweep.py`

Назначение:
- signal path;
- risk and execution logic;
- metrics and portfolio stats;
- sweep / diagnostics;
- основа для sandbox и online evolution.

### 2.3. Dataset / ML lane
Текущие модули:
- `app2/range/dataset.py`
- `app2/range/make_datasets.py`
- `app2/range/baseline_ml.py`
- `app2/range/catboost_train.py`

Назначение:
- построение Dataset A/B;
- baseline ML;
- CatBoost;
- дальнейший inference и gating.

### 2.4. Tooling lane
Текущие модули и поверхности:
- `app2/cli.py`
- `app2/tools/leakage_validator.py`
- `agent_hands.py` в корне проекта
- Qwen как внешний локальный reviewer

## 3. Целевая система по слоям

### 3.1. Data Layer
Ответственность:
- загрузка данных;
- нормализация OHLCV;
- alignment;
- data validation;
- контроль корректности источников.

Ожидаемые / будущие модули:
- loaders
- normalizers
- alignment
- validators

### 3.2. Research Feature Layer
Ответственность:
- feature builders;
- geometry features;
- regime/context features;
- подготовка входов для datasets и research analysis.

### 3.3. Strategy Core Layer
Ответственность:
- state machine;
- blocks;
- signal generation;
- regime gating;
- entry / exit policy;
- candidate universe.

Ожидаемые модули:
- `engine.py`
- `state_machine.py`
- `blocks.py`
- `signals.py`
- `regime.py`

### 3.4. Geometry Layer
Ответственность:
- вычисление L/U/H/M;
- class / score диапазонов;
- аналитика структуры рынка;
- контекст для datasets и diagnostics.

Принцип:
- geometry не управляет execution напрямую;
- AAA/AA/A не должны превращаться в жёсткий боевой рубильник без отдельного решения.

### 3.5. Risk and Execution Layer
Ответственность:
- sizing;
- SL/TP;
- дневные лимиты;
- cooldown / circuit breaker;
- sandbox/live adapters.

Ожидаемые модули:
- `risk.py`
- `execution.py`
- `execution_sandbox.py`
- `execution_live.py`

### 3.6. Metrics and Reporting Layer
Ответственность:
- trade metrics;
- symbol stats;
- portfolio stats;
- summary reports;
- sweep aggregation.

Текущая база:
- `metrics.py`
- `portfolio.py`
- `stats.py`

### 3.7. Dataset and ML Layer
Ответственность:
- Dataset A_research;
- Dataset A_policy;
- Dataset B;
- labels;
- baseline ML;
- CatBoost;
- future model registry and inference layer.

Ожидаемые модули:
- `dataset.py`
- `make_datasets.py`
- `labels.py`
- `baseline_ml.py`
- `catboost_train.py`
- `inference.py`
- `model_registry.py`

### 3.8. Orchestration Layer
Ответственность:
- experiment runner;
- training runner;
- backtest runner;
- sandbox runner;
- repeatable pipelines.

Будущие модули:
- `app2/pipelines/run_experiment.py`
- `app2/pipelines/run_training.py`
- `app2/pipelines/run_backtest.py`
- `app2/pipelines/run_sandbox.py`

### 3.9. Online Layer
Ответственность:
- broker adapters;
- online inference;
- online risk supervision;
- monitoring;
- recovery / rollback;
- paper / sandbox / live transitions.

Этот слой заранее описывается в blueprint, даже если реализован позже.

## 4. Разрешённые зависимости

Разрешено:
- core -> risk
- core -> metrics
- dataset -> feature layer
- orchestration -> верхнеуровневые pipeline modules
- reporting -> metrics / portfolio / stats
- inference -> model registry

## 5. Запрещённые зависимости

Запрещено:
- `backtest.py` не должен реализовывать business logic feature layer;
- `risk.py` не должен знать про offline labels;
- `geometry.py` не должен управлять execution напрямую;
- `dataset.py` не должен тянуть future-only labels в features;
- online execution не должен зависеть от legacy;
- agent_hands не должен иметь write-role в проекте;
- внешний reviewer не должен менять source-of-truth документы без явного решения команды.

## 6. Архитектурные риски текущего состояния

### 6.1. Перегруженный `core/backtest.py`
Это главный structural risk:
- orchestration;
- features;
- AI gating;
- execution;
- serialization;
- portfolio glue.

Если этот узел не разгрузить, весь modular core начнёт снова сходиться в монолит.

### 6.2. Несколько источников торговой логики
Сейчас логика размазана между:
- legacy;
- `range_v3.py`;
- `core/state_machine.py`;
- `core/blocks.py`;
- частично `core/geometry.py`.

Это допустимо только как переходное состояние.

### 6.3. Risk config drift
Если смысл risk-параметров дублируется между `params` и `risk_profiles`,
это создаёт опасность дрейфа результатов.

### 6.4. Dataset truth ambiguity
Если не развести `A_research` и `A_policy`, ML-ветка может уйти в неправильную постановку.

## 7. Картина к поздним фазам

К Phase 5 система должна обладать следующими слоями и модулями:

1. Data / alignment
2. Core / signals / regime / geometry
3. Risk / execution
4. Metrics / reporting
5. Dataset / labels / ML / inference / registry
6. Orchestration
7. Online / sandbox / live bridge
8. Tooling / leakage / agent governance

Подробный список по каждому модулю и его статусу ведётся в `MODULE_REGISTRY_v1.0.md`.

## 8. Источники без потери смысла

Старые документы, из которых перенесён смысл:
- `docs2/history/readme/READMEv4.9_migrated.md`
- `docs2/history/readme/READMEv4.10_migrated.md`
- `docs2/history/contract/CONTRACTv4.9_migrated.md`
- `docs2/history/contract/CONTRACTv4.10_migrated.md`
- `docs2/history/model_plan/model_plan_v0.3_migrated.md`
- `docs2/history/model_plan/model_plan_v0.4_migrated.md`
