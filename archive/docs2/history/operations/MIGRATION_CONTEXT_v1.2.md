# MIGRATION_CONTEXT_v1.2 - контекст для переезда в новый чат

## 1. Новый обязательный шаг
Перед любым анализом новый чат обязан прочитать полный свод правил:
- `docs2/policies/PROJECT_RULES_v1.0.md`

Только после этого он читает migration context, README, contract и plan.

## 2. Текущая рамка

- core-first development
- legacy = reference only
- sandbox = промежуточный этап к online
- финальная цель = исследовательски честная offline/online система
- docs2 = новый source of truth

## 3. Старые правила не считаются утерянными
Весь старый набор правил из ранних `docs/READMEv*` теперь перенесён в `PROJECT_RULES_v1.0.md`.

## 4. Что новый чат должен подтвердить
- что прочитал полный свод правил
- что принял правила версионирования и дополнения документации
- что понимает текущую стадию проекта
- что не будет пропускать migration reading order

## 5. Active code zones
- `app2/cli.py`
- `app2/range/config.json`
- `app2/range/core/`
- `app2/range/make_datasets.py`
- `app2/range/dataset.py`
- `app2/range/baseline_ml.py`
- `app2/range/catboost_train.py`
- `app2/tools/leakage_validator.py`
- `agent_hands.py`

## 6. Главные активные риски
- перегруженный `core/backtest.py`
- ambiguity Dataset A truth
- risk config source-of-truth drift
- leakage beyond validator
- потеря контекста при миграции, если не читать rules first
