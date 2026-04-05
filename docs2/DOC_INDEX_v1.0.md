# DOC_INDEX_v1.0 - индекс документации docs2

## 1. Обязательный порядок чтения

1. `docs2/READMEv1.0.md`
2. `docs2/DOC_INDEX_v1.0.md`
3. `docs2/policies/PROJECT_RULES_v1.0.md`
4. `docs2/operations/MIGRATION_CONTEXT_v1.0.md`
5. `docs2/CONTRACTv1.0.md`
6. `docs2/model_plan_v1.0.md`
7. `docs2/policies/LLM_WORKFLOW_v1.0.md`
8. `docs2/architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
9. `docs2/architecture/MODULE_REGISTRY_v1.0.md`

## 2. Базовые документы

- `READMEv1.0.md` - входная рамка, правила чтения, текущее состояние проекта
- `CONTRACTv1.0.md` - архитектурный и data-контракт проекта
- `model_plan_v1.0.md` - сводный план развития модели и системы
- `policies/PROJECT_RULES_v1.0.md` - полный свод правил проекта

## 3. Политики

- `policies/PROJECT_RULES_v1.0.md`
- `policies/RESEARCH_POLICY_v1.0.md`
- `policies/LLM_WORKFLOW_v1.0.md`

## 4. Операционные документы

- `operations/MIGRATION_CONTEXT_v1.0.md`
- `operations/NEW_CHAT_BOOTSTRAP_v1.0.md`
- `operations/CHANGELOG_v1.0.md`

## 5. Архитектура

- `architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
- `architecture/MODULE_REGISTRY_v1.0.md`

## 6. Шаблоны

- `templates/PHASE_REPORT_TEMPLATE_v1.0.md`

## 7. История

В `docs2/history/` лежат только legacy-копии из исходного `docs`:
- `history/readme/*_migrated.md`
- `history/contract/*_migrated.md`
- `history/model_plan/*_migrated.md`

## 8. Куда что писать

- правила и дисциплина работы - в `policies/PROJECT_RULES_v1.0.md`
- роли LLM и agent mode - в `policies/LLM_WORKFLOW_v1.0.md`
- исследовательские ограничения - в `policies/RESEARCH_POLICY_v1.0.md`
- актуальная стадия и риски - в `operations/MIGRATION_CONTEXT_v1.0.md`
- архитектурные инварианты - в `CONTRACTv1.0.md`
- roadmap и фазы - в `model_plan_v1.0.md`

## 9. Ключевое правило

Сначала читаются `docs2`, потом анализируется `app2`. Приоритет у фактического состояния кода, но трактуется оно через текущий source of truth из `docs2`.
