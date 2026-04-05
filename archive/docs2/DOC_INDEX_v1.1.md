# DOC_INDEX_v1.1 - индекс документации docs2

## 1. Что читать первым при миграции

Обязательный порядок:
1. `docs2/READMEv1.1.md`
2. `docs2/DOC_INDEX_v1.1.md`
3. `docs2/operations/MIGRATION_CONTEXT_v1.1.md`
4. `docs2/CONTRACTv1.1.md`
5. `docs2/model_plan_v1.1.md`
6. `docs2/policies/LLM_WORKFLOW_v1.1.md`

## 2. Базовые документы

- `READMEv1.1.md` - текущая рамка + сохранённая история README
- `CONTRACTv1.1.md` - актуальная рамка + сохранённая история контрактов
- `model_plan_v1.1.md` - актуальная рамка + сохранённая история plan

## 3. Архитектура и операции

- `architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
- `architecture/MODULE_REGISTRY_v1.0.md`
- `operations/MIGRATION_CONTEXT_v1.1.md`
- `operations/NEW_CHAT_BOOTSTRAP_v1.1.md`
- `operations/CHANGELOG_v1.1.md`

## 4. Политики

- `policies/RESEARCH_POLICY_v1.0.md`
- `policies/LLM_WORKFLOW_v1.1.md`

## 5. Правило source of truth

Новый чат не должен читать старые `docs/` раньше, чем он прочитал актуальные `docs2` документы.
Старые `docs/` считаются историей, а `docs2` - новым входным контуром.
