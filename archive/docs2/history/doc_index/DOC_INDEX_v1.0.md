# DOC_INDEX_v1.0 - индекс документации docs2

## 1. Назначение

Этот индекс определяет:
- какие документы есть в `docs2/`;
- какую роль выполняет каждый документ;
- в каком порядке читать документацию;
- куда добавлять новую информацию.

## 2. Порядок чтения

Для быстрого входа в проект:

1. `docs2/READMEv1.0.md`
2. `docs2/DOC_INDEX_v1.0.md`
3. `docs2/architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
4. `docs2/CONTRACTv1.0.md`
5. `docs2/model_plan_v1.0.md`
6. `docs2/policies/RESEARCH_POLICY_v1.0.md`
7. `docs2/policies/LLM_WORKFLOW_v1.0.md`
8. `docs2/operations/MIGRATION_CONTEXT_v1.0.md`

## 3. Состав docs2

### 3.1. Входные документы
- `READMEv1.0.md` - где проект находится сейчас и что считать главным
- `DOC_INDEX_v1.0.md` - этот файл

### 3.2. Архитектура
- `architecture/ARCHITECTURE_BLUEPRINT_v1.0.md` - целевая архитектура системы от текущего среза до поздних фаз
- `architecture/MODULE_REGISTRY_v1.0.md` - реестр модулей и зон ответственности

### 3.3. Контракт и план
- `CONTRACTv1.0.md` - архитектурный и data-контракт
- `model_plan_v1.0.md` - план развития по фазам

### 3.4. Политики
- `policies/RESEARCH_POLICY_v1.0.md` - исследовательская честность, Dataset A/B truth, leakage policy
- `policies/LLM_WORKFLOW_v1.0.md` - роли ChatGPT, Qwen, agent_hands и порядок внесения изменений

### 3.5. Операционка
- `operations/MIGRATION_CONTEXT_v1.0.md` - что нужно для миграции чата
- `operations/CHANGELOG_v1.0.md` - журнал изменений docs2

### 3.6. Шаблоны
- `templates/PHASE_REPORT_TEMPLATE_v1.0.md` - шаблон отчёта по фазе / итерации

### 3.7. История
- `history/readme/*`
- `history/contract/*`
- `history/model_plan/*`

## 4. Куда что писать

### В README
Пишем:
- текущий статус
- входные правила
- краткий контекст

Не пишем:
- модульный реестр
- детальный контракт
- длинные исторические списки

### В CONTRACT
Пишем:
- инварианты
- boundaries
- data contracts
- то, что нельзя ломать

### В model_plan
Пишем:
- фазы
- очередность
- стратегические шаги

### В ARCHITECTURE_BLUEPRINT
Пишем:
- целевую карту системы
- все ожидаемые модули и слои
- взаимодействие модулей

### В MODULE_REGISTRY
Пишем:
- текущий список модулей
- статус модулей
- responsibility
- must_not_do

### В RESEARCH_POLICY
Пишем:
- исследовательскую правду
- truth policy для Dataset A/B
- leakage policy
- split policy
- metadata requirements

### В MIGRATION_CONTEXT
Пишем:
- текущую фазу
- текущие риски
- активные зоны кода
- обязательный пакет файлов для нового чата

## 5. Правило добавления нового документа

Любой новый документ:
1. должен иметь владельца и цель;
2. должен быть добавлен в этот индекс;
3. не должен дублировать уже существующий смысл;
4. должен явно указывать, является ли он source of truth или вспомогательным.
