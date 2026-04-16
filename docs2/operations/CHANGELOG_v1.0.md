# CHANGELOG_v1.0 - журнал изменений docs2

## v1.0

Создана новая система документации `docs2/`.

### Введены новые документы
- `READMEv1.0.md`
- `DOC_INDEX_v1.0.md`
- `CONTRACTv1.0.md`
- `model_plan_v1.0.md`
- `architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
- `architecture/MODULE_REGISTRY_v1.0.md`
- `policies/RESEARCH_POLICY_v1.0.md`
- `policies/LLM_WORKFLOW_v1.0.md`
- `operations/MIGRATION_CONTEXT_v1.0.md`
- `operations/CHANGELOG_v1.0.md`
- `templates/PHASE_REPORT_TEMPLATE_v1.0.md`

### В history перенесены verbatim-копии старых документов
README:
- v4.7
- v4.8
- v4.9
- v4.10

CONTRACT:
- v4.8
- v4.9
- v4.10

model_plan:
- v0.3
- v0.4

### Главные новые принципы
- docs2 становится новой системой source of truth
- core официально признаётся главным объектом развития
- Dataset A делится на `A_research` и `A_policy`
- migration context выносится в отдельный обязательный документ
- architecture blueprint и module registry становятся обязательными частями процесса

## Updates (2026-04-16)

### Улучшения baseline и исследовательского контура

- Введены утилиты для фиксации и анализа baseline (`baseline_core.py`), сравнения baseline с AI‑фильтром (`evaluate_gating.py`), а также для генерации сводной таблицы метрик (`summary_core.py`).
- Добавлен файл `baseline_symbols.txt` для динамического задания списка тикеров; его можно редактировать без изменения кода.
- Обновлена политика исследовательской честности: в meta‑файле (манифесте) теперь обязательно указываются `dataset_kind`, `truth_policy`, `config_path`, `label_mode` и `horizon`.
- Обновлён миграционный контекст: baseline должен демонстрировать положительное ожидание, иначе оценка AI‑фильтра не имеет смысла; результаты baseline и фильтра сохраняются в `out/` и включаются в обязательный пакет артефактов.
- В CLI добавлены команды `range-core-baseline`, `range-gating-eval` и `range-core-summary`; команда `range-v3-backtest` теперь по умолчанию использует движок `core`.

## Правило ведения changelog

Фиксируем:
- создание / удаление документов;
- перенос документов между структурами;
- существенное изменение ролей документов.

Не фиксируем подробно:
- мелкие формулировочные правки;
- косметические изменения markdown.
