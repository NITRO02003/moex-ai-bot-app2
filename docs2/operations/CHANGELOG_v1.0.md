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

## Updates (2026-04-17)

### Приоритизация backlog и правила документации

- Зафиксирован новый порядок работ: сначала критические проблемы доверия к результатам, потом техдолг, и только затем полноценные тесты.
- В `model_plan_v1.0.md` добавлен приоритетный backlog P0-P4, включая deferred-доработки по docs и инфраструктуре.
- В `CONTRACTv1.0.md` уточнён source-of-truth для risk-config: профильные risk-параметры считаются canonical в `risk_profiles`, а дублирование с `params` считается ошибкой.
- В `PROJECT_RULES_v1.0.md` и связанных документах закреплено правило: новая версия имени файла нужна только при смене source of truth или структуры документа; минорные дополнения допускаются без смены имени, но должны попадать в changelog.
- Уточнены риски локальных LLM-агентов и bridge-слоёв без привязки к `agent_hands` как обязательному компоненту.

## Updates (2026-04-17, part 2)

### Реприоритизация roadmap после внешних рекомендаций

- В `model_plan_v1.0.md` зафиксировано, что sandbox больше не является ближайшим следующим этапом после P0/P1.
- Добавлен отдельный блок устойчивости оценки: WFA, time-series split, bootstrap / Monte Carlo по сделкам, calibration вероятностей и latency sensitivity.
- `Regime Engine v1` поднят в ранний следующий этап после стабилизации core и честного ML-контура.
- Ensembles / stacking, sentiment, dashboard и monitoring перенесены на следующий этап и не входят в ближайший приоритетный блок.
- `MIGRATION_CONTEXT_v1.0.md` обновлён под новый порядок работ: сначала trust fixes, затем robustness/regime block, и только потом sandbox / online bridge.

## Updates (2026-04-17, part 3)

### Смена вектора разработки на forensic-first

- В `model_plan_v1.0.md` зафиксирован новый текущий приоритет: сначала forensic-отчёт и приговор ядру Range V3, а не новые модели, calibration или sandbox.
- Добавлен критерий живого сегмента: `pf > 1.05`, лучше `> 1.1`, с достаточным числом сделок и обязательной временной проверкой по `early` / `mid` / `late`.
- В `RESEARCH_POLICY_v1.0.md` закреплено правило `rule-filter before ML`: сначала тупой фильтр на найденном сегменте, и только потом допускается ML-селектор.
- В `MIGRATION_CONTEXT_v1.0.md` обновлён порядок работ: forensic-разрезы, stability check, затем развилка `rule-filter` / `exit redesign` / `freeze Range V3`.
- Зафиксирована пауза на новые модели, Regime Engine, calibration, sandbox и online-движение до завершения forensic-цикла.



### Regime Diagnostic Shift

- baseline failure confirmed
- data layer fixed
- issue localized to regime/gating
- Phase 2 started
