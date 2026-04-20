# RESEARCH_POLICY_v1.0 - политика исследовательской честности

## 0. Назначение

Этот документ отвечает на вопрос:
что мы считаем исследовательски честной постановкой для данных, моделей, labels и backtest.

Он переносит и закрепляет:
- leakage policy из старых CONTRACT;
- новые договорённости по Dataset A;
- разграничение offline-only и online-compatible сущностей.

## 1. Главные принципы

1. future data в rolling path запрещены;
2. offline-only labels разрешены только при явной маркировке;
3. candidate universe и executed trades не смешиваются без явной цели;
4. time split обязателен;
5. symbol split желателен;
6. experiment metadata обязательна;
7. красивые офлайн-метрики не считаются ценностью сами по себе;
8. новый ML допускается только после доказательства edge хотя бы в узком rule-based сегменте.

## 2. Dataset A truth policy

### 2.1. Dataset A_research
- source: `candidates`
- цель: поиск edge и обучение входной модели
- это исследовательская правда для входного ML

Используем, когда хотим ответить:
- какие входы были доступны рынком;
- чем хорошие кандидаты отличаются от плохих;
- может ли ML дать настоящий edge, а не только улучшить already-executed policy.

### 2.2. Dataset A_policy
- source: `trades`
- цель: анализ поведения текущего rule/core policy

Используем, когда хотим ответить:
- как уже работающий policy выбирает сделки;
- какие executed trades оказались хорошими / плохими;
- где policy системно ошибается.

### 2.3. Правило по умолчанию
В docs2 принимается:
- `A_research` и `A_policy` существуют параллельно;
- они не заменяют друг друга;
- любой эксперимент обязан явно указывать, с каким типом Dataset A он работает.

## 3. Dataset B policy

Dataset B:
- строится по барам внутри сделки;
- используется для in-trade анализа и exit research;
- может содержать offline-only labels, но они должны быть явно промаркированы.

### 3.1. `y_exit`
- `y_exit` v1 считается offline-only;
- он полезен для research;
- он не может использоваться как live-совместимый signal без отдельной новой постановки.

## 4. Leakage policy

### 4.1. Что считается допустимым
- offline labelers могут использовать будущее;
- offline-only labels могут участвовать в исследовании;
- snapshots и dataset joins допустимы, если логика их доступности на t ясна и задокументирована.

### 4.2. Что считается нарушением
- `shift(-k)` в rolling decision path;
- future labels внутри feature columns;
- неявные joins, которые делают будущее доступным на t;
- случайный split как основной способ оценки research модели;
- отсутствие логирования `entry_mode`, profile, filters, universe.

### 4.3. Leakage validator
- leakage validator обязателен как guardrail;
- он не считается формальным доказательством;
- спорные случаи решаются через ручной data review.

## 5. Split policy

### 5.1. Обязательный минимум
- time split обязателен
- symbol split желателен
- metadata split strategy записывается в артефакты

### 5.2. Что не считаем основной оценкой
- random shuffle split без времени и без symbol awareness

## 6. Experiment metadata

Каждый experiment / dataset artifact должен фиксировать минимум:
- dataset type (`A_research`, `A_policy`, `B`)
- entry_mode
- interval
- symbol universe
- config id / risk profile
- active filters
- split strategy
- label type (`online-compatible`, `offline-only`)

Дополнительно meta‑файл (манифест) обязан содержать:
 - `dataset_kind` — `entry` для набора входов (Dataset A) или `intrade` для набора внутри сделок (Dataset B);
 - `truth_policy` — `candidates` (для `A_research`) или `trades` (для `A_policy`);
 - `config_path` — путь к использованному конфигурационному файлу `range`.
Эти поля делают источники данных прозрачными и позволяют однозначно определить, на каком truth‑policy и конфигурации обучалась модель.

## 6.1. Rule-filter before ML

До обучения новой модели по текущему policy-контуру обязательно выполняется правило:
- сначала ищется устойчивый rule-based сегмент через forensic-разрезы;
- затем этот сегмент проверяется как тупой фильтр без ML;
- только если rule-filter удерживает положительный профиль, допускается новый ML-селектор.

Если устойчивый сегмент не найден, новая модель не обучается, а ветка стратегии считается кандидатом на заморозку.

## 7. Что значит "исследовательски честно"

Постановка считается исследовательски честной, если одновременно выполнено:
- feature path не нарушает time causality;
- dataset type и truth policy зафиксированы;
- split policy адекватна;
- labels промаркированы;
- leakage validator пройден;
- experiment metadata сохранена;
- uplift оценивается не только через AUC/F1, но и через impact на торговую систему;
- новый ML не строится поверх отрицательного baseline без forensic-доказательства edge.

## 8. Источники без потери смысла

Исторический контекст сохранён в:
- `docs2/history/contract/CONTRACTv4.8_migrated.md`
- `docs2/history/contract/CONTRACTv4.9_migrated.md`
- `docs2/history/contract/CONTRACTv4.10_migrated.md`
- `docs2/history/model_plan/model_plan_v0.3_migrated.md`
- `docs2/history/model_plan/model_plan_v0.4_migrated.md`
