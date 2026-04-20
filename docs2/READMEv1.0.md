# READMEv1.0 - MOEX APP2+ / docs2 source of truth

Файл: `docs2/READMEv1.0.md`

Это единый рабочий README для `docs2`.
Он собран из содержимого версий `READMEv1.0`-`READMEv1.3`.
В корне `docs2` держится одна рабочая версия файла. Мелкие правки без изменения source of truth или стадии проекта делаются без bump версии.

## 0. Что читать первым

При входе в проект сначала читать:
1. `docs2/READMEv1.0.md`
2. `docs2/DOC_INDEX_v1.0.md`
3. `docs2/policies/PROJECT_RULES_v1.0.md`
4. `docs2/operations/MIGRATION_CONTEXT_v1.0.md`
5. `docs2/CONTRACTv1.0.md`
6. `docs2/model_plan_v1.0.md`
7. `docs2/policies/LLM_WORKFLOW_v1.0.md`
8. `docs2/architecture/ARCHITECTURE_BLUEPRINT_v1.0.md`
9. `docs2/architecture/MODULE_REGISTRY_v1.0.md`

## 1. Текущая рамка проекта

На текущем срезе:
- `core` - главный и единственный рабочий контур развития;
- `legacy` - только архивный reference, без отдельного roadmap и без обязательных сравнений;
- песочница - промежуточный этап к online, а не финальная цель;
- финальная цель проекта - исследовательски честная offline/online торговая система;
- `docs2` - source of truth для правил, миграции, контракта и плана;
- Dataset A truth policy фиксируется как:
  - `Dataset A_research = candidates`
  - `Dataset A_policy = trades`.

## 2. Что обязан знать новый чат

- проект уже не строит каркас, а делает боевую модель;
- `app2` - главный активный кодовый контур;
- сначала читаются правила и migration context, потом анализируется `app2`;
- выводы делаются по фактическому состоянию `app2`, а не по старым промежуточным документам;
- если файла нет в архиве или есть сомнение в его актуальности - правки стоп.

## 3. Правила ведения документации

- в рабочем дереве `docs2` держится один актуальный файл на семейство;
- `history/` используется только для legacy-копий из исходного `docs`, а не для складирования каждой промежуточной версии `docs2`;
- мелкие правки в текущем файле не требуют новой версии имени;
- новая версия имени файла нужна только если реально меняется source of truth или стадия проекта;
- новое содержание должно дополнять уже согласованное, а не обнулять его.

## 4. Что остаётся обязательным

- патчи - только архивом и только изменённые файлы;
- все результаты и логи - только через корневую `out/`;
- перед отправкой патча обязателен self-check;
- спорные архитектурные упрощения не делаются молча;
- ответы по умолчанию даются в режиме критика: жёстко, коротко, по существу.

## 5. Что лежит в `docs2/history`

В `docs2/history/` лежат только legacy-копии документов из исходного `docs`:
- `history/readme/READMEv4.7_migrated.md`
- `history/readme/READMEv4.8_migrated.md`
- `history/readme/READMEv4.9_migrated.md`
- `history/readme/READMEv4.10_migrated.md`
- `history/contract/CONTRACTv4.8_migrated.md`
- `history/contract/CONTRACTv4.9_migrated.md`
- `history/contract/CONTRACTv4.10_migrated.md`
- `history/model_plan/model_plan_v0.3_migrated.md`
- `history/model_plan/model_plan_v0.4_migrated.md`

## 6. Сохранённая история README из исходного `docs`

Ниже сохранено verbatim-содержимое legacy README из прежнего `docs`, чтобы не потерять контекст старых этапов. Эти приложения - исторический слой, а не активный source of truth.

## Appendix A - preserved `docs/READMEv4.9.md`

# READMEv4.9 — MOEX APP2+ / Range V3 → core_v4 (Phase 0.2, офлайн-ядро)

Файл: `docs/READMEv4.9.md`  
Предыдущая версия: `docs/READMEv4.8.md` (фиксировала rolling Range V3 + риск-ядро и убыточный baseline).

Этот README фиксирует статус проекта на момент старта **Phase 0.2 — Range core_v4 (offline)** и задаёт рамки,
в которых мы продолжаем работу над стратегией.

---

## 0. Текущий статус (конец 2025-12-21)

Мы по-прежнему в зоне **Фазы 0 (Range V3 Baseline)**, но теперь она раскладывается на два слоя:

1. **Legacy Range V3 (замороженное ядро):**
   - реализована rolling-логика для 30m SBER/GAZP/LKOH/GMKN/ROSN;
   - есть риск-ядро с параметрами:
     `risk_pct_per_trade`, `sl_pct`, `tp_pct`, `max_bars_in_trade`,
     `max_consecutive_losses`, `daily_dd_limit_pct`;
   - подключён robust ATR-фильтр (`atr_window`, `atr_pct_min/max`, `atr_min_valid_frac`, `atr_min_valid_bars`);
   - slope-фильтр пока используется **только в диагностике** (`mask_slope_frac`), т.к. текущий порог делает рынок «слишком плоским»;
   - по последним прогонам PF портфеля ~0.5–0.6, WinRate ~35–40% — baseline **убыточный, но честный**.

2. **Переход к core_v4 (offline-only на этом этапе):**
   - цель Phase 0.2 — выделить аккуратное **core-ядро Range** с понятной структурой (geometry / state machine / risk),
     не ломая legacy;
   - тяжёлые вещи (event-driven, microstructure, multi-TF, AAA/AA/A как боевой фильтр, продвинутый leakage-контроль)
     **не реализуем в app2 сейчас**, а оставляем в плане как задачи для будущей online-модели (app3).

Дополнительно:

- в процессе используется внешний локальный LLM **Qwen** как инструмент багфикса и диагностики;
- все решения и контракты фиксируются в `docs/`, Qwen не меняет модель напрямую;
- потенциальный **агентский режим ChatGPT** описан в CONTRACT и включается только после выхода core_v4 в плюс на offline.

---

## 1. Структура репозитория (напоминание)

В корне репо:

- `data/` — сырые реальные бары MOEX без synthetic empty slots;
- `processed/` — агрегированные рабочие бары, сохраняющие контракт реального бара; пустые интервалы и `NaN`-бары запрещены;
- `out/` — все результаты, отчёты, свипы, forward-тесты;
- `app2/` — текущее боевое ядро (Range V3, rule-based стратегии и т.д.);
- `app3/` — перспективная модульная архитектура и будущая online-модель;
- `docs/` — документация (READMEv*, CONTRACTv*, model_plan_v*).

Внутри `app2/range/`:

- `range_v3_legacy.py` — зафиксированное legacy-ядро Range V3 (rolling + риск-ядро);
- `v3_backtest_legacy.py` — старый бэктестер для legacy;
- `range_v3.py` / `core/` — будущая зона для core_v4 (по мере реализации Phase 0.2).

Никаких `data/` и `out/` внутри `app2/` и `app3` — весь вывод идёт через корневой `out/`.

---

## 2. Два режима Range: legacy и core

В `app2.cli` планируется единый интерфейс для Range V3:

```bash
python -m app2.cli range-v3-backtest \
  --symbols SBER GAZP LKOH GMKN ROSN \
  --interval 30min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --engine legacy \  # или core (когда будет готов)
  --out-prefix out/range_v3/ALL_30min_... \
  --tag ...
```

На момент v4.9:

- режим **`legacy`** — основной рабочий; `core` может быть заглушкой или экспериментальным;
- любые изменения в логике Range вносятся **либо** в legacy (минимальный bugfix), **либо** в core,
  но не в оба сразу;
- результаты legacy-служат эталоном для сравнения с core (PF/WinRate/MaxDD на одинаковом наборе данных).

---

## 3. Роль LLM-инструментов (ChatGPT, Qwen, агентский режим)

1. **ChatGPT (текущий ассистент):**
   - работает по строгим правилам:
     - не меняет контракт задним числом,
     - любые изменения фиксируются через новые версии `READMEv*`, `CONTRACTv*`, `model_plan_v*`;
   - кодовые изменения — только в виде патч-архивов с изменёнными файлами и понятным дифф-ТЗ.

2. **Qwen (локальный помощник):**
   - используется для:
     - локального багфикса,
     - быстрых экспериментов,
     - проверки гипотез по коду и данным;
   - результаты работы Qwen считаются «черновиками», пока не зафиксированы
     в нашем цикле `ТЗ → патч → бэктест → обновление docs/`.

3. **Агентский режим ChatGPT:**
   - рассматривается как отдельный инструмент для будущих фаз (после выхода core_v4 в плюс на offline);
   - допустимые задачи:
     - диагностика пайплайнов,
     - генерация отчётов / сводок по эксперимента...,
     - безопасный рефакторинг в рамках уже существующих модулей;
   - запрещено:
     - менять risk core без явного ТЗ и ревью,
     - править CONTRАCT,
     - писать/читать вне разрешённых директорий (`app2/`, `app3/`, `data/`, `processed/`, `out/`, `docs/`).

Детальное ТЗ по агентскому режиму — в `docs/CONTRACTv4.9.md` и `docs/model_plan_v0.4.md`.

---

## 4. Что брать с собой при переезде в новый чат

Для нового чата минимальный пакет контекста:

- `docs/READMEv4.9.md` — этот файл;
- `docs/CONTRACTv4.9.md` — актуальные контракты;
- `docs/model_plan_v0.4.md` — актуальный план;
- `app2/range/` (legacy + core);
- `app2/cli.py` (интерфейс);
- `app2/range/config.json`;
- небольшой срез `out/` с последними прогоном Range (legacy и/или core);
- небольшие примеры `data/` / `processed/` (30m, SBER + несколько тикеров).

С таким пакетом можно восстановить контекст и продолжить разработку без пролистывания старых чатов.


---

## Appendix B - preserved `docs/READMEv4.10.md`

# READMEv4.10 — MOEX APP2+ / Range V3 → core_v4 (Phase 1 start)

Файл: `docs/READMEv4.10.md`  
Предыдущая версия: `docs/READMEv4.9.md` (старт Phase 0.2, core_v4 scaffold).

Этот README фиксирует завершение Фазы 0 и старт Фазы 1 (датасеты).

---

## TL;DR (summary текущего чата)

- Базовое окружение проекта: `C:\Users\nitro\moex-ai-bot\.venv1` (использовать для всех запусков).
- `out/` пересоздан после прогона Phase 1.1, `app2` восстановлен (архивная копия: `archive/app2_2026-01-19/`).
- EMA‑полосы с `ma_len=20` пересчитаны по валидным сегментам.
- Добавлены return‑фичи: `ret_1/3/6`, `ret_mean_20`, `ret_vol_20`.
- ML прогоны переведены на per‑symbol split (тест по каждому тикеру).
- Multiprocessing по умолчанию включён в ключевых модулях (через `--n-jobs` или `APP2_N_JOBS`).
- Зафиксировали `APP2_N_JOBS=8` для этой машины (16 CPU), CatBoost `thread-count=8`.
- CatBoost GPU виден: `get_gpu_device_count() = 1` (RTX5080).
- COMPACT_FEATURES (сигнал 10min) закреплены: `v3_signal`, `band_width_pct`, `range_vs_atr`, `atr_14_pct`, `dist_from_ma`, `band_pos`, `edge_proximity`, `z_ma`, `hour`, `day_of_week`.
- CatBoost (compact) per‑symbol:
  - Entry: F1 ≈ 0.00 (очень малый тест по entry)
  - In‑trade: F1 ≈ 0.621–0.743 (30min/10min)
- Baseline (per‑symbol split):
  - Entry: F1 ≈ 0.07–0.32
  - In‑trade: F1 ≈ 0.415–0.637
- Прогон ALL‑тикеры × интервалы 10min/30min/1h:
  - entry_rows: 10min=191, 30min=223, 1h=147
  - intrade_rows: 10min=12160, 30min=14563, 1h=10273
  - CatBoost intrade F1: 10min≈0.743, 30min≈0.621, 1h≈0.696
  - Baseline intrade F1: 10min≈0.637, 30min≈0.415, 1h≈0.579
- 10min signal sweep (CatBoost intrade):
  - best: compact+time (hour/day_of_week) ≈ 0.743
  - compact v2 (location+time) ≈ 0.743
  - summary: `out/range_v3/ALL_10m_BASE_catboost_intrade_sig_summary.csv`
- Per‑symbol разбор (F1 + PF/winrate/Calmar по сделкам):
  - `out/range_v3/ALL_10m_BASE_catboost_intrade_per_symbol_summary.csv`
  - `out/range_v3/ALL_10m_BASE_catboost_intrade_per_symbol_summary_full.csv`
- CatBoost‑отчёт теперь включает per‑symbol trade‑метрики (PF/winrate/Calmar) в `per_symbol_trade_*`.
- Мультитаймфрейм‑идея сохранена, но базовый сигнал сейчас строим по 10min; 30min/1h подключим позже на уровне риск‑ядра (ранний TP/SL).
- Dataset A (entry candidates, 10min core): `out/range_v3/ALL_10m_CORE_entry_candidates.csv` (rows≈26960).
- Entry‑AI baseline (B3, mfe/mae): `y_entry`≈0.326, horizon=6, MFE≥0.0025 & MAE≥‑0.0025.
  - test: AUC≈0.513, F1≈0.514; top‑30% PF≈1.11, win≈0.566, Calmar≈1.08.
  - full: top‑30% PF≈1.16, win≈0.571, Calmar≈3.87.
- B4 (quantile) сохраняем как опцию для сильных тренд‑сигналов.
- Entry‑AI интегрирован в core backtest (10min, top‑30% gating, B3):
  - портфель: PF≈1.12, win≈0.512, Calmar≈0.008
  - baseline core (10min): PF≈1.03, win≈0.503, Calmar≈0.004
- Entry‑AI B4 включается только на high‑confidence тренд‑участках (slope>=0.00025):
  - портфель (B3+B4): PF≈1.13, win≈0.519, Calmar≈0.021
- Добавлен hard‑флаг `--no-hold-weekend` (core): закрываем позиции перед выходными и блокируем входы; PF/Win/Return слегка лучше, MaxDD хуже (оставляем как hard‑правило).
- Phase 1.1: core no-hold per-symbol stats (PF>=1: SNGS,SBER,TATN,OZON,GAZP,MGNT,ROSN,MTSS,PLZL,NLMK; PF<1: CHMF,VKCO,GMKN,PIKK,NVTK,YNDX,LKOH); файл `out/range_v3/ALL_10m_CORE_PROD_NOHOLD_per_symbol_stats.csv`.
- Phase 1.1: entry candidates sweep (B3) по PF/Calmar; тест топ‑30% лучше всего у `feat_no_signal` и `feat_signal_ret_vol`, full‑топ‑30% лидирует `feat_compact_ret_candle`. Тонкий свип top‑pct в core показал лучший риск‑профиль у `feat_no_signal` top_pct=0.08 (PF≈1.44, Calmar≈0.058, MaxDD≈‑0.48, trades=524); фиксируем конфиг и работаем только на 10min. Артефакты: `out/range_v3/ALL_10m_CORE_AI_FEAT_NOSIG_TP08_*`, per‑symbol `..._per_symbol_stats.csv`.
- Phase 1.1 impact: vs baseline core (no‑hold) PF 1.04 → 1.44, Calmar 0.0057 → 0.058, MaxDD −4.93 → −0.48, trades 4164 → 524. Per‑symbol deltas: топ улучшений `LKOH, GMKN, VKCO, ROSN, MTSS, CHMF`; ухудшения `TATN, NVTK, NLMK`.
- Phase 1.1 suitability reports (10min only): `out/range_v3/ALL_10m_CORE_AI_FEAT_NOSIG_TP08_ticker_stats.csv` и impact `..._ticker_impact_vs_baseline.csv` (риск‑ранжирование по PF/Calmar, low_trades flag).
- Phase 1.1 trend split (10min, TP08): nontrend доминирует по сделкам и возврату; bad‑entries почти целиком `sl`/`gap_sl` (см. `..._trend_split_summary.json`, `..._trend_per_symbol_stats.csv`, `..._bad_entries.csv`, `..._bad_exit_reasons.csv`).
- Phase 1.1 entry‑patterns (10min, TP08): отчёты `..._entry_feature_summary.csv`, `..._entry_feature_bins.csv`, `..._entry_feature_categorical.csv` (анализ bad‑entries по фичам/часам/side).
- CatBoost intrade гипотезы (per‑symbol split, sweep):
  - best: signal+vol+band_width_pct ≈ 0.716
  - compact_v1 (4 фичи, 30min): 0.712
  - worst: signal_only ≈ 0.415; vol_only ≈ 0.600
- Feature sweep (CPU, per‑symbol): best F1 ≈ 0.716 (signal+vol+band_width_pct), worst ≈ 0.415 (signal_only).
- Сводка sweep: `out/range_v3/ALL_30m_BASE_catboost_intrade_feat_summary.csv`.
- CPU vs GPU (CatBoost intrade, compact, per‑symbol):
  - CPU: ~1.2s, F1 ≈ 0.712 (compact_v1, 30min)
  - GPU: ~24.8s, F1 ≈ 0.423 (AUC not on GPU)
- Отчёты:
  - `out/range_v3/ALL_30m_BASE_catboost_report.json`

---

## 0. Текущий статус (2026-01-19)

### Фаза 0 завершена
- Выполнены baseline-прогоны legacy Range V3:
  - `out/range_v3/BASE_SBER_30m_*`
  - `out/range_v3/ALL_30m_BASE_*`
- Добавлен и выполнен debug-инструмент сегментов:
  - `out/range_v3/SEGMENTS_DEBUG_SBER_30m_2024-01-15_*`
- Зафиксированы модульные блоки core_v4 в `docs/CONTRACTv4.10.md`.

### Phase 0.2 (core_v4) закреплена
- Модульность core_v4:
  - `core/blocks.py`, `core/risk.py`, `core/metrics.py`, `core/portfolio.py`, `core/stats.py`
- Risk core:
  - вход по open_{t+1},
  - SL/TP (включая gap-обработку),
  - лимиты `daily_dd_limit_pct`, `max_consecutive_losses`.
- Leakage Validator v1:
  - severity/allowlist,
  - auto-exclude самого валидатора,
  - CLI: `python -m app2.cli leakage-check ...`.

### Фаза 1 начата
- Dataset A (Entry Snapshots) сгенерирован:
  - `out/range_v3/ALL_10m_BASE_entry_snapshots.csv`
  - `out/range_v3/ALL_30m_BASE_entry_snapshots.csv`
  - `out/range_v3/ALL_1h_BASE_entry_snapshots.csv`
- Dataset B (In-Trade Time-Series) — генератор и CLI готовы:
  - `out/range_v3/ALL_10m_BASE_intrade_timeseries.csv`
  - `out/range_v3/ALL_30m_BASE_intrade_timeseries.csv`
  - `out/range_v3/ALL_1h_BASE_intrade_timeseries.csv`
  - `out/range_v3/*_intrade_timeseries_meta.json`
- Текущий фокус по сигналам: базовый TF = 10min; 30min/1h отложены до стабилизации логики (интеграция в risk core/TP‑SL позже).
- Отчёт по `y_exit`:
  - `out/range_v3/ALL_30m_BASE_intrade_y_exit_report.json`
  - `out/range_v3/ALL_30m_BASE_intrade_y_exit_by_symbol.csv`
- Приняты параметры `y_exit` (Dataset B):
  - `exit_improve_threshold=0.0005`, `exit_min_bars=2`
  - итоговая доля `y_exit=1`: **990 / 4744 (0.209)**
- Запущен базовый ML-бейзлайн (Dataset A/B):
  - отчёт: `out/range_v3/ALL_30m_BASE_ml_baseline.json`
- Абляция без proxy‑фичей (Phase 1.1):
  - отчёт: `out/range_v3/ALL_30m_BASE_ml_baseline_ablation.json`
- Baseline compact (без proxy, fixed features):
  - отчёты: `out/range_v3/ALL_10m_BASE_ml_baseline_compact.json`, `out/range_v3/ALL_30m_BASE_ml_baseline_compact.json`, `out/range_v3/ALL_1h_BASE_ml_baseline_compact.json`
- CatBoost (compact) + per‑symbol метрики:
  - отчёт: `out/range_v3/ALL_30m_BASE_catboost_report.json`
- CatBoost intrade compact (ALL‑тикеры × интервалы):
  - `out/range_v3/ALL_10m_BASE_catboost_intrade_compact.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_compact.json`
  - `out/range_v3/ALL_1h_BASE_catboost_intrade_compact.json`
- CatBoost intrade варианты (per‑symbol split):
  - `out/range_v3/ALL_10m_BASE_catboost_intrade_compact.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_compact.json`
  - `out/range_v3/ALL_1h_BASE_catboost_intrade_compact.json`
  - `out/range_v3/ALL_10m_BASE_catboost_intrade_sig_*.json`
  - `out/range_v3/ALL_10m_BASE_catboost_intrade_sig_summary.csv`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_compact_cpu.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_compact_gpu.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_returns.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_bands.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_signal_only.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_signal.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_dist_from_ma.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_band_pos.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_band_width_pct.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_edge_proximity.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_z_ma.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_range_vs_atr.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_no_atr_14_pct.json`
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_feat_*.json` (feature sweep)
  - `out/range_v3/ALL_30m_BASE_catboost_intrade_feat_summary.csv`
- Итоги feature sweep (intrade, F1, CPU):
  - best: signal+vol+band_width_pct ≈ 0.716
  - strong: bands_only / signal+band_width_pct / vol+band_width_pct ≈ 0.716
  - compact_v1 (4 фичи): 0.712
  - weak: signal_only ≈ 0.415; vol_only ≈ 0.600
- Итоги 10min signal sweep (CatBoost intrade):
  - best: compact+time (hour/day_of_week) ≈ 0.743
  - compact v2 (location+time) ≈ 0.743
  - summary: `out/range_v3/ALL_10m_BASE_catboost_intrade_sig_summary.csv`
- MA/полосы пересчитаны по валидным сегментам (EMA, `ma_len=20`).
- Return‑фичи добавлены (см. `app2/range/features.py`).
- Time‑фичи добавлены из bar‑datetime: `hour`, `day_of_week`.
- Leakage report (reviewed):
  - `out/leakage_report.json`
  - 1 finding (high): `app2/range/dataset.py` uses `shift(-1)` for offline `y_exit` label (ожидаемо, принято).

---

## 1. Структура репозитория (напоминание)

В корне:
- `data/`, `processed/`, `app2/`, `app3/`, `docs/`, `archive/`.

Внутри `app2/range/`:
- `range_v3_legacy.py` — legacy ядро Range V3,
- `v3_backtest_legacy.py` — backtest для legacy,
- `core/` — core_v4 (modular),
- `debug_segments.py` — debug сегментов Range V3,
- `make_datasets.py` — генерация Dataset A/B.

Архивная копия `app2` сохранена в `archive/app2_2026-01-19/`.

---

## 2. Ключевые команды (актуальные)

Базовое окружение проекта: `C:\Users\nitro\moex-ai-bot\.venv1`  
(для запусков использовать `C:\Users\nitro\moex-ai-bot\.venv1\Scripts\python.exe`).

Производительность:
- По умолчанию используется multiprocessing (auto = все ядра минус одно).
- Глобально можно задать `APP2_N_JOBS` или `MOEX_N_JOBS`.
- Для RTX5080 в будущем: `--task-type GPU --devices 0` (CatBoost).
- Пример (PowerShell): `$env:APP2_N_JOBS=8`
- GPU detection: `catboost.utils.get_gpu_device_count() = 1`.

### Legacy baseline (Phase 0)
```bash
python -m app2.cli range-v3-backtest \
  --symbols SBER \
  --interval 30min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/BASE_SBER_30m \
  --tag v3seg_base \
  --engine legacy \
  --n-jobs 0
```

### Debug сегментов
```bash
python -m app2.cli range-debug-segments \
  --symbol SBER \
  --interval 30min \
  --date 2024-01-15 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/SEGMENTS_DEBUG_SBER_30m_2024-01-15
```

### Leakage Validator v1
```bash
python -m app2.cli leakage-check \
  --paths app2/ data/ \
  --extensions .py,.csv \
  --min-severity medium \
  --out out/leakage_report.json
```

### Dataset A/B (Entry + In-Trade)
```bash
python -m app2.cli range-v3-make-datasets \
  --symbols SBER GAZP LKOH GMKN ROSN \
  --interval 30min \
  --out-prefix out/range_v3/ALL_30m_BASE \
  --tag v3seg_base \
  --mode both \
  --n-jobs 0
```

### CatBoost (venv1, compact)
```bash
C:\Users\nitro\moex-ai-bot\.venv1\Scripts\python.exe -m app2.range.catboost_train \
  --mode both \
  --feature-set compact \
  --split-mode per_symbol \
  --thread-count 0

# GPU (в будущем, при готовом CUDA/драйверах):
#   --task-type GPU --devices 0

# Hypothesis examples:
#   --exclude dist_from_ma,band_pos,z_ma,edge_proximity
#   --include ret_1,ret_3,ret_6,ret_mean_20,ret_vol_20
```

### Core AI gating (фиксированный конфиг, 10min)
```bash
python -m app2.cli range-v3-backtest \
  --symbols SBER GAZP LKOH GMKN YNDX ROSN NVTK NLMK MTSS TATN CHMF SNGS PIKK PLZL MGNT VKCO OZON \
  --interval 10min \
  --equity0 1000000 \
  --config-range app2/range/config.json \
  --out-prefix out/range_v3/ALL_10m_CORE_AI_FEAT_NOSIG_TP08 \
  --tag v3seg_core_ai_feat \
  --engine core \
  --entry-model-path models/sweeps/feat_no_signal/range_catboost_entry.cbm \
  --entry-model-mode top_pct \
  --entry-model-top-pct 0.08 \
  --entry-feature-include band_width_pct,range_vs_atr,atr_14_pct,dist_from_ma,band_pos,edge_proximity,z_ma,hour,day_of_week \
  --no-hold-weekend \
  --n-jobs 8
```

### Baseline ML (per‑symbol split)
```bash
C:\Users\nitro\moex-ai-bot\.venv1\Scripts\python.exe -m app2.range.baseline_ml \
  --feature-set compact \
  --split-mode per_symbol
```

Параметры генератора Dataset B (labeling, принятые):
```bash
  --exit-improve-threshold 0.0005 \
  --exit-min-bars 2
```

---

## 3. Что дальше (Phase 1)

Следующий шаг:
- Итерация 1 (Phase 1.1):
  - проверить per‑symbol метрики из CatBoost и выделить тикеры с сигналом/без;
  - зафиксировать финальный компактный набор фич (без proxy) для 10min;
  - сравнить CatBoost vs baseline/ablation по тем же фичам;
  - при необходимости — один раунд улучшения фич (не меняя таргет).
  - после стабилизации — подключить MTF (30min/1h) как сигнал для риск‑ядра TP/SL.

---

## 4. Роль LLM-инструментов

Правила — см. `docs/CONTRACTv4.10.md` и `docs/model_plan_v0.4.md`.

---

## 5. Правила экономии контекста (LLM)

1. Читать только фрагменты больших файлов (использовать `offset/limit`).
2. Не перечитывать один и тот же файл без необходимости.
3. Делать краткие сводки вместо вставки длинных блоков.
4. Не запускать лишние команды/поиски — только по делу.
5. Заводить новый чат после больших циклов чтения (docs/много файлов).

---

## 6. Пакет контекста для нового чата

Минимальный набор файлов, которые нужно **прочитать** в новом чате:
- Базовое окружение проекта: `C:\Users\nitro\moex-ai-bot\.venv1`.
- `docs/READMEv4.10.md` — этот файл (статус и команды).
- `docs/CONTRACTv4.10.md` — инварианты и data‑контракт (Dataset A/B, risk).
- `docs/model_plan_v0.4.md` — план фаз и следующие шаги.
- `app2/cli.py` — CLI‑входы.
- `app2/range/config.json` — параметры Range V3/core.
- `app2/range/dataset.py` — логика Dataset A/B.
- `app2/range/make_datasets.py` — генератор датасетов.
- `app2/range/features.py` — EMA/ret‑фичи.
- `app2/range/baseline_ml.py` — baseline/ablation.
- `app2/range/catboost_train.py` — CatBoost + per‑symbol отчёты.
- `app2/range/range_v3_legacy.py`, `app2/range/v3_backtest_legacy.py` — legacy baseline.
- `app2/range/core/`:
  - `engine.py`, `state_machine.py`, `backtest.py`, `risk.py`,
  - `metrics.py`, `portfolio.py`, `stats.py`, `blocks.py`, `geometry.py`.
- `app2/range/debug_segments.py` — debug сегментов Range V3.

Артефакты `out/`, которые стоит открыть при восстановлении контекста:
- `out/range_v3/ALL_{10m,30m,1h}_BASE_*_stats.json`, `*_trades.csv`, `*_snapshots.csv` (backtest legacy).
- `out/range_v3/ALL_{10m,30m,1h}_BASE_entry_snapshots.csv` + `*_meta.json`.
- `out/range_v3/ALL_{10m,30m,1h}_BASE_intrade_timeseries.csv` + `*_meta.json`.
- `out/range_v3/ALL_{10m,30m,1h}_BASE_ml_baseline_compact.json`.
- `out/range_v3/ALL_{10m,30m,1h}_BASE_catboost_intrade_compact.json`.
- `out/range_v3/ALL_10m_BASE_catboost_intrade_sig_*.json`.
- `out/range_v3/ALL_10m_BASE_catboost_intrade_sig_summary.csv`.
- `out/range_v3/ALL_10m_BASE_catboost_intrade_per_symbol_summary.csv`.
- `out/range_v3/ALL_10m_BASE_catboost_intrade_per_symbol_summary_full.csv`.
- `out/range_v3/ALL_10m_CORE_AI_ALL_10min_v3seg_core_ai_stats.json`.
- `out/range_v3/ALL_10m_CORE_AI_ALL_10min_v3seg_core_ai_per_symbol_stats_calmar.csv`.
- `out/range_v3/ALL_10m_CORE_AI_B3B4_ALL_10min_v3seg_core_ai_b3b4_stats.json`.
- `out/range_v3/ALL_10m_CORE_AI_B3B4_ALL_10min_v3seg_core_ai_b3b4_per_symbol_stats_calmar.csv`.
- `out/range_v3/ALL_30m_BASE_feature_report.json` + `*_entry.csv` + `*_intrade.csv`.
- `out/range_v3/ALL_30m_BASE_catboost_report.json`.
- `out/range_v3/ALL_30m_BASE_catboost_intrade_feat_*.json` + `out/range_v3/ALL_30m_BASE_catboost_intrade_feat_summary.csv`.




## Current Focus

Phase 2 - Diagnostic Range Isolation

Goal:
Validate core, NOT optimize.
