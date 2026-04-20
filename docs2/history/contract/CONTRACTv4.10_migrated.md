# CONTRACTv4.10 — Архитектурный и data-контракт MOEX APP2+ (Range core_v4, modular blocks)

Файл: `docs/CONTRACTv4.10.md`  
База: `docs/CONTRACTv4.9.md`  
Связан с:
- `docs/READMEv4.9.md` — статус проекта и high-level план,
- `docs/model_plan_v0.4.md` — детальный план развития модели.

Это версия **v4.10** фиксирует модульные блоки core_v4 и границы ответственности.
Все положения CONTRACTv4.9 сохраняются, если не изменены ниже.

---

## 0. Область действия (дополнение)

CONTRACTv4.10 дополняет зону `app2/range/core/` и вводит обязательные блоки:

- `engine.py` — точка входа core_v4 (подготовка параметров, вызовы state machine).
- `state_machine.py` — генерация сигналов и debug_info (каузальная логика, без циклов по барам).
- `geometry.py` — геометрия диапазонов (L/U/H/M, geo_class, geo_valid_box), **только аналитика**.
- `risk.py` — риск-ядро core_v4: расчёт размера позиции, SL/TP, дневные лимиты, consecutive losses.
- `metrics.py` — метрики на уровне одной сделки/тикера (PnL, per-symbol stats).
- `portfolio.py` — агрегаты портфеля (PF/WinRate/MaxDD/TotalReturn) по списку сделок.
- `stats.py` — общие утилиты для статистик (например, `pnls_to_stats`).
- `backtest.py` — оркестрация: загрузка данных, запуск core, вызовы risk/metrics/portfolio, формирование артефактов.

---

## 1. Принцип модульности core_v4

### 1.1. Разделение ответственности

1. **State Machine** (`state_machine.py`) отвечает только за сигналы и debug_info.
2. **Risk** (`risk.py`) отвечает только за исполнение (entry/exit), ограничения и состояние риска.
3. **Metrics / Portfolio / Stats** отвечают только за вычисление метрик.
4. **Backtest** (`backtest.py`) не содержит бизнес-логики риска и метрик — только оркестрация.

### 1.2. Запреты (modularity constraints)

- Запрещено смешивать вычисления метрик и risk-логики внутри state machine.
- Запрещено дублировать функции статистик в нескольких местах (общие функции — в `stats.py`).
- Любые изменения в risk-логике должны происходить только в `risk.py`.

---

## 2. Risk core — уточнение контрактных блоков

Risk core обязан быть реализован через `risk.py` и не должен иметь side-effects вне:
- обновления risk-state,
- вычисления SL/TP/qty,
- принятия решения об открытии/закрытии.

Входы:
- `equity`, `risk_pct_per_trade`, `sl_pct`, `tp_pct`,
- дневные лимиты `daily_dd_limit_pct`,
- лимит `max_consecutive_losses`.

Выходы:
- `RiskState` (equity, max_dd, daily_dd, disabled flags),
- `post_circuit_breaker` флаг в сделках.

### 2.1. Weekend no-hold (core)
- Hard-правило риска, включаемое флагом `--no-hold-weekend` (core backtest).
- При включении: закрываем открытые позиции перед выходными и не открываем новые позиции на барах, которые уходят в weekend-gap.
- Правило не зависит от AI и не может быть переопределено soft-фильтрами.

### 2.2. Risk profiles (RangeV3)
- Конфиг: `app2/range/config.json` → `RangeV3.risk_profile` + `RangeV3.risk_profiles`.
- `prod_baseline`: боевой профиль с текущими лимитами риска.
- `diag_fullrun`: диагностический профиль с сильно ослабленными лимитами (для full-run анализа).

---

## 3. Метрики и портфельные агрегаты

- `metrics.py` — метрики на уровне тикера:
  - `compute_trade_pnl`, `compute_pnl_rel`, `build_symbol_metrics`.
- `portfolio.py` — агрегированные метрики портфеля:
  - `build_portfolio_stats`.
- `stats.py` — общий расчёт базовой статистики по массиву PnL:
  - `pnls_to_stats`.

Метрики не влияют на торговые решения и используются только для отчётности.

---

## 4. Dataset A/B (Phase 1) — data contract

Источник данных:
- артефакты `range-v3-backtest` (legacy/core) по каждому тикеру:
  - `*_trades.csv`,
  - `*_snapshots.csv`.

CLI-генератор:
```bash
python -m app2.cli range-v3-make-datasets \
  --symbols SBER GAZP LKOH GMKN ROSN \
  --interval 30min \
  --out-prefix out/range_v3/ALL_30m_BASE \
  --tag v3seg_base \
  --mode entry|intrade|both \
  --exit-improve-threshold 0.0000 \
  --exit-min-bars 0
```

### Dataset A — Entry Snapshots
- 1 строка на сделку.
- Join: `entry_dt` ←→ `snapshots.datetime`.
- Минимальный набор:
  - `symbol`, `entry_dt`, `exit_dt`, `direction`,
  - feature-колонки (все, кроме служебных: `datetime`, `symbol`, `open/high/low/close/volume` и т.п.),
  - label-метрики: `pnl_rel`, `pnl_abs`, `bars_in_trade`, `max_adverse_excursion` (если доступны).

### Dataset B — In-Trade Time-Series
- 1 строка на бар во время открытой сделки (`entry_dt`..`exit_dt` включительно).
- Цена для расчёта PnL:
  - по `close` (если есть), иначе по `open`.
- Минимальный набор колонок:
  - meta: `symbol`, `interval`, `trade_id`, `trade_uid`, `dt`,
  - trade-level: `entry_dt`, `exit_dt`, `direction`, `entry_price`, `exit_price`, `qty`,
    `trade_pnl_abs`, `trade_pnl_rel`, `trade_bars_in_trade`, `trade_exit_reason`,
  - bar-level: `bars_held`, `time_since_entry_min`, `pnl_abs`, `pnl_rel`,
    `mae_abs`, `mfe_abs`, `mae_rel`, `mfe_rel`, `y_exit`,
  - features из snapshots (по умолчанию все колонки кроме `datetime/symbol`).
- Допустимые дополнительные колонки:
  - `entry_*`, `exit_*`, `post_*`, `geo_*` (контекст входа из core/legacy).

### Label `y_exit` (v1, offline-only)
`y_exit = 1`, если:
- `pnl_rel_{t+1} >= trade_pnl_rel + exit_improve_threshold`, и
- `bars_held >= exit_min_bars`.

Иначе `y_exit = 0`. Последний бар сделки всегда `0`.

**Важно:** `y_exit` использует информацию следующего бара, поэтому это
**offline-лейбл**. Он не должен попадать в rolling-логику сигналов и не может
использоваться в live-режиме без новой версии контракта.

---

Этот CONTRACTv4.10 действует до следующей версии контракта и отражает текущий
приоритет **модульности** core_v4.

---

## 5. Статус Фазы 0 (фиксация)

- Фаза 0 завершена (2026-01-19), baseline-артефакты сохранены в `out/range_v3/`.
- Инструмент `range-debug-segments` добавлен для диагностики сегментов Range V3.
- Обновление READMEv4.8 будет выполнено отдельно (по лимиту контекста в чате).



## Regime Layer Contract (Diagnostic v0)

- Controls ONLY entry permission
- MUST NOT:
  - force exit
  - modify risk
  - generate signals

- v0 is diagnostic, self-confirming
- NOT production regime engine
