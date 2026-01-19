1. Почему упала тренд-диагностика

Запуск:

python -m app2.cli analyze-trades \
  --strategy trend \
  --symbols GAZP ROSN SBER LKOH GMKN YNDX NVTK NLMK MTSS TATN CHMF SNGS PIKK PLZL MGNT VKCO OZON \
  --interval 30min \
  --config app2/config.json \
  --equity0 1000000 \
  --profile aggressive \
  --out-prefix out/diag_trend


Логи:

GAZP: bars=67091, trades=6, total_return=nan

ROSN: bars=67091, trades=1, total_return=nan

Потом Matplotlib: “All-NaN slice encountered” →
ValueError: autodetected range of [nan, nan] is not finit при построении гистограммы.

Смысл:

Тренд-стратегия с текущими параметрами почти не торгует: 6 сделок на ~67k баров — это ни о чём.

Для некоторых тикеров (или конкретного набора сделок) trades["pnl_abs"] получается весь из NaN → np.histogram не может определить диапазон.

Это не «логическая ошибка стратегии», а:

Диагностика (analysis.py) не защищена от пустых/NaN-логов → падает при попытке строить график.

Параметры тренд-профиля слишком жёсткие → слишком мало сделок, метрики total_return = NaN, диагностика бессмысленна.

👉 Это надо зафиксировать в плане и ТЗ:

В analysis:

Перед построением гистограмм/графиков дропать NaN и проверять len > 0, иначе просто не рисовать.

Не считать метрики из пустых трейд-логов.

В стратегиях:

Ввести отдельные консервативные/агрессивные профили для trend с адекватной частотой сделок.

Делать свип около этих профилей, чтобы добиться хотя бы разумного уровня активности.

Пока код не трогаем — только фиксируем в ТЗ.

2. Сводка проблем и предложений (из твоих файлов) на фоне текущего проекта
2.1. Критические проблемы (по твоему документу)

Архитектура и циклические импорты

В примере из файла — классика: strategy -> features -> models -> strategy и риск «спагетти» при усложнении ML-части.
В текущем app2 это пока не в полную силу (ядро у нас rule-based), но:

Уже видели кучу мелких импортных проблем (циклы, неправильные ссылки, двойные main()).

Для APP3 и AI-уровня это станет критично, если не навести порядок сейчас.

👉 Вывод: на уровне ядра app2 надо жёстко держаться принципа:

модуль стратегий → чистая логика сигналов;

модуль риска → чистая логика размера позиции/стопов;

модуль ML (когда вернём) → не имеет прямой зависимости от конкретных стратегий, только от данных и интерфейсов.

Производительность

В файле явно отмечены:

медленные циклы в стратегиях;

необходимость векторизации и кэширования.

В реальности у нас:

уже есть ProcessPoolExecutor в param_sweep;

часть логики в rule_core и rule_strategies векторизована;

но свип по meanrev на 18 тикеров × большая сетка уже почти упирается в CPU.

👉 Вывод: всё, что в “Проблемы/Предложения” про:

векторизацию,

numba,

кэш (diskcache / LRU),
— это must-have на горизонте, но после того, как стратегии стабилизированы и нас устраивает логика.

Риск-менеджмент: портфель и корреляция

Сейчас в коде:

есть риск на сделку (risk_per_trade),

ATR-стопы,

но нет:

контроля суммарного риска по портфелю,

учёта корреляции (SBER/GAZP/ROSN и т.д.),

дневных/недельных лимитов.

👉 Вывод: это отдельный модуль “PortfolioRiskEngine” (как в твоём черновике),
но после того, как будут вылизаны одиночные стратегии и режимы — иначе мы просто красиво ограничим систематически убыточную стратегию.

ML/Ensemble/Uncertainty

В файлах есть:

идея ансамбля (LGBM/XGB/CatBoost + meta-модель + оценка неопределённости),

идеи по high-winrate моделям, SMOTE, несбалансированным классам и т.п.

Сейчас мы на уровне чистых rule-strategies + диагностика.

👉 Вывод:

ML-уровень логично накладывать сверху уже отфильтрованных, осмысленных правил (meta-labeling).

Для этого как раз и нужен тот анализ bar/trade-логов, который мы сейчас сделали для meanrev и хотим сделать для trend/breakout.

Метрики и мониторинг

Ты хочешь:

winrate > 58%, PF > 1.5 (чёткие целевые KPI),

плюс набор проф. метрик: Sharpe, Calmar, Omega, tail ratio, VaR/CVaR, RC-index и т.д.

Сейчас:

базовые метрики есть (winrate, total_return, max_drawdown, profit_factor),

но нет отдельного «мониторинг-слоя» и расширенных risk-метрик.

👉 Вывод: метрики — это не произвольный бонус, а часть ТЗ для улучшения стратегий:
при свипах мы должны оценивать не только total_return, но и PF, max_dd, Calmar и т.п.

3. Объединённая картинка: где мы сейчас и куда идём

Сейчас:

Ядро app2:

rule-based стратегии: trend, meanrev, breakout;

режимы: regime_detector, regime_rule_backtest;

свип по meanrev (param_sweep) с multiprocessing;

мульти-таймфреймовые данные через process-data;

forward-test готов;

диагностика bar/trade-логов (analyze-trades) для meanrev + conservative profile уже работает и дала понятную картинку для TATN/SNGS/ROSN/MTSS/…;

профили conservative / aggressive для meanrev в config.json.

Проблемы, которые вскрылись:

meanrev: портфель в лёгком минусе, edge виден только у части тикеров;

trend: профили слишком жёсткие → почти нет сделок; диагностика падает на NaN;

нет ещё консервативных/агрессивных профилей для trend/breakout;

свипы пока только по meanrev; trend/breakout не исследованы сетками;

портфельный риск и ML-уровень — в планах, но не включены в ядро.

Куда идём ближайшими шагами:

Дробим тикеры по пригодности:

где meanrev имеет право жить (TATN, возможно SNGS),

какие тикеры лучше обслуживать trend или breakout.

«Прокачиваем» сами rule-стратегии:

meanrev v2 + свип вокруг консервативных/агрессивных профилей;

trend v2 с более «живыми» параметрами (достаточно сделок),

breakout v2 с фильтрами объёма и уровней.

Свипы для всех стратегий / профилей:

раздельный свип для:

meanrev / conservative,

meanrev / aggressive,

trend / conservative,

trend / aggressive,

breakout / conservative,

breakout / aggressive,
— на всех 18 тикерах.

Режимная логика v2:

использовать новый анализ: meanrev только в range/low_vol, trend в trend, breakout в high_vol.

Потом уже — портфельный риск, ML, GPU и прочий «тяжёлый» арсенал.

4. ТЗ на правку стратегий (без кода, только постановка задач)

Разделю на три части: meanrev, trend, breakout + общие вещи.

4.1. MeanReversion v2

Цель:
Поднять качество сделок по meanrev, приблизиться к целям winrate > 58%, PF > 1.5 на «хороших» тикерах и хотя бы избавиться от системно убыточных паттернов на «плохих».

ТЗ по логике входа:

RSI + Bollinger → ужесточение экстремальности:

Усложнить условие входа: цена должна быть не просто ниже/выше полосы, а:

ниже lower_band – k·ATR (лонг),

выше upper_band + k·ATR (шорт),

RSI ближе к экстремумам (например, 20/80 для conservative, 25/75 для aggressive).

Фильтр по волатильности/режиму:

Не торговать meanrev в режиме high_vol (по regime_detector).

В идеале: включать meanrev только в range, а в trend/high_vol — отключать или передавать управление другой стратегии (это уже шаг вместе с режимной интеграцией).

Фильтр по объёму (опционально в v2, минимум — в v2.1):

Игнорировать сигналы при слишком низком объёме (thin markets),

Можно использовать «объём выше скользящей средней × factor».

ТЗ по выходам / риску:

Max_hold_bars:

Ввести жёсткий лимит на количество баров в сделке max_hold_bars (отдельный параметр, например 32–40),

Если ни TP, ни SL не сработали — закрыть по рынку, не тянуть дальше.

ATR-стопы и тейки:

Пересмотреть сетку sl_atr_mult / tp_mult в контексте meanrev:

meanrev по идее любит «чуть побольше TP, чем SL», но и не до безумия,

разные профили:

conservative: больший sl_atr_mult, умеренный tp_mult,

aggressive: меньше SL, агрессивный TP.

Отбрасывание «шумовых» сигналов:

min_gap_bars оставить, но пересмотреть значения:

слишком большой gap → мало сделок (как в trend),

слишком маленький → «пилит».

ТЗ по профилям и свипу:

В config.json:

для meanrev уже есть profiles.conservative / profiles.aggressive;
нужно:

уточнить значения (на основе диагностики),

вокруг каждого профиля задать свою локальную сетку в sweep.MeanRevParams (смещение ±1–2 шага по важным параметрам).

Свипы:

param-sweep должен уметь принимать --profile и свипать вокруг него;

для meanrev прогнать:

conservative + aggressive

на всех 18 тикерах,

с логированием лучших сетапов по PF, win_rate, max_drawdown.

4.2. Trend v2

Проблема сейчас:
Почти нет сделок, стратегия «задушена» порогами (trend_thr, min_gap_bars, EMA-параметрами).

Цель:
Сделать трендовую стратегию, которая:

генерирует разумное количество сделок (не 1–6 на 67k баров),

выбивает трендовые участки,

не убивает счёт на пилe (это уже фильтруется режимами/ATR).

ТЗ по логике входа:

Сигнал = нормированная разница EMA / ATR:

базовая идея уже есть, но:

снизить trend_thr для aggressive профиля,

подобрать комбинацию ema_fast, ema_slow, atr_len:

conservative: более медленные EMA, более высокий trend_thr,

aggressive: быстрее EMA, чуть ниже trend_thr.

Фильтр по волатильности:

тренд имеет смысл в:

high_vol (пробои),

устойчивых трендах (когда нормированная разница EMA стабильно > threshold).

можно:

отбрасывать сигналы, если ATR слишком мало (flat),

не торговать trend в «hard range» по regime_detector.

ТЗ по профилям и свипу:

В config.json добавить в profiles блоки:

profiles.trend.conservative

profiles.trend.aggressive

(мы сейчас не пишем код, просто ТЗ; по сути это словари с TrendParams + частично RuleBtParams).

Для свипа:

sweep.TrendParams сделать вокруг этих профилей: trend_thr, min_gap_bars, ema_fast/slow.

Доп. ТЗ по диагностике:

analyze-trades для trend:

обязателен skip/guard на:

trades.empty,

trades["pnl_abs"].dropna().empty,

иначе просто пропускать графики, но писать CSV и summary.

4.3. Breakout v2

Цель:
Из твоего файла с предложениями явно просится нормальный price/volume breakout: пробой уровней с фильтрами.

ТЗ по логике входа:

Канальный/уровневый breakout:

текущий channel_len / confirm_bars / min_gap_bars — оставить как основу,

дополнить идеей из «поддержка/сопротивление + объём»:

выход за максимум/минимум за N баров,

подтверждение по объёму: объём > rolling_mean × factor.

Фильтр по режиму:

breakout активно использовать в high_vol (по regime_detector),

опционально — отключать в range.

Профили и свип:

Аналогично:

profiles.breakout.conservative / profiles.breakout.aggressive,

sweep.BreakoutParams вокруг них: channel_len, confirm_bars, min_gap_bars.

4.4. Общие ТЗ (для всех стратегий)

Сводное log/diagnostics ядро:

единые форматы:

bar-лог: datetime, close, equity, position, signal, regime, atr, и т.д.;

trade-лог: entry_dt, exit_dt, side, pnl_abs, pnl_rel, max_favorable_excursion, max_adverse_excursion, exit_reason.

analysis должен уметь:

строить графики только если есть данные,

считать сводные CSV summary по тикерам/профилям/стратегиям.

Метрики качества с приоритетом:

при выборе «лучших» параметров смотреть, как минимум:

win_rate,

profit_factor,

max_drawdown,

Calmar,

количество сделок (не слишком мало, не High-frequency).

Свипы консервативных/агрессивных профилей для всех стратегий:

подготовить в config.json:

базовые профили для meanrev, trend, breakout,

локальные сетки вокруг каждого профиля;

param-sweep:

научить принимать --strategy для всех трёх (не только meanrev),

внутренне использовать соответствующие профили и сетки;

сценарии прогонов:

отдельный CSV на каждую комбинацию (strategy × profile),

по всем 18 тикерам.

5. План с учётом новых файлов и ТЗ (без кода)

Предлагаю такой уплотнённый план следующей итерации (без реализации, только постановка):

Этап 1. Диагностика и стабилизация

 Meanrev conservative — сделано, есть логи и summary.

 Trend aggressive — устранить NaN-проблему в диагностике, добиться разумного числа сделок (через профиль и свип).

 Breakout — сделать первый диагностический запуск (даже на текущих параметрах, просто чтобы увидеть логи).

Результат:
Есть diag-логи (bar/trade/summary) по всем стратегиями и профилям; диагностика не падает на NaN/empty.

Этап 2. Стратегии v2 (по ТЗ выше)

MeanRev v2:

ужесточённые входы (RSI/Bollinger/ATR),

max_hold_bars,

фильтр по high_vol / range.

Trend v2:

профили conservative/aggressive с разумной частотой сделок,

фильтр по волатильности/режимам.

Breakout v2:

уровни + объём,

фокус на high_vol.

Результат:
Обновлённая, более «умная» логика всех трёх стратегий (но ещё без портфельных рисков/ML).

Этап 3. Свипы профилей для всех стратегий

В config.json:

зафиксированы профили для всех трёх стратегий;

sweep.* определён вокруг профилей.

param-sweep:

умеет свипать trend, meanrev, breakout,

даёт 6 CSV: (strategy × profile).

Результат:
Сет параметров, где видны:

где каждая стратегия «живая»,

по каким тикерам она в принципе имеет право на существование,

какие сочетания profile+params имеют шанс дать PF > 1.5 и приемлемый win_rate.

Этап 4. Режимная интеграция v2

На основе diag-логов:

meanrev → только range / low_vol,

trend → trend,

breakout → high_vol.

regime-rule-backtest обновлён концептуально: не руками переключаем стратегии, а по стабильным правилам.

Этап 5+. Портфельный риск, ML, GPU, PostgreSQL

Реализовать PortfolioRiskEngine (лимиты, корреляция).

Перевести исторические трейды в ML-датасет:

фичи (RSI, ATR, z-score, сезонность, режим и т.д.),

таргет — качество сделок (meta-labeling).

Тест GPU/облака для тяжёлых свипов/ML.

Постепенно переносить всё это в APP3.

Черновик нового файла для переезда (условный ReadmeV4.md)

Ниже текст, который можно просто скопировать в новый ReadmeV4.md (или дописать в readmev3.md отдельным разделом):

# MOEX APP2+ – ReadmeV4 (чат от 2025-11-18)

## 1. Состояние проекта

- Рабочее ядро: `app2` + корень (`data/`, `processed/`, `out/`, `ai_models/`).
- `app` – первая версия, фактически архив.
- `app3` – целевая модульная архитектура, развиваем после стабилизации ядра APP2.
- Все результаты, логи и модели живут в корневом `out/` (через `app2/paths.py`).

## 2. Чистка репозитория

### 2.1. Что удалили/планируем удалить

- Внутри `app2/`:
  - полностью удаляем `app2/data/` и `app2/out/` (по чек-листу, всё должно жить в корне).
  - очищаем все `__pycache__/`.
- В корне:
  - можно удалить `.env`, `catboost_info/`, старые отладочные скрипты (`quick_*`, `test_moex.py`, `get_accounts.py` и т.п.), если не нужны для повседневной работы.
  - при необходимости архивируем Docker-инфраструктуру (`Dockerfile`, `docker-compose.yml`, `main.py`) и служебные файлы (`fixed_strategy_metrics.json`, `strict_calibration.json`, `app3.config.toml`, `patch.diff`).

### 2.2. Что считаем ядром и не трогаем

- `app2/cli.py`, `config.py`, `config.json`, `paths.py`, `utils.py`, `risk.py`, `validation.py`.
- `app2/data.py`, `data_pipeline.py`, `data_utils.py`.
- `app2/rule_strategies.py`, `rule_core.py`, `rule_backtest.py`.
- `app2/regime_detector.py`, `regime_rule_backtest.py`.
- `app2/param_sweep.py`, `forward_test.py`, `analysis.py`, `diagnostics.py`.
- `data/`, `processed/`, `out/`, `ai_models/`, `templates/`, `tests/`.

## 3. Профили и сетки параметров (ТЗ, без кода)

### 3.1. Trend

В `config.json` добавить:

```jsonc
"profiles": {
  "trend": {
    "conservative": {
      "TrendParams": {
        "ema_fast": 12,
        "ema_slow": 48,
        "atr_len": 14,
        "trend_thr": 2.5,
        "min_gap_bars": 30
      },
      "RuleBtParams": {
        "risk_per_trade": 0.001,
        "sl_atr_mult": 2.5,
        "tp_mult": 4.0
      }
    },
    "aggressive": {
      "TrendParams": {
        "ema_fast": 8,
        "ema_slow": 32,
        "atr_len": 14,
        "trend_thr": 2.0,
        "min_gap_bars": 20
      },
      "RuleBtParams": {
        "risk_per_trade": 0.0015,
        "sl_atr_mult": 2.0,
        "tp_mult": 3.0
      }
    }
  }
}


Локальные сетки (для param-sweep):

conservative:
ema_fast: [10, 12, 14], ema_slow: [40, 48, 56],
trend_thr: [2.0, 2.5, 3.0], min_gap_bars: [25, 30, 35].

aggressive:
ema_fast: [8, 10, 12], ema_slow: [28, 32, 40],
trend_thr: [1.8, 2.0, 2.2], min_gap_bars: [15, 20, 25].

3.2. Breakout

Расширяем BreakoutParams до:

"BreakoutParams": {
  "channel_len": 50,
  "confirm_bars": 2,
  "min_gap_bars": 20,
  "vol_window": 30,
  "vol_mult": 1.5
}


Добавляем профили:

"profiles": {
  "breakout": {
    "conservative": {
      "BreakoutParams": {
        "channel_len": 60,
        "confirm_bars": 2,
        "min_gap_bars": 30,
        "vol_window": 40,
        "vol_mult": 1.8
      },
      "RuleBtParams": {
        "risk_per_trade": 0.001,
        "sl_atr_mult": 2.0,
        "tp_mult": 4.0
      }
    },
    "aggressive": {
      "BreakoutParams": {
        "channel_len": 40,
        "confirm_bars": 1,
        "min_gap_bars": 15,
        "vol_window": 20,
        "vol_mult": 1.3
      },
      "RuleBtParams": {
        "risk_per_trade": 0.0015,
        "sl_atr_mult": 1.5,
        "tp_mult": 3.0
      }
    }
  }
}


Локальные сетки:

conservative:
channel_len: [50, 60, 70], confirm_bars: [2, 3],
min_gap_bars: [25, 30, 35], vol_window: [30, 40, 50], vol_mult: [1.6, 1.8, 2.0].

aggressive:
channel_len: [30, 40, 50], confirm_bars: [1, 2],
min_gap_bars: [10, 15, 20], vol_window: [15, 20, 30], vol_mult: [1.2, 1.3, 1.5].

4. Метрики свипа (ТЗ)

Для каждой комбинации (strategy × profile × grid):

Считаем метрики: total_return, max_drawdown, calmar, sharpe_ann, trade_count, win_rate, pnl_sum, pnl_mean, pnl_std + добавляем PF.

Фильтруем по жёстким условиям:

trade_count ≥ 50,

total_return > 0,

max_drawdown ≥ -0.2,

win_rate ≥ 0.55,

PF ≥ 1.2.

Сортируем выжившие комбинации по:

PF (desc),

trade_count (desc),

max_drawdown (desc),

calmar (desc),

sharpe_ann (desc).

Делаем агрегированные метрики по портфелю (суммарный PnL, PF_portfolio, max_dd_portfolio) и выбираем профиль/комбо для каждого strategy.

5. Следующий шаг

После фиксации профилей и сеток (и добавления PF в метрики):

обновить param_sweep под новый формат профилей,

запустить диагностические свипы для:

trend / conservative, trend / aggressive,

breakout / conservative, breakout / aggressive,

и только после этого переходить к аккуратной реализации MeanRev v2 по ТЗ.