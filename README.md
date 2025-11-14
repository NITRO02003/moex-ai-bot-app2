# app3 — Clean Backtester v2

## Quick start
```
python -m app3 data load --symbol SBER --base data
python -m app3 backtest run --symbols SBER GAZP LKOH GMKN ROSN YNDX   --min-conf 0.7 --top-k-per-day 3   --cooldown-bars 20 --min-hold-bars 10   --time-start 11:10 --time-end 18:30   --fees-bps 1.0 --slippage-bps 0.0   --periods-per-year 252   --out out/reports/app3_summary.json
```
