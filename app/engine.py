
# app/engine.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from .logging_utils import setup_logging, get_logger
from .utils import load_all, slice_by_date, ensure_dtindex_utc
from .core import TradeCosts, PortfolioRiskManager, enhanced_performance_metrics, AIRiskOverlay
from .ai_models import AIStrategyModel, AIRiskModel
from .features import build_features_15m, FINAL_FEATURE_SET
from .signals_perf import build_signals_matrix, build_risk_series
from .config import config, validate_config

setup_logging()
logger = get_logger("BacktestEngine")

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
os.makedirs(OUT_DIR, exist_ok=True)

@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list
    positions: Dict[str, float]

class BacktestEngine:
    def __init__(self, cfg=config):
        self.cfg = cfg
        self.data: Dict[str, pd.DataFrame] = {}
        self.features: Dict[str, pd.DataFrame] = {}
        self.signals: Dict[str, pd.Series] = {}
        self.risk_series: Optional[pd.Series] = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        warns = validate_config(self.cfg)
        if warns:
            logger.warning("Configuration warnings:\n" + "\n".join(f"- {w}" for w in warns))
        data = load_all(self.cfg.bt_cfg.data_dir, self.cfg.symbols_cfg.symbols)
        for s in list(data.keys()):
            data[s] = slice_by_date(data[s], self.cfg.bt_cfg.start_date, self.cfg.bt_cfg.end_date)
        self.data = data
        return data

    def compute_features(self) -> Dict[str, pd.DataFrame]:
        feats = {}
        for s, df in self.data.items():
            if df is None or df.empty:
                continue
            feats[s] = build_features_15m(df)
        self.features = feats
        return feats

    def generate_signals(self) -> Dict[str, pd.Series]:
        ai = AIStrategyModel(threshold=self.cfg.strategy_cfg.ai_threshold) if self.cfg.strategy_cfg.use_ai else None
        sigs: Dict[str, pd.Series] = {}
        for s, f in self.features.items():
            if f is None or f.empty:
                continue
            X = f[[c for c in f.columns if c in FINAL_FEATURE_SET]] if ai else f
            sig = ai.predict_series(X) if ai else pd.Series(0.0, index=f.index)
            sigs[s] = sig
        self.signals = sigs
        return sigs

    def compute_risk_series(self) -> pd.Series:
        ai_risk = AIRiskModel()
        # union index and union-features average for risk model
        idxs = [df.index for df in self.data.values() if df is not None and not df.empty]
        if not idxs:
            self.risk_series = pd.Series(dtype="float64")
            return self.risk_series
        union = idxs[0]
        for j in range(1, len(idxs)):
            union = union.union(idxs[j])
        # averaged features per timestamp
        try:
            union_feats = pd.concat([fx.reindex(union).ffill() for fx in self.features.values()], axis=1).groupby(level=0, axis=1).mean()
        except Exception:
            union_feats = None
        rs = build_risk_series(ai_risk, union, union_feats=union_feats)
        self.risk_series = rs
        return rs

    def run_simulation(self) -> BacktestResult:
        # vector-ish simulation: rebalance to sign * budget each bar
        costs = TradeCosts(
            commission_per_side=self.cfg.costs_cfg.commission_per_side,
            tick_size=self.cfg.symbols_cfg.tick_size,
            typical_spread_ticks=self.cfg.symbols_cfg.typical_spread_ticks,
        )
        risk = PortfolioRiskManager(
            max_total_drawdown=self.cfg.risk_cfg.max_total_drawdown,
            dd_smooth_threshold=self.cfg.risk_cfg.dd_smooth_threshold,
            dd_cutoff_threshold=self.cfg.risk_cfg.dd_cutoff_threshold,
            floor_risk=self.cfg.risk_cfg.floor_risk,
            ai_overlay=AIRiskOverlay(series=self.risk_series),
        )

        # Build union index
        idxs = [df.index for df in self.data.values() if df is not None and not df.empty]
        if not idxs:
            return BacktestResult(pd.Series(dtype="float64"), [], {s:0.0 for s in self.data})
        union = idxs[0]
        for j in range(1, len(idxs)):
            union = union.union(idxs[j])
        union = union.sort_values()

        sig_mat = build_signals_matrix(self.signals, union)  # shape (t, n_symbols)

        # state
        pos = {s: 0.0 for s in self.data.keys()}
        cash = float(self.cfg.bt_cfg.initial_equity)
        trades = []
        equity_curve = []

        for t in union:
            # budget per bar
            # create a simple average price vector for symbols with price at t
            cur_equity = cash
            for s, q in pos.items():
                if q == 0: 
                    continue
                df = self.data[s]
                if t in df.index: cur_equity += float(df.loc[t, "close"]) * q
                else:
                    prev = df.loc[:t]
                    if len(prev) > 0: cur_equity += float(prev["close"].iloc[-1]) * q

            budget = risk.risk_budget(t, cur_equity)

            # iterate symbols
            for s, df in self.data.items():
                if t not in df.index: 
                    continue
                price = float(df.loc[t, "close"])
                sig = float(sig_mat.loc[t, s]) if s in sig_mat.columns else 0.0
                target = budget * (1.0 if sig > 0 else -1.0 if sig < 0 else 0.0)
                delta = target - pos.get(s, 0.0)
                if abs(delta) > 1e-9:
                    # trade with slip+commission
                    slip_ticks = (costs.typical_spread_ticks or {}).get(s, 1) + (costs.slippage_ticks_extra or 0)
                    slip = slip_ticks * (costs.tick_size or {}).get(s, 0.01)
                    trade_price = costs.round_price(s, price + (slip if delta > 0 else -slip))
                    fee = abs(trade_price * delta) * costs.commission_per_side
                    cash -= trade_price * delta + fee
                    trades.append((t.isoformat(), s, "BUY" if delta>0 else "SELL", float(delta), trade_price))
                    pos[s] = pos.get(s, 0.0) + float(delta)

            # mark-to-market equity
            mtm = cash
            for s, q in pos.items():
                if q == 0: continue
                df = self.data[s]
                if t in df.index: mtm += float(df.loc[t, "close"]) * q
                else:
                    prev = df.loc[:t]
                    if len(prev) > 0: mtm += float(prev["close"].iloc[-1]) * q
            equity_curve.append((t, mtm))

        equity = pd.Series([v for _, v in equity_curve], index=[t for t,_ in equity_curve], dtype="float64")
        return BacktestResult(equity=equity, trades=trades, positions=pos)

    def run(self) -> BacktestResult:
        self.load_data()
        self.compute_features()
        self.generate_signals()
        self.compute_risk_series()
        result = self.run_simulation()
        # outputs
        eq_path = os.path.join(OUT_DIR, "equity.csv")
        result.equity.to_csv(eq_path)
        import pandas as pd
        pd.DataFrame(self.signals).to_csv(os.path.join(OUT_DIR, "signals.csv"))
        pd.DataFrame(result.trades, columns=["dt","symbol","side","qty","price"]).to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
        metrics = enhanced_performance_metrics(result.equity, result.trades)
        with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics: {metrics}")
        return result
