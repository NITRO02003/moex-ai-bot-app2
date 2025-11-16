"""Market regime‑aware rule backtesting.

This module glues together the simple regime detector from
``app2.regime_detector`` and the rule‑based strategies from
``app2.rule_strategies``.  For each bar the current market regime
(``trend``, ``range`` or ``high_vol``) is detected and the
corresponding rule strategy's signal is used.  The combined side
series is then passed through the existing rule backtester to
calculate equity curves and performance metrics.

The goal of this module is to demonstrate how regime awareness can
reduce unnecessary trades and focus on the appropriate strategy for
prevailing conditions.  All parameters controlling the detector,
strategies and risk sizing are encapsulated in dataclasses for ease
of experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .regime_detector import RegimeParams, detect_regime
from .rule_strategies import (
    TrendParams,
    MeanRevParams,
    BreakoutParams,
    generate_trend_signals,
    generate_meanrev_signals,
    generate_breakout_signals,
)
from .rule_backtest import RuleBtParams, run_rule_symbol


@dataclass
class RegimeRuleBtParams:
    """Parameter bundle for regime‑aware rule backtest.

    Attributes
    ----------
    regime_params : RegimeParams
        Controls the lookbacks and thresholds of the regime detector.
    trend_params : TrendParams
        Parameters for the trend following rule.
    meanrev_params : MeanRevParams
        Parameters for the mean‑reversion rule.
    breakout_params : BreakoutParams
        Parameters for the breakout rule.  Only used in high
        volatility regime by default.
    rule_bt_params : RuleBtParams
        Risk and trade sizing configuration passed to the lower level
        backtester.
    use_breakout_in_high_vol : bool, default True
        If ``True``, the breakout strategy is used during high
        volatility regimes.  If ``False``, the system stays flat in
        high volatility.
    """

    # Use default_factory for dataclass fields containing mutable objects.
    regime_params: RegimeParams = field(default_factory=RegimeParams)
    trend_params: TrendParams = field(default_factory=TrendParams)
    meanrev_params: MeanRevParams = field(default_factory=MeanRevParams)
    breakout_params: BreakoutParams = field(default_factory=BreakoutParams)
    rule_bt_params: RuleBtParams = field(default_factory=RuleBtParams)
    # Whether to enable breakout trades during high volatility regimes.
    # By default this is False to avoid over‑trading when the market
    # is extremely volatile.  Set to True to allow breakout signals in
    # high volatility.
    use_breakout_in_high_vol: bool = False


def run_symbol(
    prices: pd.DataFrame,
    params: RegimeRuleBtParams | None = None,
    equity0: float = 1_000_000.0,
) -> dict[str, object]:
    """Backtest a regime‑aware rule strategy on a single symbol.

    This function first detects the market regime for each bar using
    ``detect_regime``.  It then generates signals for each of the
    supported rule strategies and constructs a single side series by
    selecting the signal that corresponds to the current regime.  The
    resulting side series is passed to ``run_rule_symbol`` along with
    risk parameters.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price dataframe with at least ``close``, ``high`` and ``low`` columns.
    params : RegimeRuleBtParams, optional
        Bundle of parameters controlling the detector, strategies and risk.
    equity0 : float, default 1_000_000.0
        Starting capital in monetary units.

    Returns
    -------
    dict
        A dictionary containing the equity curve, pnl series and
        summary metrics from the underlying rule backtest.
    """
    if params is None:
        params = RegimeRuleBtParams()
    # Detect regimes
    regimes = detect_regime(prices, params.regime_params)
    # Precompute signals for each strategy
    sig_trend = generate_trend_signals(prices, params.trend_params)
    sig_range = generate_meanrev_signals(prices, params.meanrev_params)
    sig_break = generate_breakout_signals(prices, params.breakout_params)
    # Initialise combined side series as flat
    side = pd.Series(0, index=prices.index, dtype="int8")
    # Assign signals by regime
    trend_mask = regimes == "trend"
    range_mask = regimes == "range"
    high_mask = regimes == "high_vol"
    side.loc[trend_mask] = sig_trend.loc[trend_mask]
    side.loc[range_mask] = sig_range.loc[range_mask]
    if params.use_breakout_in_high_vol:
        side.loc[high_mask] = sig_break.loc[high_mask]
    else:
        # Remain flat in high volatility regime
        side.loc[high_mask] = 0
    # Pass to the rule backtest engine
    result = run_rule_symbol(prices, side, params.rule_bt_params, equity0)
    return result