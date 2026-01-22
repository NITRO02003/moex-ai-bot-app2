"""Core package for rule-based and regime-based trading strategies.

This package contains modules to generate trading signals, run simple backtests
with fixed risk rules, detect high‐level market regimes, and stitch these
components together into an adaptive strategy.  The goal of this package
is to provide a clean, modular foundation for further research and
experimentation.  It is intentionally self‑contained: no external data
sources are assumed beyond simple CSV price files, and no external APIs
are required.

Modules
=======

rule_strategies
    Contains simple, deterministic trading rules such as trend following,
    mean reversion and breakout logic.  These functions consume price
    dataframes and return side series of +1/0/-1 to indicate long, flat
    or short positioning.

rule_backtest
    Implements a minimalistic event driven backtest that opens and
    closes positions based on the side series from `rule_strategies`.
    It applies fixed per‑trade risk sizing using ATR and enforces
    commissions and slippage.

regime_detector
    Provides simple market regime classification based on ATR, EMA
    differentials and volatility.  The detector labels each bar as
    belonging to a ``trend``, ``range`` or ``high_vol`` regime.

regime_rule_backtest
    Combines the above pieces: for each bar, it looks up the detected
    regime and delegates decision making to the appropriate rule
    strategy.  It then runs the combined side series through the
    rule backtester to produce equity curves and metrics.

cli
    A command–line interface exposing backtesting functionality.  Users
    can run simple rule‑based strategies or regime‑aware strategies
    across multiple symbols and output aggregated results as JSON.
"""

__all__ = [
    "rule_strategies",
    "rule_backtest",
    "regime_detector",
    "regime_rule_backtest",
    "cli",
]