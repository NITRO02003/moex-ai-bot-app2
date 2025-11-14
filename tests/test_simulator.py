import pandas as pd
from app3.engine.simulator import simulate, SimConfig

def test_simulate_basic():
    bars = pd.DataFrame({
        "dt": pd.date_range("2024-01-01 10:00", periods=6, freq="T"),
        "open": [10,10,10,10,10,10],
        "high": [10,10,11,11,11,11],
        "low":  [10,9,9,9,9,9],
        "close":[10,10,10.5,10.7,10.2,10.0],
        "volume":[0,0,0,0,0,0],
    })
    signals = pd.DataFrame({
        "dt": [bars["dt"].iloc[1], bars["dt"].iloc[4]],
        "side": [1, 0]
    })
    eq, pnl, fills = simulate(bars, signals, SimConfig(fees_bps=0.0, slippage_bps=0.0, cooldown_bars=0, min_hold_bars=0))
    assert len(pnl) == 1
    assert len(fills) == 2
