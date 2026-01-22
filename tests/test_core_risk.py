import pandas as pd

from app2.range.core.risk import (
    calc_sl_tp,
    eval_exit,
    init_risk_state,
    on_new_day,
    update_after_exit,
)


def test_eval_exit_sl_tp_same_bar_prefers_sl():
    sl_price, tp_price = calc_sl_tp(entry_price=100.0, sl_pct=0.05, tp_pct=0.05, side=1)
    need_close, exit_reason, exit_price = eval_exit(
        position_side=1,
        price=100.0,
        bar_open=100.0,
        bar_high=106.0,
        bar_low=94.0,
        sl_price=sl_price,
        tp_price=tp_price,
        sig=0,
        bars_held=1,
        max_bars_in_trade=10,
    )
    assert need_close is True
    assert exit_reason == "sl"
    assert exit_price == sl_price


def test_eval_exit_gap_sl_long_uses_open():
    sl_price, tp_price = calc_sl_tp(entry_price=100.0, sl_pct=0.05, tp_pct=0.05, side=1)
    need_close, exit_reason, exit_price = eval_exit(
        position_side=1,
        price=98.0,
        bar_open=90.0,
        bar_high=99.0,
        bar_low=89.0,
        sl_price=sl_price,
        tp_price=tp_price,
        sig=0,
        bars_held=1,
        max_bars_in_trade=10,
    )
    assert need_close is True
    assert exit_reason == "gap_sl"
    assert exit_price == 90.0


def test_eval_exit_gap_tp_long_uses_tp():
    sl_price, tp_price = calc_sl_tp(entry_price=100.0, sl_pct=0.05, tp_pct=0.05, side=1)
    need_close, exit_reason, exit_price = eval_exit(
        position_side=1,
        price=106.0,
        bar_open=110.0,
        bar_high=112.0,
        bar_low=105.0,
        sl_price=sl_price,
        tp_price=tp_price,
        sig=0,
        bars_held=1,
        max_bars_in_trade=10,
    )
    assert need_close is True
    assert exit_reason == "gap_tp"
    assert exit_price == tp_price


def test_eval_exit_gap_sl_short_uses_open():
    sl_price, tp_price = calc_sl_tp(entry_price=100.0, sl_pct=0.05, tp_pct=0.05, side=-1)
    need_close, exit_reason, exit_price = eval_exit(
        position_side=-1,
        price=102.0,
        bar_open=112.0,
        bar_high=113.0,
        bar_low=101.0,
        sl_price=sl_price,
        tp_price=tp_price,
        sig=0,
        bars_held=1,
        max_bars_in_trade=10,
    )
    assert need_close is True
    assert exit_reason == "gap_sl"
    assert exit_price == 112.0


def test_eval_exit_gap_tp_short_uses_tp():
    sl_price, tp_price = calc_sl_tp(entry_price=100.0, sl_pct=0.05, tp_pct=0.05, side=-1)
    need_close, exit_reason, exit_price = eval_exit(
        position_side=-1,
        price=95.0,
        bar_open=88.0,
        bar_high=96.0,
        bar_low=85.0,
        sl_price=sl_price,
        tp_price=tp_price,
        sig=0,
        bars_held=1,
        max_bars_in_trade=10,
    )
    assert need_close is True
    assert exit_reason == "gap_tp"
    assert exit_price == tp_price


def test_daily_dd_limit_blocks_new_entries():
    state = init_risk_state(100.0)
    on_new_day(state, pd.Timestamp("2024-01-10"))
    post_cb = update_after_exit(
        state,
        pnl=-3.0,
        ts=pd.Timestamp("2024-01-10 12:00:00"),
        daily_dd_limit_pct=0.02,
        max_consecutive_losses=None,
    )
    assert post_cb is False
    assert state.daily_dd <= -0.02
    assert state.daily_disabled is True


def test_consecutive_losses_trips_circuit_breaker():
    state = init_risk_state(100.0)
    on_new_day(state, pd.Timestamp("2024-01-10"))
    update_after_exit(
        state,
        pnl=-1.0,
        ts=pd.Timestamp("2024-01-10 10:00:00"),
        daily_dd_limit_pct=None,
        max_consecutive_losses=2,
    )
    post_cb = update_after_exit(
        state,
        pnl=-1.0,
        ts=pd.Timestamp("2024-01-10 11:00:00"),
        daily_dd_limit_pct=None,
        max_consecutive_losses=2,
    )
    assert post_cb is True
    assert state.stopped_by_circuit_breaker is True
    assert state.daily_disabled is True
