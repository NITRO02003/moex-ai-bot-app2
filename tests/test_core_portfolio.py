from app2.range.core.portfolio import build_portfolio_stats, calc_max_drawdown_from_pnls


def test_calc_max_drawdown_from_pnls():
    # Equity: 0 -> 10 -> 5 -> 12 -> 4; max DD = (4-12)/12 = -0.666...
    dd = calc_max_drawdown_from_pnls([10.0, -5.0, 7.0, -8.0])
    assert round(dd, 6) == round(-8.0 / 12.0, 6)


def test_build_portfolio_stats_basic():
    stats = build_portfolio_stats([10.0, -5.0, 15.0], equity0=100.0, symbols=["AAA", "BBB"])
    assert stats["total_pnl"] == 20.0
    assert stats["win_rate"] == 2 / 3
    assert stats["pf"] == 25.0 / 5.0
    assert stats["total_return"] == 20.0 / 200.0


def test_build_portfolio_stats_empty():
    stats = build_portfolio_stats([], equity0=100.0, symbols=[])
    assert stats["total_pnl"] == 0.0
    assert stats["win_rate"] == 0.0
    assert stats["pf"] == 0.0
    assert stats["total_return"] == 0.0
