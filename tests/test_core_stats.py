from app2.range.core.stats import pnls_to_stats


def test_pnls_to_stats_basic():
    stats = pnls_to_stats([10.0, -5.0, 0.0, 15.0, -10.0])
    assert stats["total_pnl"] == 10.0
    assert stats["win_rate"] == 2 / 5
    assert stats["gross_profit"] == 25.0
    assert stats["gross_loss"] == 15.0
    assert stats["pf"] == 25.0 / 15.0


def test_pnls_to_stats_empty():
    stats = pnls_to_stats([])
    assert stats["total_pnl"] == 0.0
    assert stats["win_rate"] == 0.0
    assert stats["gross_profit"] == 0.0
    assert stats["gross_loss"] == 0.0
    assert stats["pf"] == 0.0
