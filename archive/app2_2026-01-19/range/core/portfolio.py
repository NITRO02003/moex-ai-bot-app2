from __future__ import annotations

from typing import Dict, Iterable, List

from .stats import pnls_to_stats


def calc_max_drawdown_from_pnls(pnls: Iterable[float]) -> float:
    equity = 0.0
    max_equity = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        max_equity = max(max_equity, equity)
        if equity < max_equity and max_equity > 0:
            dd = (equity - max_equity) / max_equity
            if dd < max_dd:
                max_dd = dd
    return max_dd


def build_portfolio_stats(
    pnls: Iterable[float], equity0: float, symbols: List[str]
) -> Dict[str, float | List[str]]:
    pnl_list = list(pnls)
    stats = pnls_to_stats(pnl_list)
    denom = equity0 * max(len(symbols), 1)
    total_return = stats["total_pnl"] / denom if denom != 0 else 0.0
    max_dd = calc_max_drawdown_from_pnls(pnl_list)
    return {
        "symbols": list(symbols),
        "total_pnl": stats["total_pnl"],
        "total_return": total_return,
        "win_rate": stats["win_rate"],
        "pf": stats["pf"],
        "max_drawdown": max_dd,
    }
