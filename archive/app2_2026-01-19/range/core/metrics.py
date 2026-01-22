from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .stats import pnls_to_stats


def compute_trade_pnl(entry_price: float, exit_price: float, qty: float, side: int) -> float:
    return (exit_price - entry_price) * qty * side


def compute_pnl_rel(pnl: float, equity: float) -> float:
    return pnl / equity if equity != 0 else 0.0


 


def build_symbol_metrics(
    symbol: str,
    pnls: Iterable[float],
    equity0: float,
    max_dd: float,
    equity_final: float,
    circuit_breaker_hit: bool,
    circuit_breaker_time: Optional[str],
) -> Dict[str, float | str | bool | None]:
    pnl_list = list(pnls)
    stats = pnls_to_stats(pnl_list)
    total_return = stats["total_pnl"] / equity0 if equity0 != 0 else 0.0
    return {
        "symbol": symbol,
        "trades": len(pnl_list),
        "total_pnl": stats["total_pnl"],
        "total_return": total_return,
        "win_rate": stats["win_rate"],
        "gross_profit": stats["gross_profit"],
        "gross_loss": stats["gross_loss"],
        "pf": stats["pf"],
        "max_drawdown": max_dd,
        "equity0": equity0,
        "equity_final": equity_final,
        "circuit_breaker_hit": bool(circuit_breaker_hit),
        "circuit_breaker_time": circuit_breaker_time,
    }


 
