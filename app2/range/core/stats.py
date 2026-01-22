from __future__ import annotations

from typing import Dict, List


def pnls_to_stats(pnl_list: List[float]) -> Dict[str, float]:
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]
    total_pnl = float(sum(pnl_list))
    win_rate = len(wins) / len(pnl_list) if pnl_list else 0.0
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "pf": pf,
    }
