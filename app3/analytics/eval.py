from __future__ import annotations
import pandas as pd

def compare_reports(reports: dict) -> pd.DataFrame:
    rows = []
    for sym, m in reports.items():
        row = {"symbol": sym}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "sharpe_ann" in df.columns:
        df = df.sort_values("sharpe_ann", ascending=False)
    return df
