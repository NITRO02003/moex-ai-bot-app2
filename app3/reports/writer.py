from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..utils.io import ensure_dir, write_json

def write_json_report(report: dict, path: str | Path) -> None:
    write_json(report, path)

def write_dataframe_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(path)
    df.to_csv(path, index=False)
