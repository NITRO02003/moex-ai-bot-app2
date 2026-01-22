from __future__ import annotations
import json, pandas as pd
from .paths import DATA_DIR, REPORTS_DIR

def data_summary(symbols):
    out = {}
    for s in symbols:
        p = DATA_DIR / f"{s}.csv"
        if not p.exists():
            out[s] = {'rows': 0, 'first': None, 'last': None}
            continue
        df = pd.read_csv(p)
        idx = pd.to_datetime(df.iloc[:,0], errors='coerce', utc=True)
        out[s] = {'rows': int(len(df)), 'first': str(idx.min()), 'last': str(idx.max())}
    return out

def save_json(name: str, obj: dict):
    path = REPORTS_DIR / f"{name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)
