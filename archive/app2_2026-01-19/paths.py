
from __future__ import annotations
from pathlib import Path

ROOT = Path('.').resolve()
OUT_DIR = ROOT / 'out'
DATA_DIR = ROOT / 'data'
MODELS_DIR = OUT_DIR / 'models'
REPORTS_DIR = OUT_DIR / 'reports'
LOGS_DIR = OUT_DIR / 'logs'

for p in (OUT_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)
