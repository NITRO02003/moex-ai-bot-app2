import json
import os
from typing import List, Any, Dict


# Базовый универсум тикеров (используется, если в CLI указать --symbols all)
DEFAULT_SYMBOLS: List[str] = [
    "SBER", "GAZP", "LKOH", "GMKN", "YNDX", "ROSN",
    "NVTK", "NLMK", "MTSS", "TATN", "CHMF", "SNGS",
    "PIKK", "PLZL", "MGNT", "VKCO", "OZON",
]


def load_symbols(symbols: List[str]) -> List[str]:
    """Нормализует список тикеров.

    - Если передано ['all'] (регистр не важен) — возвращает DEFAULT_SYMBOLS.
    - Иначе возвращает список как есть.
    """
    if not symbols:
        return []

    if len(symbols) == 1 and symbols[0].lower() == "all":
        return DEFAULT_SYMBOLS

    return symbols


def save_json(obj: Any, path: str) -> None:
    """Сохраняет объект в JSON с созданием директорий."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
