
import json
import os

def load_symbols(symbols):
    return symbols if symbols != ["all"] else [
        "GAZP", "ROSN", "SBER", "LKOH", "GMKN", "YNDX",
        "NVTK", "NLMK", "MTSS", "TATN", "CHMF",
        "SNGS", "PIKK", "PLZL", "MGNT", "VKCO", "OZON"
    ]

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
