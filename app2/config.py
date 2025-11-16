
import json

def load_config(path="app2/config.json"):
    with open(path, "r") as f:
        return json.load(f)
