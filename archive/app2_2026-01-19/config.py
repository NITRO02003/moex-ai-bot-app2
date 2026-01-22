import json


def load_config(path: str = "app2/config.json"):
    """Load a JSON configuration file.

    Parameters
    ----------
    path : str, optional
        Path to the configuration JSON. Defaults to ``"app2/config.json"``.

    Returns
    -------
    dict
        The parsed JSON as a Python dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
