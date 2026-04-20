
import pandas as pd
import numpy as np
from pathlib import Path

SYMBOLS = ["SBER", "GAZP", "LKOH"]
OUT_DIR = Path("research/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANGE_THRESHOLD = 0.005   # 0.5%
VOL_MULT = 2.0            # volume spike

def load(symbol):
    for p in [
        Path(f"processed/{symbol}_10min.csv"),
        Path(f"data/{symbol}_10min.csv"),
        Path(f"processed/{symbol}.csv"),
        Path(f"data/{symbol}.csv"),
    ]:
        if p.exists():
            df = pd.read_csv(p)
            if "begin" not in df.columns:
                df = pd.read_csv(p, header=None)
                df.columns = df.iloc[0]
                df = df.iloc[1:]
            df["begin"] = pd.to_datetime(df["begin"], utc=True)
            df = df.sort_values("begin").reset_index(drop=True)
            return df
    raise FileNotFoundError(symbol)

def compute(df):
    df["range"] = (df["high"] - df["low"]) / df["open"]
    df["vol_mean"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_mean"] * VOL_MULT
    df["range_spike"] = df["range"] > RANGE_THRESHOLD
    df["trigger"] = df["range_spike"] & df["vol_spike"]

    rows = []
    for i, r in df.iterrows():
        if not r["trigger"]:
            continue

        def ret(n):
            if i+n >= len(df): return np.nan
            return (df.iloc[i+n]["close"] - r["close"]) / r["close"]

        rows.append({
            "time": r["begin"],
            "ret_1": ret(1),
            "ret_3": ret(3),
            "ret_6": ret(6),
            "ret_12": ret(12),
        })
    return pd.DataFrame(rows)

all_events = []

for s in SYMBOLS:
    try:
        df = load(s)
        ev = compute(df)
        ev["symbol"] = s
        all_events.append(ev)
    except Exception as e:
        print("err", s, e)

res = pd.concat(all_events, ignore_index=True)

summary = res.groupby("symbol").agg({
    "ret_1":"mean",
    "ret_3":"mean",
    "ret_6":"mean",
    "ret_12":"mean"
})

res.to_csv(OUT_DIR/"event_triggers.csv", index=False)
summary.to_csv(OUT_DIR/"event_trigger_summary.csv")

print("done")
