import os
import pandas as pd
import numpy as np

def locate_out_dir() -> str:
    # 1) ENV override
    env = os.environ.get("APP_OUT_DIR")
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "out"),                               # app/out
        os.path.abspath(os.path.join(here, "..", "out")),        # project-root/out
        os.path.abspath("out"),                                   # CWD/out
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    # If none exist, prefer project-root/out and create it
    pref = os.path.abspath(os.path.join(here, "..", "out"))
    os.makedirs(pref, exist_ok=True)
    return pref

OUT_DIR = locate_out_dir()
def _path(name): return os.path.join(OUT_DIR, name)

PREFS = {
    "signal_conf": ["signal_conf","signal_conf_s","signal_conf_x","signal_conf_y"],
    "regime_conf": ["regime_conf","regime_conf_s","regime_conf_x","regime_conf_y"],
}

def _read(name):
    p = _path(name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" not in out.columns and "dt" in out.columns:
        out["ts"] = out["dt"]
    if "side" in out.columns:
        out["side"] = out["side"].astype(str).str.upper().str.strip()
    out = out.loc[:, ~out.columns.duplicated()]
    return out

def _pick_col(df: pd.DataFrame, prefs) -> str | None:
    for c in prefs:
        if c in df.columns:
            return c
    return None

def main():
    tr = _norm(_read("trades.csv"))
    sg = _norm(_read("signals.csv"))
    print("[info] OUT_DIR:", OUT_DIR)
    print("[info] trades rows:", len(tr), "signals rows:", len(sg))
    print("[info] trades columns:", list(tr.columns))
    print("[info] signals columns:", list(sg.columns))

    keys_order = (
        ["ts","symbol","side"],
        ["dt","symbol","side"],
        ["ts","symbol"],
        ["dt","symbol"],
    )
    merged = None
    used = None
    for keys in keys_order:
        if all(k in tr.columns for k in keys) and all(k in sg.columns for k in keys):
            right_cols = [c for c in ["signal_conf","regime_conf","regime"] if c in sg.columns]
            merged = tr.merge(
                sg[keys + right_cols].drop_duplicates(keys),
                on=keys, how="left", suffixes=("", "_s")
            )
            used = keys
            break
    if merged is None:
        print("[warn] cannot merge trades & signals — keys mismatch")
        merged = tr.copy()
        used = []
    print("[info] merge keys used:", used)

    sigc = _pick_col(merged, PREFS["signal_conf"])
    regc = _pick_col(merged, PREFS["regime_conf"])

    def _dist(label: str, col: str | None):
        if col is None or col not in merged.columns:
            print(f"[missing] {label} not found")
            return
        s = pd.to_numeric(merged[col], errors="coerce")
        if s.notna().any():
            print(f"[dist] {label}<{col}>: count={s.notna().sum()} "
                  f"min={s.min():.3f} p25={s.quantile(0.25):.3f} "
                  f"med={s.median():.3f} p75={s.quantile(0.75):.3f} max={s.max():.3f}")
        else:
            print(f"[dist] {label}<{col}>: all NaN")

    _dist("signal_conf", sigc)
    _dist("regime_conf", regc)

    if "pnl" in merged.columns and len(merged) > 0:
        r = pd.to_numeric(merged["pnl"], errors="coerce")
        print("[dist] pnl<pnl>: count=%d min=%.3f p25=%.3f med=%.3f p75=%.3f max=%.3f" %
              (r.notna().sum(), r.min(), r.quantile(0.25), r.median(), r.quantile(0.75), r.max()))
        print("[perf] count=", r.notna().sum(),
              "sum=", float(r.sum()),
              "mean=", float(r.mean()),
              "win_rate=", float((r > 0).mean()))
        if "symbol" in merged.columns:
            g = (merged.assign(_p=r).groupby("symbol").agg(
                trades=("symbol","size"),
                win_rate=("_p", lambda s: (s > 0).mean()),
                avg_pnl=("_p","mean"),
                total_pnl=("_p","sum"),
            ).reset_index())
            print("[perf] per symbol head:")
            print(g.head(10).to_string(index=False))

    if sigc is None:
        print("\n[GUIDE] 'signal_conf' отсутствует после merge. Убедись, что "
              "out/signals.csv содержит ключи [ts|dt, symbol, side] и колонку signal_conf.")
    else:
        print("\n[OK] signal_conf колонка найдена:", sigc)

if __name__ == "__main__":
    main()
