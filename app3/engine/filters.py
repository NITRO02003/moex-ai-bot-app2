from __future__ import annotations
import pandas as pd

def apply_filters(signals: pd.DataFrame, *,
                  min_conf: float | None = None,
                  top_k_per_day: int | None = None,
                  time_start: str | None = None,
                  time_end: str | None = None,
                  atr_threshold: float | None = None,
                  max_spread_bps: float | None = None,
                  use_regime: bool = False) -> pd.DataFrame:
    s = signals.copy()
    s["dt"] = pd.to_datetime(s["dt"])
    s = s.sort_values("dt")

    if min_conf is not None and "conf" in s.columns:
        s = s[s["conf"] >= float(min_conf)]

    if top_k_per_day and top_k_per_day > 0:
        s["date"] = s["dt"].dt.date
        order_cols = [c for c in ["conf","score"] if c in s.columns] or ["dt"]
        s = (s.sort_values(["date"] + order_cols, ascending=[True] + [False]*len(order_cols))
               .groupby("date", as_index=False)
               .head(top_k_per_day))

    if time_start and time_end:
        try:
            t_start = pd.to_datetime(time_start).time()
            t_end   = pd.to_datetime(time_end).time()
            s = s[(s["dt"].dt.time >= t_start) & (s["dt"].dt.time <= t_end)]
        except Exception:
            pass

    if atr_threshold is not None and "atr" in s.columns:
        s = s[s["atr"] >= float(atr_threshold)]
    if max_spread_bps is not None and "spread_bps" in s.columns:
        s = s[s["spread_bps"] <= float(max_spread_bps)]

    if use_regime and "regime_ok" in s.columns:
        s = s[s["regime_ok"]]

    if "side" not in s.columns:
        s["side"] = 0
    s["side"] = s["side"].astype(int).clip(-1, 1)

    return s[["dt","side"] + [c for c in s.columns if c not in ("dt","side","date")]]
