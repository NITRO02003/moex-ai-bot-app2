
"""
Trade-level feature sweep for range strategy.

This module aggregates *_snapshots.csv produced by range-backtest/range-batch
and computes univariate / bivariate statistics of trade outcomes by feature bins.

All outputs are expected to be written under the root-level out/ directory.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SweepConfig:
    snapshots_root: str
    interval: str
    tag: str
    symbols: Sequence[str]
    features: Sequence[str]
    out_prefix_root: str
    do_bivariate: bool
    bivar_pairs: Sequence[Tuple[str, str]]
    by_symbol: bool
    by_profile: bool


def _find_snapshot_paths(cfg: SweepConfig) -> List[str]:
    root = cfg.snapshots_root
    interval = cfg.interval
    tag = cfg.tag

    pattern_all = os.path.join(root, f"*_{interval}_{tag}*_snapshots.csv")
    all_paths = glob.glob(pattern_all)

    if not all_paths:
        print(f"[range-trade-sweep] no snapshots found by pattern: {pattern_all}")
        return []

    if len(cfg.symbols) == 1 and cfg.symbols[0].lower() == "all":
        paths = sorted(all_paths)
        print(f"[range-trade-sweep] using ALL symbols, snapshots files={len(paths)}")
        return paths

    wanted = {s.upper() for s in cfg.symbols}
    paths: List[str] = []
    for p in all_paths:
        base = os.path.basename(p)
        sym = base.split("_", 1)[0].upper()
        if sym in wanted:
            paths.append(p)

    paths = sorted(paths)
    print(
        f"[range-trade-sweep] symbols={sorted(wanted)} -> "
        f"matched snapshot files={len(paths)}"
    )
    return paths


def _ensure_profile_column(df: pd.DataFrame) -> pd.DataFrame:
    if "profile" not in df.columns:
        df = df.copy()
        df["profile"] = np.nan
    return df


def load_snapshots(paths: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[range-trade-sweep] failed to read {path}: {exc}")
            continue

        # required columns check (soft)
        required_cols = {"symbol", "entry_dt", "pnl_rel"}
        missing = required_cols.difference(df.columns)
        if missing:
            print(
                f"[range-trade-sweep] {path} missing required columns: {sorted(missing)}; skipping"
            )
            continue

        # parse entry_dt
        try:
            df["entry_dt"] = pd.to_datetime(df["entry_dt"])
        except Exception as exc:
            print(f"[range-trade-sweep] failed to parse entry_dt in {path}: {exc}")

        df = _ensure_profile_column(df)
        frames.append(df)

    if not frames:
        print("[range-trade-sweep] no valid snapshots loaded")
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)

    # add time-based features
    if "entry_dt" in all_df.columns:
        all_df["hour"] = all_df["entry_dt"].dt.hour
        all_df["dow"] = all_df["entry_dt"].dt.dayofweek

    return all_df


def _compute_univariate_for_feature(
    df: pd.DataFrame,
    feature: str,
    n_bins: int,
    by_symbol: bool,
    by_profile: bool,
) -> Optional[pd.DataFrame]:
    if feature not in df.columns:
        print(f"[range-trade-sweep] feature {feature!r} not in snapshots; skipping")
        return None

    ser = pd.to_numeric(df[feature], errors="coerce")
    mask_valid = np.isfinite(ser.values)
    ser_valid = ser[mask_valid]

    if ser_valid.empty:
        print(f"[range-trade-sweep] feature {feature!r} has no valid values; skipping")
        return None

    try:
        cats, bins = pd.qcut(
            ser_valid, q=n_bins, retbins=True, duplicates="drop"
        )
    except ValueError as exc:
        print(
            f"[range-trade-sweep] qcut failed for feature {feature!r}: {exc}; skipping"
        )
        return None

    # map back bin indices to full series (including NaNs / invalids as -1)
    bin_codes = pd.Series(-1, index=ser.index, dtype="int64")
    bin_codes.loc[ser_valid.index] = cats.cat.codes

    # ignore rows that did not fall into any bin (code = -1)
    mask_bins = bin_codes >= 0
    if not mask_bins.any():
        print(
            f"[range-trade-sweep] feature {feature!r}: no rows assigned to bins; skipping"
        )
        return None

    df_loc = df.loc[mask_bins].copy()
    df_loc["bin_idx"] = bin_codes[mask_bins]

    # build bin_low/bin_high mapping
    # bins is array of edges, length = n_bins_effective + 1
    # for bin k: [bins[k], bins[k+1]]
    edges = bins
    n_effective = len(edges) - 1

    # We may have fewer effective bins than requested; filter bin_idx accordingly
    df_loc = df_loc[df_loc["bin_idx"] < n_effective]

    df_loc["bin_low"] = df_loc["bin_idx"].map(lambda k: float(edges[int(k)]))
    df_loc["bin_high"] = df_loc["bin_idx"].map(lambda k: float(edges[int(k) + 1]))

    group_keys = ["feature", "bin_idx", "bin_low", "bin_high"]
    df_loc["feature"] = feature

    if by_symbol and "symbol" in df_loc.columns:
        group_keys.append("symbol")
    if by_profile and "profile" in df_loc.columns:
        group_keys.append("profile")

    records = []
    for keys, grp in df_loc.groupby(group_keys, dropna=False):
        grp = grp.copy()
        trades = len(grp)
        if trades == 0:
            continue

        pnl = pd.to_numeric(grp["pnl_rel"], errors="coerce")
        pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
        if pnl.empty:
            continue

        wins = (pnl > 0).sum()
        win_rate = wins / len(pnl)

        mean_pnl = pnl.mean()
        sum_pos = pnl[pnl > 0].sum()
        sum_neg = pnl[pnl < 0].sum()

        if sum_neg < 0:
            pf = float(sum_pos / abs(sum_neg)) if sum_pos > 0 else 0.0
        elif sum_pos > 0:
            pf = float("inf")
        else:
            pf = 0.0

        rec = {
            "feature": feature,
            "bin_idx": int(keys[1] if isinstance(keys, tuple) else grp["bin_idx"].iloc[0]),
            "bin_low": float(grp["bin_low"].iloc[0]),
            "bin_high": float(grp["bin_high"].iloc[0]),
            "trades": int(trades),
            "win_rate": float(win_rate),
            "pf": float(pf),
            "mean_pnl_rel": float(mean_pnl),
        }

        # append symbol/profile if present in keys
        if by_symbol and "symbol" in df_loc.columns:
            rec["symbol"] = grp["symbol"].iloc[0]
        if by_profile and "profile" in df_loc.columns:
            rec["profile"] = grp["profile"].iloc[0]

        records.append(rec)

    if not records:
        return None

    res = pd.DataFrame.from_records(records)

    # share: relative fraction of trades per feature (ignoring symbol/profile dimensions)
    total_by_feature = res.groupby("feature")["trades"].transform("sum")
    res["share"] = res["trades"] / total_by_feature.replace(0, np.nan)

    return res


def univariate_sweep(
    df: pd.DataFrame,
    features: Sequence[str],
    n_bins: int,
    by_symbol: bool,
    by_profile: bool,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for feat in features:
        res = _compute_univariate_for_feature(
            df=df,
            feature=feat,
            n_bins=n_bins,
            by_symbol=by_symbol,
            by_profile=by_profile,
        )
        if res is not None and not res.empty:
            frames.append(res)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _compute_bivariate_for_pair(
    df: pd.DataFrame,
    f1: str,
    f2: str,
    n_bins: int,
    by_symbol: bool,
    by_profile: bool,
) -> Optional[pd.DataFrame]:
    if f1 not in df.columns or f2 not in df.columns:
        print(
            f"[range-trade-sweep] pair ({f1!r}, {f2!r}) missing in snapshots; skipping"
        )
        return None

    s1 = pd.to_numeric(df[f1], errors="coerce")
    s2 = pd.to_numeric(df[f2], errors="coerce")

    mask_valid = np.isfinite(s1.values) & np.isfinite(s2.values)
    s1v = s1[mask_valid]
    s2v = s2[mask_valid]
    if s1v.empty or s2v.empty:
        print(f"[range-trade-sweep] pair ({f1!r}, {f2!r}) has no valid values; skipping")
        return None

    try:
        cats1, bins1 = pd.qcut(s1v, q=n_bins, retbins=True, duplicates="drop")
        cats2, bins2 = pd.qcut(s2v, q=n_bins, retbins=True, duplicates="drop")
    except ValueError as exc:
        print(
            f"[range-trade-sweep] qcut failed for pair ({f1!r}, {f2!r}): {exc}; skipping"
        )
        return None

    codes1 = pd.Series(-1, index=s1.index, dtype="int64")
    codes2 = pd.Series(-1, index=s2.index, dtype="int64")
    codes1.loc[s1v.index] = cats1.cat.codes
    codes2.loc[s2v.index] = cats2.cat.codes

    mask_bins = (codes1 >= 0) & (codes2 >= 0)
    if not mask_bins.any():
        print(
            f"[range-trade-sweep] pair ({f1!r}, {f2!r}): no rows assigned to bins; skipping"
        )
        return None

    df_loc = df.loc[mask_bins].copy()
    df_loc["bin_x_idx"] = codes1[mask_bins]
    df_loc["bin_y_idx"] = codes2[mask_bins]

    edges1 = bins1
    edges2 = bins2
    n1 = len(edges1) - 1
    n2 = len(edges2) - 1

    df_loc = df_loc[
        (df_loc["bin_x_idx"] < n1) & (df_loc["bin_y_idx"] < n2)
    ].copy()

    df_loc["bin_x_low"] = df_loc["bin_x_idx"].map(lambda k: float(edges1[int(k)]))
    df_loc["bin_x_high"] = df_loc["bin_x_idx"].map(
        lambda k: float(edges1[int(k) + 1])
    )
    df_loc["bin_y_low"] = df_loc["bin_y_idx"].map(lambda k: float(edges2[int(k)]))
    df_loc["bin_y_high"] = df_loc["bin_y_idx"].map(
        lambda k: float(edges2[int(k) + 1])
    )

    group_keys = [
        "feature_x",
        "feature_y",
        "bin_x_idx",
        "bin_x_low",
        "bin_x_high",
        "bin_y_idx",
        "bin_y_low",
        "bin_y_high",
    ]
    df_loc["feature_x"] = f1
    df_loc["feature_y"] = f2

    if by_symbol and "symbol" in df_loc.columns:
        group_keys.append("symbol")
    if by_profile and "profile" in df_loc.columns:
        group_keys.append("profile")

    records = []
    for keys, grp in df_loc.groupby(group_keys, dropna=False):
        grp = grp.copy()
        trades = len(grp)
        if trades == 0:
            continue

        pnl = pd.to_numeric(grp["pnl_rel"], errors="coerce")
        pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
        if pnl.empty:
            continue

        wins = (pnl > 0).sum()
        win_rate = wins / len(pnl)

        mean_pnl = pnl.mean()
        sum_pos = pnl[pnl > 0].sum()
        sum_neg = pnl[pnl < 0].sum()

        if sum_neg < 0:
            pf = float(sum_pos / abs(sum_neg)) if sum_pos > 0 else 0.0
        elif sum_pos > 0:
            pf = float("inf")
        else:
            pf = 0.0

        rec = {
            "feature_x": f1,
            "feature_y": f2,
            "bin_x_idx": int(grp["bin_x_idx"].iloc[0]),
            "bin_x_low": float(grp["bin_x_low"].iloc[0]),
            "bin_x_high": float(grp["bin_x_high"].iloc[0]),
            "bin_y_idx": int(grp["bin_y_idx"].iloc[0]),
            "bin_y_low": float(grp["bin_y_low"].iloc[0]),
            "bin_y_high": float(grp["bin_y_high"].iloc[0]),
            "trades": int(trades),
            "win_rate": float(win_rate),
            "pf": float(pf),
            "mean_pnl_rel": float(mean_pnl),
        }

        if by_symbol and "symbol" in df_loc.columns:
            rec["symbol"] = grp["symbol"].iloc[0]
        if by_profile and "profile" in df_loc.columns:
            rec["profile"] = grp["profile"].iloc[0]

        records.append(rec)

    if not records:
        return None

    res = pd.DataFrame.from_records(records)

    total_by_pair = res.groupby(["feature_x", "feature_y"])["trades"].transform("sum")
    res["share"] = res["trades"] / total_by_pair.replace(0, np.nan)

    return res


def bivariate_sweep(
    df: pd.DataFrame,
    feature_pairs: Sequence[Tuple[str, str]],
    n_bins: int,
    by_symbol: bool,
    by_profile: bool,
) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for f1, f2 in feature_pairs:
        res = _compute_bivariate_for_pair(
            df=df,
            f1=f1,
            f2=f2,
            n_bins=n_bins,
            by_symbol=by_symbol,
            by_profile=by_profile,
        )
        if res is not None and not res.empty:
            frames.append(res)
    return frames


def _parse_bivar_pairs(raw_pairs: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in raw_pairs:
        if ":" not in item:
            continue
        f1, f2 = item.split(":", 1)
        f1 = f1.strip()
        f2 = f2.strip()
        if not f1 or not f2:
            continue
        pairs.append((f1, f2))
    return pairs


def main(args) -> None:
    # Build config from args (as passed from CLI wrapper)
    cfg = SweepConfig(
        snapshots_root=getattr(args, "snapshots_root", "out/range"),
        interval=getattr(args, "interval", "30min"),
        tag=getattr(args, "tag", "rangeV2"),
        symbols=getattr(args, "symbols", ["all"]),
        features=getattr(
            args,
            "features",
            ["dist_from_ma", "band_pos", "atr_14_pct", "bar_range_pct", "bar_body_pct"],
        ),
        out_prefix_root=getattr(args, "out_prefix_root", "out/range"),
        do_bivariate=bool(getattr(args, "do_bivariate", False)),
        bivar_pairs=_parse_bivar_pairs(
            getattr(
                args,
                "bivar_pairs",
                ["atr_14_pct:band_pos", "band_pos:bar_range_pct", "atr_14_pct:bar_body_pct"],
            )
        ),
        by_symbol=bool(getattr(args, "by_symbol", False)),
        by_profile=bool(getattr(args, "by_profile", False)),
    )

    paths = _find_snapshot_paths(cfg)
    if not paths:
        print("[range-trade-sweep] nothing to do, exiting")
        return

    df = load_snapshots(paths)
    if df.empty:
        print("[range-trade-sweep] loaded empty DataFrame, exiting")
        return

    print(
        f"[range-trade-sweep] loaded snapshots: rows={len(df)}, "
        f"symbols={sorted(df['symbol'].unique())}"
    )

    # Univariate sweep
    uni = univariate_sweep(
        df=df,
        features=cfg.features,
        n_bins=10,
        by_symbol=cfg.by_symbol,
        by_profile=cfg.by_profile,
    )

    if not uni.empty:
        os.makedirs(cfg.out_prefix_root, exist_ok=True)
        uni_path = os.path.join(
            cfg.out_prefix_root,
            f"ALL_{cfg.interval}_{cfg.tag}_trades_univariate.csv",
        )
        uni.to_csv(uni_path, index=False)
        print(
            f"[range-trade-sweep] written univariate sweep -> {uni_path} "
            f"(rows={len(uni)})"
        )
    else:
        print("[range-trade-sweep] univariate sweep produced empty result")

    # Bivariate sweep
    if cfg.do_bivariate and cfg.bivar_pairs:
        for f1, f2 in cfg.bivar_pairs:
            frames = bivariate_sweep(
                df=df,
                feature_pairs=[(f1, f2)],
                n_bins=6,
                by_symbol=cfg.by_symbol,
                by_profile=cfg.by_profile,
            )
            if not frames:
                continue
            res = frames[0]
            os.makedirs(cfg.out_prefix_root, exist_ok=True)
            bivar_path = os.path.join(
                cfg.out_prefix_root,
                f"ALL_{cfg.interval}_{cfg.tag}_trades_bivariate_{f1}_{f2}.csv",
            )
            res.to_csv(bivar_path, index=False)
            print(
                f"[range-trade-sweep] written bivariate sweep ({f1},{f2}) -> "
                f"{bivar_path} (rows={len(res)})"
            )
