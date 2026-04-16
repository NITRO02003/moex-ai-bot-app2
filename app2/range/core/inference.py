from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .artifact_validator import validate_inference_artifact
from .inference_artifacts import InferenceArtifact


def run_entry_inference(
    *,
    artifact_path: str,
    snapshots_path: str,
    out_prefix: str,
    tag: str,
    expected_config_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load artifact + snapshots and emit scored entry decisions.

    Output contains the original snapshot rows plus:
    - entry_prob
    - entry_decision
    """
    validate_inference_artifact(
        artifact_path,
        expected_dataset_kind="entry",
        expected_config_path=expected_config_path,
    )
    art = InferenceArtifact.load_from_file(artifact_path)

    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover
        raise SystemExit("CatBoost is required for inference") from exc

    snapshots = Path(snapshots_path)
    if not snapshots.exists():
        raise ValueError(f"Snapshots CSV not found: {snapshots_path}")
    df = pd.read_csv(snapshots)
    missing = [c for c in art.features if c not in df.columns]
    if missing:
        raise ValueError(f"Snapshots CSV missing required feature columns: {missing}")

    model = CatBoostClassifier()
    model.load_model(art.model_path)
    x = df[art.features].copy()
    probs = model.predict_proba(x)[:, 1]
    threshold = float(art.threshold if art.threshold is not None else 0.5)
    decisions = (probs >= threshold).astype(int)

    out_df = df.copy()
    out_df["entry_prob"] = probs
    out_df["entry_decision"] = decisions

    out_path = Path(f"{out_prefix}_inference_{tag}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[range-inference] rows={len(out_df)} threshold={threshold:.4f}")
        print(f"[range-inference] saved {out_path}")
    return out_df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run offline entry inference from exported artifact")
    p.add_argument("--artifact-path", required=True, type=str)
    p.add_argument("--snapshots-path", required=True, type=str)
    p.add_argument("--out-prefix", required=True, type=str)
    p.add_argument("--tag", required=True, type=str)
    p.add_argument("--config-path", dest="expected_config_path", default=None, type=str)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    return p


def main(args=None) -> int:
    if args is None:
        ns = _build_parser().parse_args()
    else:
        ns = args
    run_entry_inference(
        artifact_path=ns.artifact_path,
        snapshots_path=ns.snapshots_path,
        out_prefix=ns.out_prefix,
        tag=ns.tag,
        expected_config_path=getattr(ns, "expected_config_path", None),
        verbose=getattr(ns, "verbose", True),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
