"""Utilities for exporting inference artifacts.

This module defines helper functions for constructing and persisting
``InferenceArtifact`` instances.  An inference artifact is the bundle of
metadata required to perform inference in a sandbox or online environment.

An artifact encapsulates:

* ``model_path``: Path to the serialized model (e.g., CatBoost ``.cbm`` file).
* ``features``: Ordered list of feature column names expected by the model.
* ``truth_policy``: ``"trades"`` for policy datasets or ``"candidates"`` for research.
* ``dataset_kind``: ``"entry"`` or ``"intrade"``, matching the dataset manifest.
* ``config_path``: Path to the range configuration used to generate the dataset.
* ``config_fingerprint``: SHA-256 hash of the config file contents (optional).
* ``label_mode`` and ``horizon``: Labeling parameters used when creating the dataset.
* ``threshold``: Numeric value used to convert model probabilities into decisions.
* ``risk_profile``: Name of the risk profile applied in backtesting.
* ``dataset_version``: Optional dataset or model version tag.
* ``extra``: Dictionary for arbitrary additional metadata.

The exporter reads the dataset manifest (``*_meta.json``) to populate
fields such as ``dataset_kind``, ``truth_policy``, ``label_mode`` and
``horizon``.  It can also compute a fingerprint of the range config file.

Example::

    from .artifact_exporter import export_inference_artifact

    export_inference_artifact(
        model_path="out/models/entry_model.cbm",
        dataset_meta_path="out/datasets/entry_snapshots_meta.json",
        config_path="app2/range/config.json",
        features=["feature1", "feature2", ...],
        threshold=0.55,
        risk_profile="default",
        output_path="out/artifacts/entry_model_policy.json",
    )

Once exported, the resulting JSON artifact can be loaded using
``InferenceArtifact.load_from_file`` and validated with the corresponding
validator functions.

"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .inference_artifacts import InferenceArtifact

__all__ = ["export_inference_artifact", "compute_config_fingerprint"]


def compute_config_fingerprint(config_path: Optional[str]) -> Optional[str]:
    """Compute a SHA-256 fingerprint of the given config file.

    If ``config_path`` is ``None`` or does not exist, returns ``None``.

    Parameters
    ----------
    config_path: Optional[str]
        Path to the JSON configuration file used to generate the dataset.

    Returns
    -------
    Optional[str]
        Hexadecimal SHA-256 digest of the file contents, or ``None`` if
        the file does not exist or ``config_path`` is ``None``.
    """
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        return None
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _load_meta(meta_path: str) -> Dict[str, Any]:
    """Load dataset manifest from a JSON file.

    The manifest must contain at least ``dataset_kind``, ``truth_policy``
    and ``config_path`` keys.  Additional fields (e.g., horizon, label_mode)
    may also be present.

    Parameters
    ----------
    meta_path: str
        Path to the ``*_meta.json`` file produced by ``make_datasets.py``.

    Returns
    -------
    Dict[str, Any]
        Parsed manifest as a dictionary.

    Raises
    ------
    ValueError
        If the file cannot be read or lacks required keys.
    """
    p = Path(meta_path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"Dataset meta file not found: {meta_path}")
    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    required_keys = {"dataset_kind", "truth_policy", "config_path"}
    missing = required_keys - meta.keys()
    if missing:
        raise ValueError(f"Dataset meta missing required keys: {missing}")
    return meta


def export_inference_artifact(
    model_path: str,
    dataset_meta_path: str,
    features: Iterable[str],
    threshold: float,
    risk_profile: str,
    output_path: str,
    *,
    config_path: Optional[str] = None,
    dataset_version: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> InferenceArtifact:
    """Create and serialize an ``InferenceArtifact``.

    Parameters
    ----------
    model_path: str
        Path to the serialized model (e.g., CatBoost ``.cbm`` file).

    dataset_meta_path: str
        Path to the dataset manifest file (``*_meta.json``) produced by
        ``make_datasets.py``.  The manifest supplies fields such as
        ``dataset_kind``, ``truth_policy``, ``config_path`` and
        labeling parameters.

    features: Iterable[str]
        Ordered list of features used by the model.

    threshold: float
        Decision threshold applied to model probabilities.  Values between
        0 and 1 typically correspond to positive class probabilities.

    risk_profile: str
        Name of the risk profile used during backtesting.

    output_path: str
        Destination path for the exported artifact (JSON).

    config_path: Optional[str], optional
        Path to the range configuration file.  If provided, the exporter
        computes a SHA-256 fingerprint of its contents to record in the
        artifact.  Defaults to the ``config_path`` specified in the
        dataset manifest if not supplied.

    dataset_version: Optional[str], optional
        User-provided version string (e.g., commit hash, experiment ID)
        identifying the dataset or model.  Included verbatim in the
        ``InferenceArtifact`` for traceability.

    extra: Optional[Dict[str, Any]], optional
        Additional arbitrary metadata to record.  Keys should be JSON
        serializable.  Useful for storing experiment notes, training
        parameters or debugging information.

    Returns
    -------
    InferenceArtifact
        The constructed artifact instance.  The artifact is also
        serialized to the specified ``output_path``.

    Raises
    ------
    ValueError
        If the dataset manifest is missing required fields.
    """
    meta = _load_meta(dataset_meta_path)
    # Determine config_path: preference order is explicit argument,
    # then value from manifest (for backward compatibility), else None.
    cfg_path = config_path or meta.get("config_path")
    config_fingerprint = compute_config_fingerprint(cfg_path)
    # Extract label_mode and horizon if present; for research datasets
    # the meta may record label parameters under entry_* keys.
    label_mode: Optional[str] = meta.get("entry_label_mode") or meta.get("label_mode")
    horizon = meta.get("entry_horizon_bars") or meta.get("horizon")
    # Compose artifact
    art = InferenceArtifact(
        model_path=model_path,
        features=list(features),
        truth_policy=str(meta["truth_policy"]),
        dataset_kind=str(meta["dataset_kind"]),
        config_path=str(cfg_path) if cfg_path else None,
        config_fingerprint=config_fingerprint,
        label_mode=label_mode,
        horizon=horizon,
        threshold=threshold,
        risk_profile=risk_profile,
        dataset_version=dataset_version,
        extra=extra or {},
    )
    # Serialize artifact to JSON
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(art.to_dict(), f, ensure_ascii=False, indent=2)
    return art
