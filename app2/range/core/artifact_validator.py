"""Validation utilities for inference artifacts.

This module provides helper functions to validate inference artifacts
against expected parameters.  The goal is to ensure that artifacts
consumed by inference pipelines match the expected schema, dataset
kind, truth policy, feature ordering and configuration fingerprint.

Validation is a critical safety measure to prevent subtle bugs where
the wrong model, dataset, configuration or feature set is used in a
production or sandbox environment.  A mismatch can lead to silent
degradation or financial losses.

Example::

    from .artifact_validator import validate_inference_artifact

    validate_inference_artifact(
        artifact_path="out/artifacts/entry_model_policy.json",
        expected_dataset_kind="entry",
        expected_truth_policy="trades",
        expected_features=["open", "close", ...],
        expected_config_path="app2/range/config.json",
    )

If validation passes, the function returns ``True``.  Otherwise it
raises a ``ValueError`` detailing the mismatch.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable, Optional

from .inference_artifacts import InferenceArtifact
from .artifact_exporter import compute_config_fingerprint

__all__ = ["validate_inference_artifact"]


def _load_artifact(path: str) -> InferenceArtifact:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"Artifact file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return InferenceArtifact(**data)


def validate_inference_artifact(
    artifact_path: str,
    *,
    expected_dataset_kind: Optional[str] = None,
    expected_truth_policy: Optional[str] = None,
    expected_features: Optional[Iterable[str]] = None,
    expected_config_path: Optional[str] = None,
) -> bool:
    """Validate an inference artifact against expected parameters.

    Parameters
    ----------
    artifact_path: str
        Path to the JSON file containing the ``InferenceArtifact``.

    expected_dataset_kind: Optional[str], optional
        Expected value of ``dataset_kind`` (e.g., ``"entry"`` or ``"intrade"``).
        If provided, the validator ensures that the artifact's
        ``dataset_kind`` matches exactly.

    expected_truth_policy: Optional[str], optional
        Expected value of ``truth_policy`` (e.g., ``"trades"`` or ``"candidates"``).
        If provided, the validator ensures a match.

    expected_features: Optional[Iterable[str]], optional
        Sequence of expected feature names.  If provided, the validator
        ensures that the artifact's ``features`` list matches exactly in
        both content and order.  Extra or missing features will trigger
        an error.

    expected_config_path: Optional[str], optional
        Path to the JSON configuration file that should have been used
        when creating the dataset.  If provided, its SHA-256 fingerprint
        is computed and compared against the artifact's
        ``config_fingerprint``.  If the artifact has no fingerprint, a
        mismatch is also raised.

    Returns
    -------
    bool
        ``True`` if validation succeeds.  Otherwise raises
        ``ValueError``.
    """
    art = _load_artifact(artifact_path)
    # Check dataset_kind
    if expected_dataset_kind is not None:
        if art.dataset_kind != expected_dataset_kind:
            raise ValueError(
                f"Artifact dataset_kind mismatch: expected {expected_dataset_kind}, got {art.dataset_kind}"
            )
    # Check truth_policy
    if expected_truth_policy is not None:
        if art.truth_policy != expected_truth_policy:
            raise ValueError(
                f"Artifact truth_policy mismatch: expected {expected_truth_policy}, got {art.truth_policy}"
            )
    # Check features
    if expected_features is not None:
        exp_list = list(expected_features)
        if art.features != exp_list:
            raise ValueError(
                f"Artifact features mismatch: expected {exp_list}, got {art.features}"
            )
    # Check config fingerprint
    if expected_config_path is not None:
        expected_fp = compute_config_fingerprint(expected_config_path)
        if art.config_fingerprint is None:
            raise ValueError(
                f"Artifact missing config_fingerprint; expected fingerprint of {expected_config_path}"
            )
        if art.config_fingerprint != expected_fp:
            raise ValueError(
                f"Artifact config_fingerprint mismatch: expected {expected_fp}, got {art.config_fingerprint}"
            )
    return True
