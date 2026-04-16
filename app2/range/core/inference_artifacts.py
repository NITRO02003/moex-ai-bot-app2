from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class InferenceArtifact:
    """Serializable contract for offline->sandbox inference artifacts.

    The artifact captures the exact model path, ordered feature schema,
    truth policy, dataset kind, configuration provenance and threshold used
    by inference consumers.  This object is intentionally strict: sandbox or
    live consumers must fail closed if the artifact does not match expected
    inputs.
    """

    model_path: str
    features: List[str]
    truth_policy: str
    dataset_kind: str
    config_path: Optional[str]
    config_fingerprint: Optional[str]
    label_mode: Optional[str]
    horizon: Optional[int]
    threshold: Optional[float]
    risk_profile: str
    dataset_version: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, path: str | Path) -> "InferenceArtifact":
        src = Path(path)
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
