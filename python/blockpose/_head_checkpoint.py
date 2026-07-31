"""Shared directory-checkpoint format for frozen-backbone + trainable-head modules.

Used by both ``rgb_verifier.py`` and ``rgb_region_proposer.py``: a directory
containing one torch state-dict file for the trainable head only (the frozen
DINOv2 backbone is never serialized -- it is re-loaded from ``config.model_id``)
plus a ``config.json`` with the head's config fields and a schema version.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def write_checkpoint_config(path: Path, config: Any, schema_version: int, **extra: Any) -> None:
    payload = {"schema_version": schema_version, **extra, **asdict(config)}
    (Path(path) / "config.json").write_text(json.dumps(payload, indent=2) + "\n")


def read_checkpoint_config(
    path: Path, *, weights_filename: str, schema_version: int, checkpoint_kind: str
) -> dict:
    """Validate and return the raw config payload; caller builds the typed config."""
    path = Path(path)
    config_path, weights_path = path / "config.json", path / weights_filename
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"{path} is not a {checkpoint_kind} checkpoint (missing config/weights)"
        )
    payload = json.loads(config_path.read_text())
    if payload.get("schema_version") != schema_version:
        raise ValueError(
            f"unsupported {checkpoint_kind} checkpoint schema {payload.get('schema_version')!r}"
        )
    return payload


def config_from_payload(config_cls: type, payload: dict) -> Any:
    """Construct a frozen config dataclass from only the fields it declares."""
    fields = asdict(config_cls())
    return config_cls(**{field: payload[field] for field in fields if field in payload})
