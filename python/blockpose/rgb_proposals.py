"""Replayable RGB instance proposals and conservative LiDAR association.

RGB is intentionally *proposal evidence* in this module.  It may select or
weight cloud points for a geometric fitter, but it never creates a metric pose,
covariance, verification state, or a ``Missing`` decision.  Those remain the
responsibility of the 3-D detector/verification pipeline.

The interfaces use plain arrays and JSON/NPZ caches so they can be replayed in
Python and implemented directly by the future C++ ROS 2 package.  SAM2 is an
optional provider and is not imported unless it is explicitly invoked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Protocol, Sequence

import numpy as np
import yaml
from scipy import ndimage

from blockpose.io import CameraModel, load_rgb
from blockpose.projection import project_points, world_to_camera
from blockpose.rgb_annotation import TransformersSam2Backend, load_candidate_masks


@dataclass(frozen=True)
class ImageProposal:
    """One RGB instance proposal with sufficient provenance for replay."""

    mask: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    class_label: str = "concrete_block"
    score: float = 1.0
    provider: str = "unknown"
    model_id: str | None = None
    detection_id: int | None = None
    timestamp_ns: int | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or not mask.size:
            raise ValueError("proposal mask must be a non-empty 2-D array")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.bbox_xywh):
            raise ValueError("bbox_xywh values must be integers")
        x, y, width, height = self.bbox_xywh
        if min(x, y) < 0 or width < 1 or height < 1:
            raise ValueError("bbox_xywh must be non-negative with positive extent")
        if x + width > mask.shape[1] or y + height > mask.shape[0]:
            raise ValueError("bbox_xywh must lie inside proposal mask")
        if not np.isfinite(self.score):
            raise ValueError("proposal score must be finite")
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class ProposalBatch:
    """Deterministic output of an :class:`ImageProposalProvider`."""

    proposals: tuple[ImageProposal, ...]
    provider: str
    image_shape: tuple[int, int]
    runtime_ms: float
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("image_shape must be (height, width)")
        if not np.isfinite(self.runtime_ms) or self.runtime_ms < 0:
            raise ValueError("runtime_ms must be finite and non-negative")
        for proposal in self.proposals:
            if proposal.mask.shape != self.image_shape:
                raise ValueError("all proposal masks must match batch image_shape")


class ImageProposalProvider(Protocol):
    """Provider contract; no provider gets ground truth or detector status."""

    def propose(
        self,
        rgb: np.ndarray | None,
        *,
        snapshot: Path | None = None,
        timestamp_ns: int | None = None,
    ) -> ProposalBatch: ...


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("proposal masks must contain at least one foreground pixel")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


class GeometryOnlyProposalProvider:
    """Explicit empty RGB path for the geometry-only baseline/fallback."""

    def propose(
        self,
        rgb: np.ndarray | None,
        *,
        snapshot: Path | None = None,
        timestamp_ns: int | None = None,
    ) -> ProposalBatch:
        shape = tuple(np.asarray(rgb).shape[:2]) if rgb is not None else (1, 1)
        return ProposalBatch((), "geometry_only", shape, 0.0, {"mode": "no_rgb"})


class StoredMaskProposalProvider:
    """Replay masks recorded by the previous segmentor, without model runtime.

    This is a deterministic offline ablation provider, not a production model.
    The caller must provide the snapshot folder; stored masks never use target
    annotations or registrations as prompts.
    """

    def propose(
        self,
        rgb: np.ndarray | None,
        *,
        snapshot: Path | None = None,
        timestamp_ns: int | None = None,
    ) -> ProposalBatch:
        if snapshot is None:
            raise ValueError("stored-mask replay requires snapshot")
        image = load_rgb(snapshot / "rgb.png") if rgb is None else np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        started = perf_counter_ns()
        masks = load_candidate_masks(snapshot, tuple(image.shape[:2]))
        proposals = tuple(
            ImageProposal(
                mask=mask,
                bbox_xywh=_bbox(mask),
                provider="stored_mask_replay",
                detection_id=detection_id,
                timestamp_ns=timestamp_ns,
                provenance={"snapshot": snapshot.name, "source": "metadata.candidate_masks"},
            )
            for detection_id, mask in sorted(masks.items())
            if np.any(mask)
        )
        return ProposalBatch(
            proposals,
            "stored_mask_replay",
            tuple(image.shape[:2]),
            (perf_counter_ns() - started) / 1e6,
            {"snapshot": snapshot.name, "replay": "candidate_masks"},
        )


class Sam2PointPromptProvider:
    """Optional SAM2 provider driven only by externally supplied image prompts.

    ``prompt_fn`` is deliberately required.  A caller can generate prompts from
    geometry or a projected prior, but this provider will not derive prompts
    from annotations, registered poses, or ground truth.
    """

    def __init__(
        self,
        prompt_fn: Callable[[np.ndarray], Sequence[tuple[int, int]]],
        *,
        backend: TransformersSam2Backend | None = None,
        model_id: str = "facebook/sam2.1-hiera-small",
        device: str = "auto",
    ) -> None:
        self.prompt_fn = prompt_fn
        self.backend = backend or TransformersSam2Backend(model_id=model_id, device=device)
        self.model_id = model_id
        self.device = device

    def propose(
        self,
        rgb: np.ndarray | None,
        *,
        snapshot: Path | None = None,
        timestamp_ns: int | None = None,
    ) -> ProposalBatch:
        if rgb is None:
            raise ValueError("SAM2 proposal generation requires RGB")
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        started = perf_counter_ns()
        proposals: list[ImageProposal] = []
        for index, prompt in enumerate(self.prompt_fn(image)):
            mask = np.asarray(self.backend.segment(image, prompt), dtype=bool)
            if mask.shape != image.shape[:2] or not np.any(mask):
                continue
            proposals.append(
                ImageProposal(
                    mask=mask,
                    bbox_xywh=_bbox(mask),
                    provider="sam2_point_prompt",
                    model_id=self.model_id,
                    detection_id=index,
                    timestamp_ns=timestamp_ns,
                    provenance={"prompt_u": str(prompt[0]), "prompt_v": str(prompt[1])},
                )
            )
        return ProposalBatch(
            tuple(proposals),
            "sam2_point_prompt",
            tuple(image.shape[:2]),
            (perf_counter_ns() - started) / 1e6,
            {"model_id": self.model_id, "device": self.device, "prompt_source": "external"},
        )


@dataclass(frozen=True)
class CacheProvenance:
    """Complete immutable identity of cached RGB support.

    The proposal cache is intentionally invalidated by calibration/association
    changes too.  That is conservative (those inputs do not alter a raw mask),
    but prevents an ablation from accidentally reusing a proposal artifact in a
    physically different experiment or final fold.
    """

    provider: str
    provider_config_sha256: str
    model_id: str | None
    model_revision: str | None
    model_weights_sha256: str | None
    prompt_config_sha256: str
    calibration_sha256: str
    extrinsics_sha256: str
    association_params_sha256: str
    asset_hashes: tuple[tuple[str, str], ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.provider or self.schema_version != 2:
            raise ValueError("unsupported or incomplete cache provenance")
        for value in (
            self.provider_config_sha256,
            self.prompt_config_sha256,
            self.calibration_sha256,
            self.extrinsics_sha256,
            self.association_params_sha256,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("provenance hashes must be lowercase SHA-256 hex")
        if tuple(sorted(self.asset_hashes)) != self.asset_hashes:
            raise ValueError("asset hashes must be sorted for deterministic replay")
        if self.model_weights_sha256 is not None and (
            len(self.model_weights_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.model_weights_sha256)
        ):
            raise ValueError("model_weights_sha256 must be lowercase SHA-256 hex")
        if self.provider.startswith("sam") and (
            not self.model_id or not self.model_revision or not self.model_weights_sha256
        ):
            raise ValueError(
                "SAM proposal caching requires resolved model_id, model_revision, and weights hash"
            )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _declared_snapshot_assets(snapshot: Path) -> tuple[tuple[str, str], ...]:
    """Hash RGB, metadata, and only safe mask paths declared in metadata."""
    root = snapshot.resolve()
    paths: set[Path] = set()
    for relative in ("rgb.png", "metadata.yaml"):
        candidate = snapshot / relative
        if candidate.is_file():
            paths.add(candidate.resolve())
    metadata_path = snapshot / "metadata.yaml"
    if metadata_path.is_file():
        try:
            metadata = yaml.safe_load(metadata_path.read_text()) or {}
            for entry in metadata.get("candidate_masks", []):
                candidate = (snapshot / str(entry["file"])).resolve()
                candidate.relative_to(root)
                if candidate.is_file():
                    paths.add(candidate)
        except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
            pass
    return tuple(
        sorted((str(path.relative_to(root)), _sha256_bytes(path.read_bytes())) for path in paths)
    )


def make_cache_provenance(
    snapshot: Path,
    *,
    provider: str,
    provider_config: object | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    model_weights_sha256: str | None = None,
    prompt_config: object | None = None,
    calibration: object | None = None,
    extrinsics: object | None = None,
    association_params: MaskAssociationParams | None = None,
) -> CacheProvenance:
    """Build the full cache manifest; callers must supply their actual configs."""
    return CacheProvenance(
        provider=provider,
        provider_config_sha256=_sha256_value(provider_config),
        model_id=model_id,
        model_revision=model_revision,
        model_weights_sha256=model_weights_sha256,
        prompt_config_sha256=_sha256_value(prompt_config),
        calibration_sha256=_sha256_value(calibration),
        extrinsics_sha256=_sha256_value(extrinsics),
        association_params_sha256=_sha256_value(
            asdict(association_params) if association_params is not None else None
        ),
        asset_hashes=_declared_snapshot_assets(snapshot),
    )


def proposal_cache_key(
    snapshot: Path,
    provider: str,
    version: str = "v2",
    *,
    provenance: CacheProvenance | None = None,
) -> str:
    """Stable cache key; the legacy shorthand has no physics/config identity.

    New callers must pass ``provenance``.  The positional parameters are kept
    only to provide a clear error to old experiments rather than silently
    creating an unsafe cache.
    """
    if provenance is None:
        raise ValueError("proposal caches require full CacheProvenance")
    if provider != provenance.provider or version != "v2":
        raise ValueError("cache key provider/version does not match provenance")
    return _sha256_value({"version": version, "provenance": asdict(provenance)})


def save_proposal_cache(
    path: Path, batch: ProposalBatch, *, cache_key: str, provenance: CacheProvenance
) -> None:
    """Write deterministic replay data as adjacent JSON metadata and NPZ masks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = {f"mask_{index:04d}": item.mask for index, item in enumerate(batch.proposals)}
    np.savez_compressed(path.with_suffix(".npz"), **masks)
    records = []
    for index, proposal in enumerate(batch.proposals):
        record = asdict(proposal)
        record.pop("mask")
        record["mask_key"] = f"mask_{index:04d}"
        records.append(record)
    payload = {
        "schema_version": 1,
        "cache_key": cache_key,
        "cache_provenance": asdict(provenance),
        "provider": batch.provider,
        "image_shape": list(batch.image_shape),
        "runtime_ms": batch.runtime_ms,
        "provenance": batch.provenance,
        "proposals": records,
    }
    path.with_suffix(".json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def load_proposal_cache(
    path: Path, *, expected_key: str, expected_provenance: CacheProvenance
) -> ProposalBatch | None:
    """Load only an exact, schema-compatible cache; stale data is ignored."""
    metadata_path, masks_path = path.with_suffix(".json"), path.with_suffix(".npz")
    if not metadata_path.is_file() or not masks_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text())
        if payload.get("schema_version") != 1 or payload.get("cache_key") != expected_key:
            return None
        stored_provenance = CacheProvenance(
            **{
                **payload["cache_provenance"],
                "asset_hashes": tuple(
                    tuple(item) for item in payload["cache_provenance"]["asset_hashes"]
                ),
            }
        )
        if stored_provenance != expected_provenance:
            return None
        with np.load(masks_path, allow_pickle=False) as loaded:
            proposals = tuple(
                ImageProposal(
                    mask=np.asarray(loaded[raw.pop("mask_key")], dtype=bool),
                    **{**raw, "bbox_xywh": tuple(raw["bbox_xywh"])}
                )
                for item in payload["proposals"]
                for raw in [dict(item)]
            )
        return ProposalBatch(
            proposals,
            str(payload["provider"]),
            tuple(int(value) for value in payload["image_shape"]),
            float(payload["runtime_ms"]),
            {str(key): str(value) for key, value in payload.get("provenance", {}).items()},
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class MaskAssociationParams:
    """Image-space association margins, including calibration uncertainty."""

    dilation_px: int = 4
    calibration_sigma_px: float = 2.0
    calibration_sigma_multiplier: float = 2.0
    soft_margin_px: float = 12.0
    min_weight: float = 1e-3

    @property
    def effective_dilation_px(self) -> int:
        values = (
            self.dilation_px,
            self.calibration_sigma_px,
            self.calibration_sigma_multiplier,
            self.soft_margin_px,
            self.min_weight,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("association parameters must be finite")
        if self.dilation_px < 0 or self.calibration_sigma_px < 0 or self.soft_margin_px < 0:
            raise ValueError("association margins must be non-negative")
        if self.calibration_sigma_multiplier < 0 or self.min_weight < 0:
            raise ValueError("association parameters must be non-negative")
        uncertainty = self.calibration_sigma_px * self.calibration_sigma_multiplier
        return self.dilation_px + int(np.ceil(uncertainty))


@dataclass(frozen=True)
class MaskAssociation:
    """Per-point RGB support; callers must still run geometric fitting."""

    hard_indices: np.ndarray
    soft_indices: np.ndarray
    soft_weights: np.ndarray
    projected: np.ndarray
    effective_mask: np.ndarray
    calibration_margin_px: int
    reason: str = "ok"


def _validate_world_camera_transform(T_world_camera: np.ndarray) -> np.ndarray:
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_world_camera must be a finite (4, 4) transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T_world_camera must have homogeneous final row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("T_world_camera rotation must be a proper orthonormal rotation")
    return transform


def associate_mask_to_lidar(
    points_world: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    proposal: ImageProposal,
    params: MaskAssociationParams | None = None,
) -> MaskAssociation:
    """Project LiDAR points and associate a mask with calibration-aware slack.

    Hard selection includes the dilated mask. Soft selection extends farther and
    assigns a linear image-distance weight in ``(0, 1]``.  It has no opinion on
    whether an object exists; an empty result is merely unavailable RGB support.
    """
    params = params or MaskAssociationParams()
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_world must be a finite (N, 3) array")
    if proposal.mask.shape != (camera.height, camera.width):
        raise ValueError("proposal mask shape does not match camera model")
    transform = _validate_world_camera_transform(T_world_camera)
    camera_points = world_to_camera(points, transform)
    uv, in_front = project_points(camera_points, camera)
    u, v = np.rint(uv[:, 0]).astype(np.int64), np.rint(uv[:, 1]).astype(np.int64)
    projected = in_front & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    dilation = params.effective_dilation_px
    expanded = (
        ndimage.binary_dilation(proposal.mask, iterations=dilation)
        if dilation
        else proposal.mask
    )
    hard = np.zeros(len(points), dtype=bool)
    valid = np.flatnonzero(projected)
    hard[valid] = expanded[v[valid], u[valid]]
    # distance to accepted support (zero inside) is robust to overlapping masks;
    # each proposal keeps independent weights and is never assigned a target ID.
    distance = ndimage.distance_transform_edt(~expanded)
    weights = np.zeros(len(points), dtype=np.float64)
    if params.soft_margin_px == 0:
        weights[hard] = 1.0
    else:
        sampled = distance[v[valid], u[valid]]
        weights[valid] = np.clip(1.0 - sampled / params.soft_margin_px, 0.0, 1.0)
    soft = weights >= params.min_weight
    return MaskAssociation(
        np.flatnonzero(hard),
        np.flatnonzero(soft),
        weights[soft],
        projected,
        expanded,
        dilation,
    )


def proposal_point_indices(
    geometry_indices: np.ndarray,
    association: MaskAssociation | None,
    mode: str,
) -> np.ndarray:
    """Select input points for ablation without allowing RGB to remove fallback.

    ``geometry`` uses only geometric points. ``hard`` and ``soft`` use RGB only
    when it is available.  ``union`` is the production-safe form: it includes
    all geometric points plus soft RGB support, hence an image dropout has the
    same result as geometry-only.
    """
    geometry = np.unique(np.asarray(geometry_indices, dtype=int))
    if mode == "geometry" or association is None:
        return geometry
    rgb = association.hard_indices if mode == "hard" else association.soft_indices
    if mode in {"hard", "soft"}:
        return np.unique(rgb)
    if mode == "union":
        return np.union1d(geometry, association.soft_indices)
    raise ValueError("mode must be geometry, hard, soft, or union")
