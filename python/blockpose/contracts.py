"""Portable detector contracts for the Python-to-C++ ROS 2 boundary.

The runtime detector keeps its existing NumPy-rich API.  These immutable,
stdlib-only records define the small language-neutral wire contract used for
golden fixtures and the later C++ port.  Point clouds, images, and masks remain
out-of-band and are referred to by metadata identifiers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
TANGENT_ORDER = ("x", "y", "z", "rx", "ry", "rz")
CANONICAL_BLOCK_DIMENSIONS_M = (0.9, 0.6, 0.6)


class DetectorMode(str, Enum):
    SCENE_DISCOVERY = "SCENE_DISCOVERY"
    REFINE_BLOCK = "REFINE_BLOCK"
    REFINE_GRASPED = "REFINE_GRASPED"


MODE_TO_ENGINE_METHOD: Mapping[DetectorMode, str] = {
    DetectorMode.SCENE_DISCOVERY: "detect_free",
    DetectorMode.REFINE_BLOCK: "verify_placed",
    DetectorMode.REFINE_GRASPED: "estimate_gripped",
}


class ResultStatus(str, Enum):
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    DEVIATED = "deviated"
    MISSING = "missing"
    NOT_OBSERVABLE = "not_observable"


class Observability(str, Enum):
    WELL_CONSTRAINED = "well_constrained"
    PARTIALLY_CONSTRAINED = "partially_constrained"
    SYMMETRY_AMBIGUOUS = "symmetry_ambiguous"
    LAYOUT_INFERRED = "layout_inferred"
    UNOBSERVABLE = "unobservable"


class ErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    UNAVAILABLE_TRANSFORM = "unavailable_transform"
    EMPTY_ROI = "empty_roi"
    TIMEOUT = "timeout"
    INSUFFICIENT_OBSERVABILITY = "insufficient_observability"
    UNSUPPORTED_MODE = "unsupported_mode"
    INVALID_PRECONDITION = "invalid_precondition"


class EvidenceKind(str, Enum):
    GEOMETRY = "geometry"
    GEOMETRY_WITH_RGB_SUPPORT = "geometry_with_rgb_support"
    LAYOUT_INFERRED = "layout_inferred"
    PRIOR_ONLY = "prior_only"


class CovarianceKind(str, Enum):
    UNAVAILABLE = "unavailable"
    FIT_CURVATURE_UNCALIBRATED = "fit_curvature_uncalibrated"
    CALIBRATED = "calibrated"
    PRIOR = "prior"
    LAYOUT_PROPAGATED = "layout_propagated"


class ConfidenceKind(str, Enum):
    UNAVAILABLE = "unavailable"
    UNCALIBRATED_SCORE = "uncalibrated_score"
    CALIBRATED_VALIDITY = "calibrated_validity"


class PriorSource(str, Enum):
    PLAN = "plan"
    WORLD_MODEL = "world_model"
    FK_GRASP = "fk_grasp"


class EvidenceChannel(str, Enum):
    INTENSITY = "intensity"
    RELATIVE_TIME = "relative_time"
    BEAM_ID = "beam_id"
    SENSOR_TRAJECTORY = "sensor_trajectory"
    NO_RETURN_RAYS = "no_return_rays"
    GRIPPER_TRAJECTORY = "gripper_trajectory"
    RGB = "rgb"
    CAMERA_CALIBRATION = "camera_calibration"
    PER_POINT_SENSOR_ORIGINS = "per_point_sensor_origins"
    NO_RETURN_RAY_ORIGINS = "no_return_ray_origins"
    NO_RETURN_RELATIVE_TIME = "no_return_relative_time"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(values: Sequence[Any], size: int, name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, name) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: str | None, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _canonical_dimensions(values: Sequence[Any]) -> tuple[float, float, float]:
    dimensions = _vector(values, 3, "dimensions_m")
    if dimensions != CANONICAL_BLOCK_DIMENSIONS_M:
        raise ValueError(
            "dimensions_m must use canonical local axes (0.9, 0.6, 0.6); "
            "standing/laid orientation belongs in pose"
        )
    return dimensions


def _validate_psd(values: tuple[float, ...]) -> None:
    """Tolerant pivoted-free LDL decomposition for a symmetric PSD 6x6 matrix."""

    matrix = [[values[row * 6 + column] for column in range(6)] for row in range(6)]
    scale = max(1.0, max(abs(matrix[index][index]) for index in range(6)))
    tolerance = 1e-12 * scale
    lower = [[0.0] * 6 for _ in range(6)]
    diagonal = [0.0] * 6
    for row in range(6):
        pivot = matrix[row][row] - sum(
            lower[row][prior] ** 2 * diagonal[prior] for prior in range(row)
        )
        if pivot < -tolerance:
            raise ValueError("covariance must be positive semidefinite")
        diagonal[row] = max(pivot, 0.0)
        lower[row][row] = 1.0
        for following in range(row + 1, 6):
            residual = matrix[following][row] - sum(
                lower[following][prior] * lower[row][prior] * diagonal[prior]
                for prior in range(row)
            )
            if diagonal[row] <= tolerance:
                if abs(residual) > tolerance:
                    raise ValueError("covariance must be positive semidefinite")
                lower[following][row] = 0.0
            else:
                lower[following][row] = residual / diagonal[row]


def _extensions(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in result:
        if not isinstance(key, str) or "." not in key:
            raise ValueError("extension keys must be namespaced, e.g. 'blockpose.raw_score'")
    _canonical(result)
    return result


def normalize_unavailable_evidence(
    values: Sequence[str | EvidenceChannel],
) -> tuple[EvidenceChannel, ...]:
    """Map runtime ``unavailable_evidence`` strings without changing polarity."""

    return tuple(sorted({EvidenceChannel(value) for value in values}, key=lambda item: item.value))


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("contract JSON object keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize with stable keys/separators and reject all non-finite values."""

    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


@dataclass(frozen=True)
class PoseContract:
    frame_id: str
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id"))
        object.__setattr__(self, "position_m", _vector(self.position_m, 3, "position_m"))
        quaternion = _vector(self.quaternion_xyzw, 4, "quaternion_xyzw")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError("quaternion_xyzw must be normalized")
        quaternion = tuple(value / norm for value in quaternion)
        # q and -q are identical. Canonicalize the sign for byte-stable fixtures.
        first = next((value for value in reversed(quaternion) if abs(value) > 1e-15), 1.0)
        if first < 0.0:
            quaternion = tuple(-value for value in quaternion)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "position_m": self.position_m,
            "quaternion_xyzw": self.quaternion_xyzw,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PoseContract:
        return cls(value["frame_id"], tuple(value["position_m"]), tuple(value["quaternion_xyzw"]))


@dataclass(frozen=True)
class FrameContract:
    frame_id: str
    stamp_ns: int
    query_stamp_ns: int
    point_count: int
    cloud_reference: str
    unavailable_evidence: tuple[EvidenceChannel, ...] = ()
    image_reference: str | None = None
    calibration_id: str | None = None
    burst_stamps_ns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id"))
        object.__setattr__(self, "cloud_reference", _text(self.cloud_reference, "cloud_reference"))
        for name in ("stamp_ns", "query_stamp_ns", "point_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        evidence = normalize_unavailable_evidence(self.unavailable_evidence)
        object.__setattr__(self, "unavailable_evidence", evidence)
        burst = tuple(self.burst_stamps_ns)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in burst
        ):
            raise ValueError("burst_stamps_ns must contain non-negative integers")
        object.__setattr__(self, "burst_stamps_ns", burst)
        object.__setattr__(
            self,
            "image_reference",
            _optional_text(self.image_reference, "image_reference"),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _optional_text(self.calibration_id, "calibration_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "burst_stamps_ns": self.burst_stamps_ns,
            "calibration_id": self.calibration_id,
            "cloud_reference": self.cloud_reference,
            "frame_id": self.frame_id,
            "image_reference": self.image_reference,
            "point_count": self.point_count,
            "query_stamp_ns": self.query_stamp_ns,
            "stamp_ns": self.stamp_ns,
            "unavailable_evidence": self.unavailable_evidence,
        }


@dataclass(frozen=True)
class CovarianceContract:
    kind: CovarianceKind
    values_row_major: tuple[float, ...] | None
    frame_id: str | None = None

    def __post_init__(self) -> None:
        kind = CovarianceKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.values_row_major is None:
            if kind is not CovarianceKind.UNAVAILABLE:
                raise ValueError("available covariance kind requires values")
            if self.frame_id is not None:
                raise ValueError("unavailable covariance has no frame_id")
            return
        if kind is CovarianceKind.UNAVAILABLE:
            raise ValueError("unavailable covariance cannot contain values")
        values = _vector(self.values_row_major, 36, "covariance")
        for row in range(6):
            if values[row * 6 + row] < 0.0:
                raise ValueError("covariance diagonal must be non-negative")
            for column in range(row + 1, 6):
                if not math.isclose(
                    values[row * 6 + column], values[column * 6 + row], abs_tol=1e-10
                ):
                    raise ValueError("covariance must be symmetric")
        _validate_psd(values)
        object.__setattr__(self, "values_row_major", values)
        object.__setattr__(self, "frame_id", _text(self.frame_id, "covariance frame_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "kind": self.kind,
            "tangent_order": TANGENT_ORDER,
            "values_row_major": self.values_row_major,
        }


@dataclass(frozen=True)
class ConfidenceContract:
    kind: ConfidenceKind
    value: float | None
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        kind = ConfidenceKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.value is None:
            if kind is not ConfidenceKind.UNAVAILABLE or self.calibration_id is not None:
                raise ValueError("unavailable confidence has no value or calibration_id")
            return
        value = _finite(self.value, "confidence")
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if kind is ConfidenceKind.UNAVAILABLE:
            raise ValueError("unavailable confidence cannot contain a value")
        if kind is ConfidenceKind.CALIBRATED_VALIDITY and not self.calibration_id:
            raise ValueError("calibrated confidence requires calibration_id")
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "kind": self.kind,
            "value": self.value,
        }


@dataclass(frozen=True)
class BlockPriorContract:
    block_id: str
    source: PriorSource
    pose: PoseContract
    dimensions_m: tuple[float, float, float]
    covariance: CovarianceContract = field(
        default_factory=lambda: CovarianceContract(CovarianceKind.UNAVAILABLE, None)
    )
    plan_id: str | None = None
    course_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id"))
        object.__setattr__(self, "source", PriorSource(self.source))
        object.__setattr__(self, "dimensions_m", _canonical_dimensions(self.dimensions_m))
        object.__setattr__(self, "plan_id", _optional_text(self.plan_id, "plan_id"))
        object.__setattr__(self, "course_id", _optional_text(self.course_id, "course_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "course_id": self.course_id,
            "covariance": self.covariance,
            "dimensions_m": self.dimensions_m,
            "plan_id": self.plan_id,
            "pose": self.pose,
            "source": self.source,
        }


@dataclass(frozen=True)
class ImageProposalContract:
    proposal_id: str
    image_reference: str
    image_frame_id: str
    stamp_ns: int
    bbox_xywh_pixels: tuple[float, float, float, float]
    class_id: str
    score: float
    mask_reference: str | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("proposal_id", "image_reference", "image_frame_id", "class_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if (
            isinstance(self.stamp_ns, bool)
            or not isinstance(self.stamp_ns, int)
            or self.stamp_ns < 0
        ):
            raise ValueError("stamp_ns must be a non-negative integer")
        bbox = _vector(self.bbox_xywh_pixels, 4, "bbox_xywh_pixels")
        if bbox[0] < 0.0 or bbox[1] < 0.0 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise ValueError("bbox origin must be non-negative and extent positive")
        object.__setattr__(self, "bbox_xywh_pixels", bbox)
        score = _finite(self.score, "proposal score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("proposal score must lie in [0, 1]")
        object.__setattr__(self, "score", score)
        object.__setattr__(
            self,
            "mask_reference",
            _optional_text(self.mask_reference, "mask_reference"),
        )
        object.__setattr__(self, "model_id", _optional_text(self.model_id, "model_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xywh_pixels": self.bbox_xywh_pixels,
            "class_id": self.class_id,
            "image_frame_id": self.image_frame_id,
            "image_reference": self.image_reference,
            "mask_reference": self.mask_reference,
            "model_id": self.model_id,
            "proposal_id": self.proposal_id,
            "score": self.score,
            "stamp_ns": self.stamp_ns,
        }


@dataclass(frozen=True)
class EvidenceContract:
    kind: EvidenceKind
    support_points: int | None = None
    residual_rms_m: float | None = None
    free_space_rays: int | None = None
    visible_surface_fraction: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        for name in ("support_points", "free_space_rays"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.residual_rms_m is not None:
            residual = _finite(self.residual_rms_m, "residual_rms_m")
            if residual < 0.0:
                raise ValueError("residual_rms_m must be non-negative")
            object.__setattr__(self, "residual_rms_m", residual)
        if self.visible_surface_fraction is not None:
            fraction = _finite(self.visible_surface_fraction, "visible_surface_fraction")
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("visible_surface_fraction must lie in [0, 1]")
            object.__setattr__(self, "visible_surface_fraction", fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_space_rays": self.free_space_rays,
            "kind": self.kind,
            "residual_rms_m": self.residual_rms_m,
            "support_points": self.support_points,
            "visible_surface_fraction": self.visible_surface_fraction,
        }


@dataclass(frozen=True)
class BlockResultContract:
    block_id: str
    pose: PoseContract
    dimensions_m: tuple[float, float, float]
    status: ResultStatus
    observability: Observability
    evidence: EvidenceContract
    covariance: CovarianceContract
    confidence: ConfidenceContract
    unconstrained_directions: tuple[str, ...] = ()
    prior_translation_deviation_m: float | None = None
    prior_rotation_deviation_deg: float | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id"))
        object.__setattr__(self, "dimensions_m", _canonical_dimensions(self.dimensions_m))
        status = ResultStatus(self.status)
        observability = Observability(self.observability)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "observability", observability)
        allowed = set(TANGENT_ORDER)
        directions = tuple(self.unconstrained_directions)
        if len(set(directions)) != len(directions) or not set(directions) <= allowed:
            raise ValueError("unconstrained_directions must be unique SE(3) tangent axes")
        object.__setattr__(self, "unconstrained_directions", directions)
        for name in ("prior_translation_deviation_m", "prior_rotation_deviation_deg"):
            value = getattr(self, name)
            if value is not None:
                value = _finite(value, name)
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, value)
        if self.evidence.kind is EvidenceKind.PRIOR_ONLY:
            if status is not ResultStatus.NOT_OBSERVABLE:
                raise ValueError("prior-only evidence must be NotObservable")
            if observability is not Observability.UNOBSERVABLE:
                raise ValueError("prior-only evidence must be Unobservable")
            if self.covariance.kind not in {
                CovarianceKind.UNAVAILABLE,
                CovarianceKind.PRIOR,
            }:
                raise ValueError("prior-only evidence cannot carry measurement covariance")
            if self.confidence.kind is not ConfidenceKind.UNAVAILABLE:
                raise ValueError("prior-only evidence cannot carry measurement confidence")
        if status is ResultStatus.MISSING and self.evidence.free_space_rays in (None, 0):
            raise ValueError("Missing requires explicit free-space ray evidence")
        object.__setattr__(self, "extensions", _extensions(self.extensions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "confidence": self.confidence,
            "covariance": self.covariance,
            "dimensions_m": self.dimensions_m,
            "evidence": self.evidence,
            "extensions": self.extensions,
            "observability": self.observability,
            "pose": self.pose,
            "prior_rotation_deviation_deg": self.prior_rotation_deviation_deg,
            "prior_translation_deviation_m": self.prior_translation_deviation_m,
            "status": self.status,
            "unconstrained_directions": self.unconstrained_directions,
        }


@dataclass(frozen=True)
class DetectorRequestContract:
    query_id: str
    mode: DetectorMode
    frame: FrameContract
    priors: tuple[BlockPriorContract, ...] = ()
    image_proposals: tuple[ImageProposalContract, ...] = ()
    target_block_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        object.__setattr__(self, "query_id", _text(self.query_id, "query_id"))
        mode = DetectorMode(self.mode)
        object.__setattr__(self, "mode", mode)
        expected = 0 if mode is DetectorMode.SCENE_DISCOVERY else 1
        if len(self.priors) != expected:
            raise ValueError(f"{mode.value} requires exactly {expected} prior(s)")
        if mode is DetectorMode.REFINE_BLOCK:
            if not self.target_block_id or self.target_block_id != self.priors[0].block_id:
                raise ValueError("REFINE_BLOCK target_block_id must match its prior")
        if (
            mode is DetectorMode.REFINE_GRASPED
            and self.priors[0].source is not PriorSource.FK_GRASP
        ):
            raise ValueError("REFINE_GRASPED requires an FK_GRASP prior")
        if mode is DetectorMode.REFINE_GRASPED and self.target_block_id not in (
            None,
            self.priors[0].block_id,
        ):
            raise ValueError("REFINE_GRASPED target_block_id must match its FK/grasp prior")

    @property
    def engine_method(self) -> str:
        return MODE_TO_ENGINE_METHOD[self.mode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "image_proposals": self.image_proposals,
            "mode": self.mode,
            "priors": self.priors,
            "query_id": self.query_id,
            "schema_version": self.schema_version,
            "target_block_id": self.target_block_id,
        }

    def to_json(self) -> str:
        return canonical_json(self)


@dataclass(frozen=True)
class FailureContract:
    code: ErrorCode
    message: str
    stage: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ErrorCode(self.code))
        object.__setattr__(self, "message", _text(self.message, "message"))
        object.__setattr__(self, "stage", _optional_text(self.stage, "stage"))

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "stage": self.stage}


@dataclass(frozen=True)
class DetectorResponseContract:
    query_id: str
    mode: DetectorMode
    blocks: tuple[BlockResultContract, ...]
    failures: tuple[FailureContract, ...]
    runtime_ms: float
    target_block_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        object.__setattr__(self, "query_id", _text(self.query_id, "query_id"))
        object.__setattr__(self, "mode", DetectorMode(self.mode))
        object.__setattr__(
            self, "target_block_id", _optional_text(self.target_block_id, "target_block_id")
        )
        runtime = _finite(self.runtime_ms, "runtime_ms")
        if runtime < 0.0:
            raise ValueError("runtime_ms must be non-negative")
        object.__setattr__(self, "runtime_ms", runtime)
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("response block IDs must be unique")
        if self.failures and self.blocks:
            raise ValueError("failed response cannot contain blocks")
        if self.mode is DetectorMode.SCENE_DISCOVERY:
            if self.target_block_id is not None:
                raise ValueError("SCENE_DISCOVERY response cannot have target_block_id")
            if any(block.status is not ResultStatus.DETECTED for block in self.blocks):
                raise ValueError("SCENE_DISCOVERY may return only detected blocks")
        else:
            if self.target_block_id is None:
                raise ValueError("refinement response requires target_block_id")
            if not self.failures:
                if len(self.blocks) != 1 or self.blocks[0].block_id != self.target_block_id:
                    raise ValueError("successful refinement must return exactly its target block")
                if self.blocks[0].status is ResultStatus.DETECTED:
                    raise ValueError("refinement cannot return detected status")
                if (
                    self.mode is DetectorMode.REFINE_GRASPED
                    and self.blocks[0].status is ResultStatus.MISSING
                ):
                    raise ValueError("REFINE_GRASPED cannot return missing status")

    @property
    def success(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": self.blocks,
            "failures": self.failures,
            "mode": self.mode,
            "query_id": self.query_id,
            "runtime_ms": self.runtime_ms,
            "schema_version": self.schema_version,
            "success": self.success,
            "target_block_id": self.target_block_id,
        }

    def to_json(self) -> str:
        return canonical_json(self)
