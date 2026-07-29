"""Build a labelled candidate set for one annotated block.

V1 is a candidate-set ranking task: for a labelled block, real PCL candidates
plus controlled perturbations are labelled ``positive`` / ``negative`` /
``ignore``.  Orientation labelling goes through ``rotation_error_deg``, which is
symmetry-aware, so a symmetry-equivalent orientation is never mislabelled as a
hard negative (design doc: "Yaw equivalence must respect the block's geometric
symmetries").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from blockpose.evaluate import rotation_error_deg, translation_error
from blockpose.pose import BlockPose

# Heavily-occluded / not-RGB-judgeable annotations become ``ignore`` samples.
# Values match the real annotation vocabulary (full / partial / heavy_occlusion
# / unknown); ``partial`` stays judgeable, so only the last two are ignored.
_OCCLUDED_VISIBILITIES = ("heavy_occlusion", "unknown")


@dataclass(frozen=True)
class CandidateSetParams:
    """Tolerances and perturbation magnitudes for building a candidate set."""

    candidate_radius_m: float = 0.50
    positive_pos_tol_m: float = 0.25
    positive_rot_tol_deg: float = 30.0
    num_positive_perturbations: int = 4
    positive_perturb_pos_m: float = 0.05
    positive_perturb_rot_deg: float = 8.0
    yaw_negative_deg: tuple[float, ...] = (90.0, 270.0)
    position_negative_m: tuple[float, ...] = (0.35, 0.6)
    tilt_negative_deg: tuple[float, ...] = (25.0, 45.0)
    occluded_visibilities: tuple[str, ...] = _OCCLUDED_VISIBILITIES


@dataclass(frozen=True)
class LabeledCandidate:
    """One candidate pose with its label and where it came from."""

    candidate_id: str
    pose: BlockPose
    label: str  # "positive" | "negative" | "ignore"
    provenance: str
    details: dict[str, float] = field(default_factory=dict)


def _rot_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _with_pose(pose: BlockPose, position: np.ndarray, rotation: np.ndarray) -> BlockPose:
    return BlockPose(position=position, rotation=rotation, dims=pose.dims)


def _is_positive(pose: BlockPose, gt: BlockPose, params: CandidateSetParams) -> bool:
    return (
        translation_error(pose, gt) <= params.positive_pos_tol_m
        and rotation_error_deg(pose, gt) <= params.positive_rot_tol_deg
    )


def build_candidate_set(
    target_id: str,
    gt: BlockPose,
    visibility: str,
    raw_hypotheses: list[tuple[str, BlockPose]],
    params: CandidateSetParams,
    rng: np.random.Generator,
) -> list[LabeledCandidate]:
    """Assemble positives, hard negatives, and (when occluded) an ignore sample.

    ``raw_hypotheses`` are ``(id, pose)`` from the geometry detector's trace; only
    those within ``candidate_radius_m`` of the block are considered.
    """
    if visibility in params.occluded_visibilities:
        return [LabeledCandidate(f"{target_id}/ignore", gt, "ignore", "occluded_annotation")]

    candidates: list[LabeledCandidate] = [
        LabeledCandidate(f"{target_id}/gt", gt, "positive", "ground_truth")
    ]

    # Positive augmentations: small perturbations that stay within tolerance.
    for i in range(params.num_positive_perturbations):
        position = gt.position + rng.normal(0.0, params.positive_perturb_pos_m, size=3)
        axis = rng.normal(size=3)
        rotation = (
            _rot_about_axis(axis, rng.normal(0.0, params.positive_perturb_rot_deg)) @ gt.rotation
        )
        pose = _with_pose(gt, position, rotation)
        if _is_positive(pose, gt, params):
            candidates.append(
                LabeledCandidate(
                    f"{target_id}/pos_perturb_{i}", pose, "positive", "positive_perturbation"
                )
            )

    # Real PCL siblings: hard negatives (or extra positives) drawn from geometry.
    for hyp_id, pose in raw_hypotheses:
        if translation_error(pose, gt) > params.candidate_radius_m:
            continue
        positive = _is_positive(pose, gt, params)
        candidates.append(
            LabeledCandidate(
                f"{target_id}/raw_{hyp_id}",
                pose,
                "positive" if positive else "negative",
                "raw_hypothesis",
                {
                    "translation_error_m": translation_error(pose, gt),
                    "rotation_error_deg": rotation_error_deg(pose, gt),
                },
            )
        )

    # Controlled yaw-sibling negatives about world Z (kept only when the
    # symmetry-aware error confirms they are genuinely wrong orientations).
    world_z = np.array([0.0, 0.0, 1.0])
    for angle in params.yaw_negative_deg:
        pose = _with_pose(gt, gt.position, _rot_about_axis(world_z, angle) @ gt.rotation)
        if not _is_positive(pose, gt, params):
            candidates.append(
                LabeledCandidate(
                    f"{target_id}/yaw_{int(angle)}",
                    pose,
                    "negative",
                    "yaw_sibling",
                    {"rotation_error_deg": rotation_error_deg(pose, gt)},
                )
            )

    # Controlled position negatives along the block's long axis.
    long_axis = gt.rotation[:, 0]
    for offset in params.position_negative_m:
        pose = _with_pose(gt, gt.position + offset * long_axis, gt.rotation)
        if not _is_positive(pose, gt, params):
            candidates.append(
                LabeledCandidate(
                    f"{target_id}/shift_{offset:g}",
                    pose,
                    "negative",
                    "position_perturbation",
                    {"translation_error_m": translation_error(pose, gt)},
                )
            )

    # Controlled roll/pitch (tilt) negatives about the block's short axis.
    short_axis = gt.rotation[:, 1]
    for angle in params.tilt_negative_deg:
        pose = _with_pose(gt, gt.position, _rot_about_axis(short_axis, angle) @ gt.rotation)
        if not _is_positive(pose, gt, params):
            candidates.append(
                LabeledCandidate(
                    f"{target_id}/tilt_{int(angle)}",
                    pose,
                    "negative",
                    "tilt_perturbation",
                    {"rotation_error_deg": rotation_error_deg(pose, gt)},
                )
            )

    return candidates
