#!/usr/bin/env python3
"""Measure `dino_verifier` in the role it was actually trained for: within-block pose
disambiguation, not cross-candidate reranking.

Context (`docs/production-readiness-plan.md`,
`experiments/dino_proposer_pooled_ablation.md`): `experiments/rgb_candidate_prior`
trains `dino_verifier` on candidate sets built by
`experiments.rgb_candidate_prior.candidates.build_candidate_set` --
positives = ground truth plus small perturbations (<=0.05 m, <=8 deg,
`CandidateSetParams.positive_perturb_pos_m`/`positive_perturb_rot_deg`), hard
negatives = the SAME block yawed 90/270 deg about world Z, shifted
0.35/0.6 m, or tilted 25/45 deg
(`CandidateSetParams.yaw_negative_deg`/`position_negative_m`/`tilt_negative_deg`).
That is a WITHIN-BLOCK "does this projected outline align with the block in
the RGB" task -- every candidate in one training group is a pose hypothesis
for the SAME physical block, sharing one crop. It was never trained to compare
two DIFFERENT blocks' candidates against each other, which is exactly why
`tools/eval_dino_proposer_pooled.py`'s cross-candidate additive-scalar rerank
measured a near-uniform, non-discriminating shift (see that script's report).
`dino_verifier`'s own held-out top-1 accuracy on the within-block task is
93.6% (`experiments/rgb_candidate_prior/report.py` / training manifest).

This script tests that task directly, end to end on the real detector output:

  1. Run the whole-cloud PCL detector (`annotation_candidates._run_detection`,
     the same entry point `tools/eval_dino_proposer_pooled.py` uses) to get
     accepted ("fate == final") candidates for each of the 25 annotated
     snapshots.
  2. For each accepted candidate, build a POSE HYPOTHESIS SET: the candidate
     as-is plus yaw rotations of 90/180/270 deg about the candidate's OWN up
     axis (the body axis nearest world +Z) through its own position. This
     mirrors `candidates.py`'s hard-negative construction
     (`_rot_about_axis(world_z, angle) @ gt.rotation`, `yaw_negative_deg =
     (90, 270)`), so a hypothesis here looks like precisely what the model
     was taught to reject; the training targets are annotated blocks resting
     near-level, where their own up axis and world +Z coincide to well inside
     the +-8 deg `positive_perturb_rot_deg` the model was trained to still
     accept as a positive.
     A rotation that is a genuine symmetry of the block's own extents
     (`blockpose.pose.symmetry_equivalents` / `_cuboid_symmetry_group`) is
     dropped: it is not a distinguishable hypothesis, it is the same physical
     placement relabelled. 180 deg about a box's own vertical axis is a
     symmetry for ANY rectangular cuboid (the "flip end for end" element
     every box's symmetry group contains, regardless of which extents are
     equal), and since 270 deg = 180 deg composed with 90 deg about the same
     axis, 270 deg then duplicates 90 deg too. So the distinguishable
     hypothesis set for a flat-resting candidate is just {0 deg, 90 deg} --
     two hypotheses, not four.

     Using world +Z as the yaw axis instead, as an earlier revision of this
     script did, makes that de-duplication silently fail: 180 deg about world
     +Z differs from the block's true symmetry by roughly twice the
     candidate's tilt, and the accepted candidates on this dataset have a
     median tilt of 1.70 deg from vertical (measured over the 59 accepted
     candidates across all 25 snapshots; only 5% are under 0.5 deg). All four
     hypotheses then survive as two near-degenerate PAIRS, which quietly
     turns the intended head-to-head `base` vs `yaw90` argmax into a
     best-of-two-per-side comparison -- a different, noisier estimator than
     the one being reported. Rotating about the body axis makes the symmetry
     exact, so the dedup is a hard algebraic identity rather than something
     that has to clear an angular tolerance.
  3. Render every hypothesis' RGB crop + conditioning image
     (`experiments.rgb_candidate_prior.render`), with all hypotheses for ONE
     block sharing ONE crop bbox (`render.union_crop_bbox`). This is the
     shared-crop rule `render.py`'s own module docstring states exists so
     ranking cannot leak pose through crop framing -- getting this wrong
     (giving each hypothesis its own crop, like the cross-candidate reranker
     does) would let the model win just by getting a tighter/looser box, not
     by reading which projected outline actually matches the pixels.
  4. Score every hypothesis with `dino_verifier`
     (`score_candidate_with_dino` -> `DinoVerifierHead.forward`), take the
     ARGMAX by calibrated score, and use that hypothesis's pose as the
     corrected candidate pose. Ties / all-abstained hypothesis sets keep the
     original (uncorrected) pose.
  5. Score recall before vs. after correction with this repo's standard
     metrics: `blockpose.evaluate.match_detections` at 0.30 m for centre
     recall, `blockpose.experiment_record.strict_match` for strict
     (symmetry-aware rotation + oriented-box IoU) recall -- and report
     gained/broken PER BLOCK, not just the aggregate. The prior audit's
     failure mode was an aggregate hiding 9 regressions behind 1 gain; this
     script's whole point is to make that impossible to hide.

**Addressable pool.** Only candidates that are already correct at the centre
level but wrong at the strict level can possibly be fixed by disambiguation
-- pose correction cannot manufacture a detection `match_detections` never
made in the first place, and it cannot fix a genuinely mis-fit candidate that
strict-fails on IoU alone. That pool is measured, not assumed, and reported
explicitly; see the "Addressable pool" section of the generated report. It is
small (on the order of 2-3 blocks across all 25 snapshots), so a +1/+2 result
on 62 GT blocks is explicitly flagged as being inside this dataset's noise.

Usage:
    uv run python tools/eval_verifier_pose_disambiguation.py \
        --report-out experiments/verifier_pose_disambiguation.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.annotation_candidates import _run_detection  # noqa: E402
from blockpose.annotation_tool import AnnotationDocument, EditableTarget  # noqa: E402
from blockpose.core_types import DetectionParameters  # noqa: E402
from blockpose.evaluate import match_detections  # noqa: E402
from blockpose.experiment_record import (  # noqa: E402
    DEFAULT_MAX_ROTATION_DEG,
    DEFAULT_MIN_IOU,
    experiment_stamp,
    strict_match,
)
from blockpose.io import (  # noqa: E402
    CameraModel,
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_tf_snapshot,
)
from blockpose.pose import BlockPose, symmetry_equivalents  # noqa: E402
from blockpose.rgb_verifier import load_dino_verifier, score_candidate_with_dino  # noqa: E402

from experiments.rgb_candidate_prior import render  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "seibersdorf_dataset"
DEFAULT_MANIFEST = REPO_ROOT / "config" / "annotated_snapshots_seibersdorf.yaml"
DEFAULT_TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
DEFAULT_VERIFIER_CHECKPOINT = REPO_ROOT / "models" / "production" / "dino_verifier"

# Mirrors `candidates.py`'s hard-negative construction (see module docstring):
# the same block yawed 90/180/270 deg about its own up axis through its own
# centre. All three are passed through `generate_pose_hypotheses`, which drops
# whichever ones turn out to be symmetry-equivalent to a hypothesis already
# kept (`is_symmetry_duplicate`) rather than assuming which will be -- 180 deg
# about a block's own vertical axis is always a symmetry of a rectangular
# cuboid, and 270 deg is then always equivalent to 90 deg (see module
# docstring), so this yields {0 deg, 90 deg} for a laid block. 180 deg is
# included even though `candidates.py` never trains on it precisely so that
# the dedup is measured rather than assumed.
DEFAULT_YAW_HYPOTHESES_DEG: tuple[float, ...] = (90.0, 180.0, 270.0)


# --------------------------------------------------------------------------
# Pose hypothesis generation (pure, GPU-free -- see tests/python/
# test_verifier_pose_disambiguation.py)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseHypothesis:
    """One candidate pose in a within-block disambiguation set."""

    id: str
    pose: BlockPose
    yaw_deg: float  # 0.0 for the untouched candidate


def body_up_axis(rotation: np.ndarray) -> np.ndarray:
    """The pose's own up axis: its body axis nearest world +Z, signed upwards.

    Yawing about this axis rather than about world +Z is what makes
    `is_symmetry_duplicate` exact. 180 deg about a body axis is an element of
    every rectangular cuboid's symmetry group (`_cuboid_symmetry_group`
    contains `diag(-1, -1, 1)` for all extents), so the duplicate is detected
    at 0 deg error. About world +Z the same rotation misses the true symmetry
    by roughly twice the candidate's tilt, which on this dataset's accepted
    candidates is a median 1.70 deg -- enough to defeat any sane tolerance and
    let a symmetry-equivalent pose masquerade as a distinct hypothesis.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    axis = rotation[:, int(np.argmax(np.abs(rotation[2, :])))]
    return -axis if axis[2] < 0.0 else axis


def _rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotation matrix for a right-handed rotation of `angle_deg` about `axis`."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (deg) of the rotation that takes orientation `a` to orientation `b`."""
    dR = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    trace_val = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace_val)))


def is_symmetry_duplicate(
    rotation: np.ndarray,
    dims: tuple[float, float, float],
    existing_rotations: list[np.ndarray],
    *,
    atol_deg: float = 1.0,
) -> bool:
    """True when `rotation` describes the same physical placement as one already kept.

    Compares every symmetry-equivalent orientation of `rotation`
    (`blockpose.pose.symmetry_equivalents`, derived from `dims` via
    `_cuboid_symmetry_group` -- not a fixed assumed group, see that function's
    docstring) against each rotation already in the hypothesis set. A match
    within `atol_deg` means the candidate rotation is not a new,
    distinguishable hypothesis -- it is the same box placement relabelled --
    and must not be counted as a separate pose the verifier could ever
    legitimately be asked to tell apart from the one it duplicates.
    """
    equivalents = symmetry_equivalents(rotation, dims)
    for existing in existing_rotations:
        if any(_rotation_angle_deg(eq, existing) <= atol_deg for eq in equivalents):
            return True
    return False


def generate_pose_hypotheses(
    pose: BlockPose,
    yaw_angles_deg: tuple[float, ...] = DEFAULT_YAW_HYPOTHESES_DEG,
    *,
    dedup_atol_deg: float = 1.0,
) -> list[PoseHypothesis]:
    """The candidate pose plus its distinguishable yaw siblings.

    Always includes the untouched pose first (`id="base"`, `yaw_deg=0.0`).
    Each `yaw_angles_deg` entry is rotated about the candidate's OWN up axis
    (`body_up_axis`) through its own position (position is unchanged --
    rotating a rigid body about an axis through its own centre only changes
    orientation) and kept only when `is_symmetry_duplicate` says it is NOT the
    same physical placement as a hypothesis already in the set. 180 deg about
    a block's own vertical axis is always a symmetry of a rectangular cuboid,
    and 270 deg is then always equivalent to the kept 90 deg hypothesis (see
    module docstring), so this yields {base, yaw90} -- 2 distinguishable
    hypotheses per laid candidate, not 4, and it does so for a tilted
    candidate exactly as it does for a perfectly level one. Yawing about
    world +Z instead would leave both duplicates in the set for any candidate
    tilted more than ~0.5 deg, which is 95% of this dataset's.
    """
    hypotheses = [PoseHypothesis("base", pose, 0.0)]
    kept_rotations = [pose.rotation]
    up = body_up_axis(pose.rotation)
    for yaw in yaw_angles_deg:
        rotation = _rotation_about_axis(up, yaw) @ pose.rotation
        if is_symmetry_duplicate(rotation, pose.dims, kept_rotations, atol_deg=dedup_atol_deg):
            continue
        rotated_pose = BlockPose(position=pose.position.copy(), rotation=rotation, dims=pose.dims)
        hypotheses.append(PoseHypothesis(f"yaw{int(round(yaw))}", rotated_pose, float(yaw)))
        kept_rotations.append(rotation)
    return hypotheses


# --------------------------------------------------------------------------
# Rendering + scoring (needs torch/DINOv2 + the annotated dataset)
# --------------------------------------------------------------------------


def render_hypothesis_crops(
    hypotheses: list[PoseHypothesis],
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    *,
    crop_margin_frac: float,
    out_size: int,
) -> dict[str, tuple[np.ndarray | None, np.ndarray | None]]:
    """`{hypothesis.id: (rgb_crop, conditioning)}`, all sharing ONE union crop bbox.

    All hypotheses in `hypotheses` are pose hypotheses for the SAME physical
    block, so they share one crop (`render.union_crop_bbox`) -- exactly how
    `experiments/rgb_candidate_prior/export.py` builds training crops for one
    candidate group. Giving each hypothesis its own, independently-fit crop
    would let the verifier win by framing rather than by reading which
    projected outline matches the pixels, invalidating the comparison (see
    module docstring and `render.py`'s own docstring on this rule).
    """
    per_hypothesis_bbox: dict[str, tuple[int, int, int, int] | None] = {}
    for hyp in hypotheses:
        _, uv, in_front = render.project_corners(hyp.pose, T_world_camera, camera)
        per_hypothesis_bbox[hyp.id] = render.crop_bbox(
            uv, in_front, camera, margin_frac=crop_margin_frac
        )
    valid_bboxes = [b for b in per_hypothesis_bbox.values() if b is not None]
    shared_bbox = render.union_crop_bbox(valid_bboxes, camera)

    result: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    if shared_bbox is None:
        return {hyp.id: (None, None) for hyp in hypotheses}
    rgb_crop = render.render_crop(rgb, shared_bbox, out_size=out_size)
    for hyp in hypotheses:
        conditioning = render.render_conditioning(
            hyp.pose, T_world_camera, camera, shared_bbox, out_size=out_size
        )
        result[hyp.id] = (rgb_crop, conditioning)
    return result


def disambiguate_candidate(
    pose: BlockPose,
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    verifier_head,
    verifier_config,
    verifier_temperature: float,
    *,
    yaw_angles_deg: tuple[float, ...],
    crop_margin_frac: float,
    out_size: int,
) -> tuple[BlockPose, dict]:
    """Argmax-by-calibrated-score correction of one candidate's pose.

    Returns `(corrected_pose, diagnostics)`. Falls back to the original
    `pose` unchanged whenever every hypothesis abstains (no crop, no model,
    or high uncertainty) -- correction must never invent a pose the verifier
    had no real evidence for.
    """
    hypotheses = generate_pose_hypotheses(pose, yaw_angles_deg)
    crops = render_hypothesis_crops(
        hypotheses,
        rgb,
        T_world_camera,
        camera,
        crop_margin_frac=crop_margin_frac,
        out_size=out_size,
    )
    scored = []
    for hyp in hypotheses:
        rgb_crop, conditioning = crops[hyp.id]
        result = score_candidate_with_dino(
            verifier_head,
            verifier_config,
            rgb_crop,
            conditioning,
            calibration_temperature=verifier_temperature,
        )
        scored.append((hyp, result))

    usable = [(hyp, result) for hyp, result in scored if not result.abstained]
    diagnostics = {
        "num_hypotheses": len(hypotheses),
        "num_usable": len(usable),
        "chosen_yaw_deg": 0.0,
        "corrected": False,
    }
    if not usable:
        return pose, diagnostics

    best_hyp, best_result = max(usable, key=lambda item: item[1].calibrated_score)
    diagnostics["chosen_yaw_deg"] = best_hyp.yaw_deg
    diagnostics["corrected"] = best_hyp.yaw_deg != 0.0
    diagnostics["best_calibrated_score"] = best_result.calibrated_score
    return best_hyp.pose, diagnostics


# --------------------------------------------------------------------------
# Snapshot manifest + GT loading (same conventions as eval_dino_proposer_pooled.py)
# --------------------------------------------------------------------------


def load_manifest_snapshots(manifest_path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(manifest_path.read_text()) or {}
    result = []
    for scene, info in (data.get("scenes") or {}).items():
        for entry in info.get("annotated") or []:
            result.append((scene, str(entry["name"])))
    return result


def approved_targets(doc: AnnotationDocument) -> list[EditableTarget]:
    return [t for t in doc.targets if t.review_status == "approved"]


# --------------------------------------------------------------------------
# Per-snapshot evaluation
# --------------------------------------------------------------------------


@dataclass
class BlockOutcome:
    scene: str
    name: str
    gt_id: str
    centre_before: bool
    centre_after: bool
    strict_before: bool
    strict_after: bool
    chosen_yaw_deg: float
    num_hypotheses: int
    num_usable: int


@dataclass
class SnapshotResult:
    scene: str
    name: str
    gt: list[EditableTarget]
    candidate_poses: list[BlockPose]
    corrected_poses: list[BlockPose]
    outcomes: list[BlockOutcome]


def evaluate_snapshot(
    scene: str,
    name: str,
    dataset_root: Path,
    verifier_head,
    verifier_config,
    verifier_temperature: float,
    camera: CameraModel,
    *,
    match_dist: float,
    yaw_angles_deg: tuple[float, ...],
    crop_margin_frac: float,
    out_size: int,
    detection_params: DetectionParameters | None = None,
) -> SnapshotResult:
    d = dataset_root / name
    tf = load_tf_snapshot(d / "tf.yaml")
    points, _ = load_snapshot(
        d, DEFAULT_TRANSFORM_CONFIG, T_world_cloud=tf.T_world_cloud, require_registrations=False
    )
    doc = AnnotationDocument.load(d / "annotations.yaml")
    gt = approved_targets(doc)
    rgb = load_rgb(d / "rgb.png")
    sensor_origin = tf.T_world_cloud[:3, 3]

    raw_hypotheses = _run_detection(points, sensor_origin, detection_params)
    candidate_poses = [h.pose for h in raw_hypotheses if h.fate == "final"]

    corrected_poses: list[BlockPose] = []
    per_candidate_diag: list[dict] = []
    for pose in candidate_poses:
        corrected, diag = disambiguate_candidate(
            pose,
            rgb,
            tf.T_world_camera,
            camera,
            verifier_head,
            verifier_config,
            verifier_temperature,
            yaw_angles_deg=yaw_angles_deg,
            crop_margin_frac=crop_margin_frac,
            out_size=out_size,
        )
        corrected_poses.append(corrected)
        per_candidate_diag.append(diag)

    gt_poses = [BlockPose(t.position, t.rotation, dims=t.dimensions) for t in gt]

    def matched_and_strict(poses: list[BlockPose]) -> tuple[dict[int, int], set[int]]:
        pairs = match_detections(poses, gt_poses, max_dist=match_dist)
        gt_to_det = {gi: di for di, gi in pairs}
        strict = {
            gi
            for gi, di in gt_to_det.items()
            if strict_match(poses[di], gt_poses[gi])["strict_match"]
        }
        return gt_to_det, strict

    before_matches, before_strict = matched_and_strict(candidate_poses)
    after_matches, after_strict = matched_and_strict(corrected_poses)

    outcomes: list[BlockOutcome] = []
    for gi, target in enumerate(gt):
        before_di = before_matches.get(gi)
        after_di = after_matches.get(gi)
        chosen_yaw = per_candidate_diag[after_di]["chosen_yaw_deg"] if after_di is not None else 0.0
        num_hyp = per_candidate_diag[after_di]["num_hypotheses"] if after_di is not None else 0
        num_usable = per_candidate_diag[after_di]["num_usable"] if after_di is not None else 0
        outcomes.append(
            BlockOutcome(
                scene=scene,
                name=name,
                gt_id=target.id,
                centre_before=before_di is not None,
                centre_after=after_di is not None,
                strict_before=gi in before_strict,
                strict_after=gi in after_strict,
                chosen_yaw_deg=chosen_yaw,
                num_hypotheses=num_hyp,
                num_usable=num_usable,
            )
        )

    return SnapshotResult(
        scene=scene,
        name=name,
        gt=gt,
        candidate_poses=candidate_poses,
        corrected_poses=corrected_poses,
        outcomes=outcomes,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def build_report(results: list[SnapshotResult], args: argparse.Namespace) -> str:
    all_outcomes = [o for r in results for o in r.outcomes]
    gt_total = len(all_outcomes)
    centre_before = sum(o.centre_before for o in all_outcomes)
    centre_after = sum(o.centre_after for o in all_outcomes)
    strict_before = sum(o.strict_before for o in all_outcomes)
    strict_after = sum(o.strict_after for o in all_outcomes)

    addressable = [o for o in all_outcomes if o.centre_before and not o.strict_before]
    gained = [o for o in addressable if o.strict_after]
    broken = [
        o for o in all_outcomes if o.strict_before and not o.strict_after
    ]  # should stay empty unless correction moved a previously-fine pose off

    lines: list[str] = []
    lines.append("# `dino_verifier` within-block pose disambiguation")
    lines.append("")
    lines.append(
        "Tests `dino_verifier` in the role its own training data "
        "(`experiments/rgb_candidate_prior/candidates.py`) actually built it for: "
        "picking the right pose among yaw siblings of the SAME block, not "
        "comparing different blocks against each other. See module docstring "
        "of `tools/eval_verifier_pose_disambiguation.py` for the full method."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "1. Run the whole-cloud PCL detector (`annotation_candidates._run_detection`) "
        'on each of the 25 annotated snapshots; keep `fate == "final"` candidates.\n'
        "2. For each accepted candidate, build a pose hypothesis set: the candidate "
        "as-is plus yaw rotations of "
        f"{', '.join(f'{y:g}' for y in args.yaw_hypotheses_deg)} deg about the "
        "candidate's OWN up axis (its body axis nearest world +Z) through its own "
        "position, mirroring `candidates.py`'s hard-negative construction. A rotation "
        "that is a genuine symmetry of the block's own extents "
        "(`blockpose.pose.symmetry_equivalents`) is dropped as not distinguishable -- "
        "for the canonical 0.9x0.6x0.6 laid block this collapses the set to "
        "`{0 deg, 90 deg}` (180 deg is always a symmetry of a rectangular cuboid "
        "about its own vertical axis, and 270 deg is then always equivalent to "
        "90 deg). Yawing about world +Z instead would leave both duplicates in the "
        "set for any candidate tilted more than ~0.5 deg -- 95% of this dataset's -- "
        "silently turning the intended head-to-head argmax into a "
        "best-of-two-per-side one; see the tool's module docstring.\n"
        "3. Render every hypothesis' RGB crop + conditioning image "
        "(`experiments.rgb_candidate_prior.render`), with ALL hypotheses for one "
        "block sharing ONE crop bbox (`render.union_crop_bbox`) so ranking cannot "
        "leak pose through crop framing.\n"
        "4. Score every hypothesis with `dino_verifier` "
        "(`score_candidate_with_dino`), take the argmax by calibrated score, and use "
        "that hypothesis's pose as the corrected pose. An all-abstained hypothesis "
        "set (no usable crop, high uncertainty) keeps the original pose.\n"
        "5. Score centre recall (`blockpose.evaluate.match_detections`, "
        f"{args.match_dist:.2f} m) and strict recall "
        "(`blockpose.experiment_record.strict_match`) before and after correction, "
        "per block."
    )
    lines.append("")
    lines.append(
        "**In-sample numbers**: the loaded checkpoint "
        f"(`{args.verifier_checkpoint}`) is trained on the union of all 25 "
        "annotated snapshots this reports on, unless it points at a "
        "held-out-fold checkpoint instead."
    )
    lines.append("")
    lines.append("| Metric | Before | After |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Centre recall (<= {args.match_dist:.2f} m) | {centre_before}/{gt_total} "
        f"| {centre_after}/{gt_total} |"
    )
    lines.append(f"| Strict recall | {strict_before}/{gt_total} | {strict_after}/{gt_total} |")
    lines.append("")
    lines.append(
        "**Strict** additionally requires symmetry-aware rotation error <= "
        f"{DEFAULT_MAX_ROTATION_DEG:.0f} deg and oriented-box IoU >= {DEFAULT_MIN_IOU:.2f}."
    )
    lines.append("")

    lines.append("## Addressable pool")
    lines.append("")
    lines.append(
        f"Pose disambiguation can only ever change blocks that are already found at "
        f"centre level but fail strict -- it cannot manufacture a centre-level match "
        f"`match_detections` never made, and it cannot fix a candidate that fails "
        f"strict on IoU alone (a yaw rotation does not change fit quality). "
        f"That pool has **{len(addressable)}** block(s) out of {gt_total} GT across "
        f"all {len(results)} snapshots. Of those, {len(gained)} were fixed "
        f"(centre-match retained, strict now passes) and {len(addressable) - len(gained)} "
        f"were not."
    )
    lines.append("")
    if broken:
        lines.append(
            f"**{len(broken)} block(s) that were strict-passing BEFORE correction "
            f"broke AFTER it** -- correction moved a candidate that was already right "
            f"onto a wrong hypothesis. This is a real regression, not noise, and is "
            f"itemised in the per-block table below."
        )
    else:
        lines.append("No block that was strict-passing before correction was broken by it.")
    lines.append("")
    if gt_total > 0:
        lines.append(
            f"**Read this in context**: {len(addressable)} out of {gt_total} GT blocks "
            "is a small pool. A result of +1 or +2 strict recoveries here is inside "
            "this dataset's noise floor and should not be reported as a generalizable "
            "improvement without a held-out check across more scenes."
        )
        lines.append("")

    lines.append("## Per-block outcomes")
    lines.append("")
    lines.append(
        "Only blocks in the addressable pool (centre-matched, strict-failing before "
        "correction) or that regressed are listed individually; every other block's "
        "strict status is unchanged by construction (see Addressable pool above)."
    )
    lines.append("")
    lines.append(
        "| Scene | Snapshot | GT id | Centre before/after | Strict before/after "
        "| Chosen yaw (deg) | Usable/total hypotheses | Outcome |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    interesting = sorted(
        (
            o
            for o in all_outcomes
            if (o.centre_before and not o.strict_before) or (o.strict_before and not o.strict_after)
        ),
        key=lambda o: (o.scene, o.name, o.gt_id),
    )
    for o in interesting:
        if o.strict_before and not o.strict_after:
            outcome = "BROKEN"
        elif not o.strict_before and o.strict_after:
            outcome = "GAINED"
        else:
            outcome = "unchanged"
        lines.append(
            f"| {o.scene} | {o.name} | {o.gt_id} | "
            f"{'Y' if o.centre_before else 'N'}/{'Y' if o.centre_after else 'N'} | "
            f"{'Y' if o.strict_before else 'N'}/{'Y' if o.strict_after else 'N'} | "
            f"{o.chosen_yaw_deg:.0f} | {o.num_usable}/{o.num_hypotheses} | {outcome} |"
        )
    if not interesting:
        lines.append("| (none) | | | | | | | |")
    lines.append("")

    lines.append("## Per-scene aggregate")
    lines.append("")
    lines.append("| Scene | GT | Strict before | Strict after |")
    lines.append("|---|---|---|---|")
    for scene in sorted({o.scene for o in all_outcomes}):
        scene_outcomes = [o for o in all_outcomes if o.scene == scene]
        n = len(scene_outcomes)
        b = sum(o.strict_before for o in scene_outcomes)
        a = sum(o.strict_after for o in scene_outcomes)
        lines.append(f"| {scene} | {n} | {b}/{n} | {a}/{n} |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if not gained and not broken:
        lines.append(
            "No block changed strict status in either direction. "
            f"Aggregate strict recall is unchanged ({strict_before}/{gt_total} -> "
            f"{strict_after}/{gt_total}). This run gives no evidence that within-block "
            "disambiguation, as implemented here, does anything on this dataset."
        )
    elif len(gained) > len(broken):
        lines.append(
            f"Net gain of {len(gained) - len(broken)} block(s) "
            f"({len(gained)} gained, {len(broken)} broken). Aggregate strict recall "
            f"moved {strict_before}/{gt_total} -> {strict_after}/{gt_total}. Given "
            f"the addressable pool is only {len(addressable)} block(s), treat this as "
            "a weak, dataset-scale signal, not a validated improvement."
        )
    elif len(broken) > len(gained):
        lines.append(
            f"Net loss of {len(broken) - len(gained)} block(s) "
            f"({len(gained)} gained, {len(broken)} broken). Disambiguation made "
            "recall worse on this dataset; do not deploy it in this form."
        )
    else:
        lines.append(
            f"{len(gained)} block(s) gained and {len(broken)} block(s) broken -- "
            "they exactly cancel, so the aggregate strict recall is unchanged "
            f"({strict_before}/{gt_total} both before and after) even though "
            f"{len(gained) + len(broken)} individual blocks' poses actually changed. "
            "This is precisely the failure mode the audit that motivated this "
            "script was worried about: an unchanged (or improved) aggregate number "
            "can hide real per-block regressions. On this dataset, within-block "
            "disambiguation via argmax over a {90 deg, base} hypothesis set is "
            "evidence-neutral at best -- it does not net-improve strict recall, and "
            "it is not free of regressions. It should not be described as a working "
            "correction mechanism without a larger, held-out evaluation."
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--verifier-checkpoint", type=Path, default=DEFAULT_VERIFIER_CHECKPOINT)
    parser.add_argument("--match-dist", type=float, default=0.30)
    parser.add_argument("--crop-margin-frac", type=float, default=0.25)
    parser.add_argument("--out-size", type=int, default=224)
    parser.add_argument(
        "--yaw-hypotheses-deg",
        type=float,
        nargs="+",
        default=list(DEFAULT_YAW_HYPOTHESES_DEG),
        help="Yaw angles (deg, about the candidate's own up axis through its own position) "
        "to add to the hypothesis set. Symmetry-equivalent angles for a given "
        "block's dims are dropped automatically.",
    )
    parser.add_argument(
        "--cluster-max-extent-xy", type=float, default=DetectionParameters().cluster_max_extent_xy
    )
    parser.add_argument("--dbscan-eps", type=float, default=DetectionParameters().dbscan_eps)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    from dataclasses import replace

    camera = load_camera_model(args.calibration)
    manifest = load_manifest_snapshots(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    detection_params = replace(
        DetectionParameters(),
        cluster_max_extent_xy=args.cluster_max_extent_xy,
        dbscan_eps=args.dbscan_eps,
    )
    verifier_head, verifier_config, verifier_temperature = load_dino_verifier(
        args.verifier_checkpoint
    )

    results: list[SnapshotResult] = []
    for scene, name in manifest:
        print(f"[{len(results) + 1}/{len(manifest)}] {name}", file=sys.stderr)
        results.append(
            evaluate_snapshot(
                scene,
                name,
                args.dataset_root,
                verifier_head,
                verifier_config,
                verifier_temperature,
                camera,
                match_dist=args.match_dist,
                yaw_angles_deg=tuple(args.yaw_hypotheses_deg),
                crop_margin_frac=args.crop_margin_frac,
                out_size=args.out_size,
                detection_params=detection_params,
            )
        )

    stamp = experiment_stamp(
        detector=detection_params,
        match_dist=args.match_dist,
        yaw_hypotheses_deg=list(args.yaw_hypotheses_deg),
        verifier_checkpoint=str(args.verifier_checkpoint),
    )
    report = build_report(results, args)
    report = f"<!-- provenance: commit {stamp['git_commit']} -->\n" + report
    print(report)
    if args.report_out is not None:
        args.report_out.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
