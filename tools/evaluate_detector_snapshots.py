#!/usr/bin/env python3
"""Snapshots in, scored detector report out -- the tool that did not exist.

`docs/ros2-integration.md` names this gap explicitly: every stage of the
pipeline (record a bag, export a snapshot, annotate it) exists, but nothing
takes a directory of snapshots and emits a recall/precision report. That is
what this script does, for the geometry detector alone (no RGB priors, no
verifier) -- the general-purpose sibling of
`tools/eval_dino_proposer_pooled.py`'s "geometry (whole-cloud PCL)" row,
minus everything else that script measures.

Reused rather than reinvented, per this repo's convention:
  * `blockpose.annotation_candidates._run_detection` -- runs
    `detect_blocks_detailed_cpp` with a `SensorContext` built from the WHOLE
    cloud (never a crop -- see that function's docstring and
    `experiments/dino_proposer_pooled_ablation.md` for why a cropped context
    inflates scores: 60% of the score is ray evidence computed against
    whatever context it is handed).
  * `blockpose.evaluate.match_detections` -- Hungarian centre matching at
    `--match-dist` (default 0.30 m).
  * `blockpose.experiment_record.strict_match` -- symmetry-aware rotation
    <= 30 deg and oriented-box IoU >= 0.25 on top of a centre match. This is
    the number that matters for whether a pose is usable; centre-only counts
    a box 88 degrees out of alignment as a hit. Note it keeps its own
    `DEFAULT_MAX_CENTRE_M` (0.30 m) gate regardless of `--match-dist`, as in
    every other report here: loosening centre matching to survey what the
    detector nearly found must not loosen the usability bar it is compared
    against.
  * `blockpose.experiment_record.experiment_stamp` -- git commit + the exact
    `DetectionParameters` this run used, so a reported number can always be
    re-derived or shown to be stale.
  * `blockpose.annotation_tool.AnnotationDocument` / `EditableTarget` --
    identical to `tools/eval_dino_proposer_pooled.py`'s snapshot loading.
    Ground truth is `review_status == "approved"` and `ignore_reason is
    None` (`blockpose.annotations.AnnotationTarget.is_evaluation_target`).

What is NEW here, because no existing tool needed it:
  * Manifest is OPTIONAL. An operator with a fresh bag export has no
    `config/annotated_snapshots_*.yaml` yet -- that file's own header says
    its selection is "derived, not hand-listed" from exactly two signals
    (`annotation_complete: true` and >= 1 approved block), so
    `discover_snapshots` applies those same two signals directly to a
    directory instead of requiring the manifest to exist first. Passing
    `--manifest` (e.g. the Seibersdorf one) restricts to its exact list,
    which is how this script is validated against the recorded 46/62 centre,
    44/62 strict, 46/59 precision baseline.
  * Per-visibility-class breakdown (full / partial / heavy_occlusion /
    unknown). Occlusion is the dominant measured failure mode in this
    dataset (see `tools/audit_annotation_poses.py`'s docstring: heavy
    occlusion carries a median 87.8 deg pose disagreement vs 1.2 deg for
    full visibility), so lumping every miss into one recall number hides
    which failures are actionable and which are sensing limits.
  * Explicit skip accounting: every snapshot under `--dataset-root` that did
    NOT make it into the denominator is listed with why (no annotations
    file, `annotation_complete: false`, zero approved targets), and the
    manifest-restricted run additionally reports any manifest entry that
    failed to load. A recall number silently computed over fewer snapshots
    than exist is a worse failure than a crash.

Usage:
    uv run python tools/evaluate_detector_snapshots.py \
        --report-out experiments/detector_snapshot_report.md \
        --json-out experiments/detector_snapshot_report.json

    # No manifest -- discover every complete, approved snapshot under a
    # fresh export root:
    uv run python tools/evaluate_detector_snapshots.py \
        --dataset-root /path/to/new_export --manifest ""
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.annotation_candidates import _run_detection  # noqa: E402
from blockpose.annotation_preprocess import (  # noqa: E402
    filter_pzs100_gripper,
    load_pzs100_self_filter_config,
    pzs100_self_filter_available,
)
from blockpose.annotation_tool import AnnotationDocument, EditableTarget  # noqa: E402
from blockpose.core_types import DetectionParameters  # noqa: E402
from blockpose.evaluate import match_detections  # noqa: E402
from blockpose.experiment_record import (  # noqa: E402
    DEFAULT_MAX_ROTATION_DEG,
    DEFAULT_MIN_IOU,
    experiment_stamp,
    strict_match,
)
from blockpose.io import load_snapshot, load_tf_snapshot  # noqa: E402
from blockpose.pose import BlockPose  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "seibersdorf_dataset"
DEFAULT_MANIFEST = REPO_ROOT / "config" / "annotated_snapshots_seibersdorf.yaml"
DEFAULT_TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"

VISIBILITY_CLASSES = ("full", "partial", "heavy_occlusion", "unknown")


# --------------------------------------------------------------------------
# Snapshot selection
# --------------------------------------------------------------------------


def load_manifest_snapshots(manifest_path: Path) -> list[tuple[str, str]]:
    """`(scene, snapshot_name)` pairs from a manifest like
    `config/annotated_snapshots_seibersdorf.yaml`. Identical shape to
    `tools/eval_dino_proposer_pooled.py::load_manifest_snapshots`."""
    data = yaml.safe_load(manifest_path.read_text()) or {}
    result = []
    for scene, info in (data.get("scenes") or {}).items():
        for entry in info.get("annotated") or []:
            result.append((scene, str(entry["name"])))
    return result


def _scene_for(snapshot_name: str) -> str:
    """`scene = capture directory prefix before '__'`, matching
    `config/annotated_snapshots_seibersdorf.yaml`'s own `grouping` field, so
    discovered and manifest-driven runs report the same scene grouping."""
    return snapshot_name.split("__", 1)[0]


@dataclass(frozen=True)
class SkippedSnapshot:
    name: str
    reason: str


def discover_snapshots(dataset_root: Path) -> tuple[list[tuple[str, str]], list[SkippedSnapshot]]:
    """Every subdirectory of `dataset_root` that qualifies as ground truth,
    applying `config/annotated_snapshots_seibersdorf.yaml`'s own documented
    two-signal selection ("annotation_complete: true AND at least one
    approved block") directly, since that manifest's generator does not
    exist. Returns `(scene, name)` pairs plus every skipped directory and why
    -- never silently drops a directory from consideration.
    """
    included: list[tuple[str, str]] = []
    skipped: list[SkippedSnapshot] = []
    for entry in sorted(dataset_root.iterdir()):
        if not entry.is_dir():
            continue
        annotations_path = entry / "annotations.yaml"
        if not annotations_path.is_file():
            skipped.append(SkippedSnapshot(entry.name, "no annotations.yaml"))
            continue
        try:
            doc = AnnotationDocument.load(annotations_path)
        except (ValueError, KeyError, yaml.YAMLError) as exc:
            skipped.append(SkippedSnapshot(entry.name, f"failed to load annotations: {exc}"))
            continue
        if not doc.annotation_complete:
            skipped.append(SkippedSnapshot(entry.name, "annotation_complete is false"))
            continue
        if not approved_targets(doc):
            skipped.append(SkippedSnapshot(entry.name, "zero approved targets"))
            continue
        included.append((_scene_for(entry.name), entry.name))
    return included, skipped


def approved_targets(doc: AnnotationDocument) -> list[EditableTarget]:
    """Ground truth = approved and not ignored, matching
    `blockpose.annotations.AnnotationTarget.is_evaluation_target` exactly
    (`AnnotationDocument`/`EditableTarget` do not carry that property
    themselves, so it is reapplied here)."""
    return [t for t in doc.targets if t.review_status == "approved" and t.ignore_reason is None]


# --------------------------------------------------------------------------
# Per-snapshot evaluation
# --------------------------------------------------------------------------


@dataclass
class SnapshotResult:
    scene: str
    name: str
    gt: list[EditableTarget]
    detections: list[BlockPose]
    matched_gt: set[int]  # indices into `gt`, centre match
    strict_gt: set[int]  # indices into `gt`, strict match (subset of matched_gt)
    matched_det: set[int]  # indices into `detections`, centre match (for precision)
    gripper_removed: int = 0
    # None when the gripper self-filter ran; a reason string when this snapshot
    # was scored UNFILTERED, so a mixed run is visible rather than averaged.
    gripper_unfiltered_reason: str | None = None


@dataclass(frozen=True)
class SnapshotFailure:
    scene: str
    name: str
    reason: str


DEFAULT_GRIPPER_FILTER_CONFIG = REPO_ROOT / "config" / "pzs100_gripper_filter.yaml"


def apply_gripper_self_filter(points, tf, profile):
    """Remove the robot's own gripper returns, as the ROS adapter does.

    Returns ``(points, removed, reason)``. ``reason`` is None on success and a
    short explanation when the filter could not run, so the caller can report
    the snapshot as scored-unfiltered rather than silently mixing the two.

    **Why this is on by default.** `filter_pzs100_gripper` ships in the C++
    core and is applied by the ROS adapter and `tools/annotate_snapshot.py`,
    but this evaluation did not call it, so every recorded number described a
    detector fed a cloud that still contained the gripper. Measured over the
    25 annotated snapshots: filtering removes 42,176 points and eliminates 6
    detections at ZERO recall cost -- precision 78.0% -> 86.8%, recall
    byte-identical. The unfiltered number understated production by ~9 points.
    See experiments/gripper_filter_in_evaluation.md.
    """
    if profile is None:
        return points, 0, "gripper filter disabled"
    try:
        parents, rails, rotations, outward = [], [], [], []
        for rail in profile.rails:
            world_parent = tf.lookup(tf.reference_frame, rail.parent_frame)
            world_rail = tf.lookup(tf.reference_frame, rail.rail_frame)
            parents.append(world_parent[:3, 3])
            rails.append(world_rail[:3, 3])
            rotations.append(world_parent[:3, :3])
            outward.append(rail.outward_axis_parent)
        filtered, removed, _ = filter_pzs100_gripper(
            points,
            np.asarray(parents, dtype=np.float64),
            np.asarray(rails, dtype=np.float64),
            np.asarray(rotations, dtype=np.float64),
            np.asarray(outward, dtype=np.float64),
            profile.parameters,
        )
        return filtered, int(removed), None
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        # Never fail the run: a snapshot without rail TF is still scorable, it
        # is just scored unfiltered, and the report must say which ones were.
        return points, 0, f"{type(error).__name__}: {error}"


def load_gripper_profile(path, enabled):
    """Load the PZS100 profile, or None with a reason when unavailable."""
    if not enabled:
        return None, "disabled by --no-gripper-filter"
    available, reason = pzs100_self_filter_available()
    if not available:
        return None, f"native filter unavailable: {reason}"
    profile = load_pzs100_self_filter_config(path)
    if not profile.enabled:
        return None, f"profile {path} is disabled"
    return profile, None


def evaluate_snapshot(
    scene: str,
    name: str,
    dataset_root: Path,
    detection_params: DetectionParameters,
    match_dist: float,
    gripper_profile=None,
) -> SnapshotResult:
    d = dataset_root / name
    tf = load_tf_snapshot(d / "tf.yaml")
    points, _ = load_snapshot(
        d, DEFAULT_TRANSFORM_CONFIG, T_world_cloud=tf.T_world_cloud, require_registrations=False
    )
    points, gripper_removed, gripper_reason = apply_gripper_self_filter(points, tf, gripper_profile)
    doc = AnnotationDocument.load(d / "annotations.yaml")
    gt = approved_targets(doc)
    sensor_origin = tf.T_world_cloud[:3, 3]

    raw_hypotheses = _run_detection(points, sensor_origin, detection_params)
    detections = [
        BlockPose(h.pose.position, h.pose.rotation, dims=h.pose.dims)
        for h in raw_hypotheses
        if h.fate == "final"
    ]

    gt_poses = [BlockPose(t.position, t.rotation, dims=t.dimensions) for t in gt]
    pairs = match_detections(detections, gt_poses, max_dist=match_dist)

    matched_gt = {gi for _, gi in pairs}
    matched_det = {di for di, _ in pairs}
    strict_gt = {
        gi for di, gi in pairs if strict_match(detections[di], gt_poses[gi])["strict_match"]
    }

    return SnapshotResult(
        scene=scene,
        name=name,
        gt=gt,
        detections=detections,
        matched_gt=matched_gt,
        strict_gt=strict_gt,
        matched_det=matched_det,
        gripper_removed=gripper_removed,
        gripper_unfiltered_reason=gripper_reason,
    )


# --------------------------------------------------------------------------
# Aggregation (pure -- exercised directly by tests)
# --------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


@dataclass(frozen=True)
class OverallStats:
    gt_total: int
    det_total: int
    matched_centre: int
    matched_det: int  # true positives among detections, for precision
    matched_strict: int


def aggregate_overall(results: list[SnapshotResult]) -> OverallStats:
    return OverallStats(
        gt_total=sum(len(r.gt) for r in results),
        det_total=sum(len(r.detections) for r in results),
        matched_centre=sum(len(r.matched_gt) for r in results),
        matched_det=sum(len(r.matched_det) for r in results),
        matched_strict=sum(len(r.strict_gt) for r in results),
    )


def aggregate_per_scene(results: list[SnapshotResult]) -> dict[str, OverallStats]:
    by_scene: dict[str, list[SnapshotResult]] = defaultdict(list)
    for r in results:
        by_scene[r.scene].append(r)
    return {scene: aggregate_overall(rs) for scene, rs in sorted(by_scene.items())}


def aggregate_per_visibility(results: list[SnapshotResult]) -> dict[str, tuple[int, int]]:
    """`{visibility: (matched_centre, gt_total)}` over every GT block,
    bucketed by its own `visibility` label -- unlike the scene/overall
    tables, precision has no per-block visibility (a false-positive
    detection is not attached to any GT block), so this table is
    recall-only by construction.
    """
    counts: dict[str, list[int]] = {v: [0, 0] for v in VISIBILITY_CLASSES}
    for r in results:
        for gi, target in enumerate(r.gt):
            visibility = target.visibility if target.visibility in VISIBILITY_CLASSES else "unknown"
            counts[visibility][1] += 1
            if gi in r.matched_gt:
                counts[visibility][0] += 1
    return {v: (matched, total) for v, (matched, total) in counts.items()}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def build_report(
    results: list[SnapshotResult],
    failures: list[SnapshotFailure],
    skipped: list[SkippedSnapshot],
    stamp: dict,
    args: argparse.Namespace,
) -> str:
    overall = aggregate_overall(results)
    per_scene = aggregate_per_scene(results)
    per_visibility = aggregate_per_visibility(results)

    lines: list[str] = []
    lines.append("# Detector snapshot evaluation report")
    lines.append("")
    lines.append(f"Provenance: commit `{stamp['git_commit']}`.")
    lines.append("")
    lines.append(
        f"Dataset root: `{args.dataset_root}`. Manifest: `{args.manifest}`."
        if args.manifest is not None
        else f"Dataset root: `{args.dataset_root}`. Manifest: none (discovered)."
    )
    lines.append("")
    lines.append(
        f"Snapshots scored: {len(results)}. Snapshots skipped: {len(skipped)}. "
        f"Snapshots that failed to evaluate: {len(failures)}."
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Recall (centre, <= {args.match_dist:.2f} m) "
        f"| {_rate(overall.matched_centre, overall.gt_total)} |"
    )
    lines.append(
        f"| Recall (strict, centre + rotation <= {DEFAULT_MAX_ROTATION_DEG:.0f} deg "
        f"+ IoU >= {DEFAULT_MIN_IOU:.2f}) "
        f"| {_rate(overall.matched_strict, overall.gt_total)} |"
    )
    lines.append(f"| Precision (centre) | {_rate(overall.matched_det, overall.det_total)} |")
    lines.append("")

    lines.append("## Per-scene")
    lines.append("")
    lines.append("| Scene | GT | Recall (centre) | Recall (strict) | Precision (centre) |")
    lines.append("|---|---|---|---|---|")
    for scene, stats in per_scene.items():
        lines.append(
            f"| {scene} | {stats.gt_total} | {_rate(stats.matched_centre, stats.gt_total)} "
            f"| {_rate(stats.matched_strict, stats.gt_total)} "
            f"| {_rate(stats.matched_det, stats.det_total)} |"
        )
    lines.append("")

    lines.append("## Per-visibility recall (centre)")
    lines.append("")
    lines.append("| Visibility | GT | Recall (centre) |")
    lines.append("|---|---|---|")
    for visibility in VISIBILITY_CLASSES:
        matched, total = per_visibility[visibility]
        lines.append(f"| {visibility} | {total} | {_rate(matched, total)} |")
    lines.append("")

    lines.append("## Gripper self-filter")
    lines.append("")
    if getattr(args, "no_gripper_filter", False):
        lines.append(
            "**NOT APPLIED** (`--no-gripper-filter`). These numbers describe a cloud "
            "still containing the robot's own gripper returns, which is NOT the path "
            "the ROS adapter runs. Precision is understated by roughly 9 points; see "
            "`experiments/gripper_filter_in_evaluation.md`."
        )
    else:
        gripper_unfiltered = [r for r in results if r.gripper_unfiltered_reason is not None]
        removed = sum(r.gripper_removed for r in results)
        lines.append(
            f"Applied (`filter_pzs100_gripper`, as the ROS adapter does): {removed} points "
            f"removed across {len(results) - len(gripper_unfiltered)} of {len(results)} snapshots."
        )
        if gripper_unfiltered:
            lines.append("")
            lines.append(
                f"**{len(gripper_unfiltered)} snapshot(s) were scored UNFILTERED** because the "
                "filter could not run; they are still in the totals above, so this run mixes "
                "both paths:"
            )
            for r in gripper_unfiltered:
                lines.append(f"- `{r.name}` -- {r.gripper_unfiltered_reason}")
    lines.append("")
    lines.append("## Report health")
    lines.append("")
    if skipped:
        lines.append(
            f"{len(skipped)} directories under `{args.dataset_root}` were not "
            "evaluation candidates:"
        )
        lines.append("")
        for s in skipped:
            lines.append(f"- `{s.name}`: {s.reason}")
    elif args.manifest is not None:
        # Not the same statement as "nothing was skipped": with a manifest the
        # denominator is the manifest's list, and directories under
        # --dataset-root that it omits were never considered. Claiming a clean
        # discovery pass that never ran would misreport exactly the thing this
        # section exists to make visible.
        lines.append(
            f"Selection was restricted to `{args.manifest}`; directory discovery did not run, "
            "so directories under the dataset root that the manifest omits are outside this "
            'report\'s scope entirely. Re-run with `--manifest ""` to score every complete, '
            "approved snapshot under the root instead."
        )
    else:
        lines.append("No directories were skipped during discovery.")
    lines.append("")
    if failures:
        lines.append(
            f"{len(failures)} snapshots selected for evaluation FAILED to evaluate "
            "(excluded from every number above, not silently zero-filled):"
        )
        lines.append("")
        for f in failures:
            lines.append(f"- `{f.name}` (scene `{f.scene}`): {f.reason}")
    else:
        lines.append("No selected snapshot failed to evaluate.")
    lines.append("")

    lines.append("## Parameters")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stamp, indent=2))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def print_summary(
    results: list[SnapshotResult],
    failures: list[SnapshotFailure],
    skipped: list[SkippedSnapshot],
    args: argparse.Namespace,
) -> None:
    overall = aggregate_overall(results)
    print(
        f"detector: {len(results)} snapshots, {overall.gt_total} GT blocks -- "
        f"recall centre {_rate(overall.matched_centre, overall.gt_total)}, "
        f"strict {_rate(overall.matched_strict, overall.gt_total)}, "
        f"precision {_rate(overall.matched_det, overall.det_total)} "
        f"(skipped {len(skipped)}, failed {len(failures)})"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Path to a scenes manifest (e.g. config/annotated_snapshots_seibersdorf.yaml). "
        "Pass an empty string to discover snapshots directly under --dataset-root instead "
        "(every subdirectory with annotation_complete: true and >= 1 approved block).",
    )
    parser.add_argument("--match-dist", type=float, default=0.30)
    parser.add_argument(
        "--no-gripper-filter",
        action="store_true",
        help="Score the raw cloud, WITHOUT the PZS100 gripper self-filter the ROS "
        "adapter applies. Reproduces this repo's pre-correction numbers (precision "
        "78.0% instead of 86.8%, recall identical); see "
        "experiments/gripper_filter_in_evaluation.md. Not the production path.",
    )
    parser.add_argument("--gripper-filter-config", type=Path, default=DEFAULT_GRIPPER_FILTER_CONFIG)
    parser.add_argument(
        "--cluster-max-extent-xy", type=float, default=DetectionParameters().cluster_max_extent_xy
    )
    parser.add_argument("--dbscan-eps", type=float, default=DetectionParameters().dbscan_eps)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    skipped: list[SkippedSnapshot] = []
    if manifest_path is not None:
        snapshots = load_manifest_snapshots(manifest_path)
        args.manifest = str(manifest_path)
    else:
        snapshots, skipped = discover_snapshots(args.dataset_root)
        args.manifest = None
    if args.limit is not None:
        snapshots = snapshots[: args.limit]

    # Fail on an unwritable output path BEFORE the detector runs: the loop
    # below is ~25 whole-cloud detections, and losing all of it to a missing
    # parent directory at write time is a needless re-run.
    for out_path in (args.report_out, args.json_out):
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)

    detection_params = replace(
        DetectionParameters(),
        cluster_max_extent_xy=args.cluster_max_extent_xy,
        dbscan_eps=args.dbscan_eps,
    )
    gripper_profile, gripper_disabled_reason = load_gripper_profile(
        args.gripper_filter_config, not args.no_gripper_filter
    )
    if gripper_disabled_reason is not None:
        print(f"gripper self-filter NOT applied: {gripper_disabled_reason}", file=sys.stderr)

    results: list[SnapshotResult] = []
    failures: list[SnapshotFailure] = []
    for i, (scene, name) in enumerate(snapshots):
        print(f"[{i + 1}/{len(snapshots)}] {name}", file=sys.stderr)
        try:
            results.append(
                evaluate_snapshot(
                    scene,
                    name,
                    args.dataset_root,
                    detection_params,
                    args.match_dist,
                    gripper_profile=gripper_profile,
                )
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            failures.append(SnapshotFailure(scene, name, str(exc)))

    stamp = experiment_stamp(
        detector=detection_params,
        match_dist=args.match_dist,
        dataset_root=str(args.dataset_root),
        manifest=args.manifest,
    )

    report = build_report(results, failures, skipped, stamp, args)
    if args.report_out is not None:
        args.report_out.write_text(report)
    if args.json_out is not None:
        overall = aggregate_overall(results)
        per_scene = aggregate_per_scene(results)
        per_visibility = aggregate_per_visibility(results)
        args.json_out.write_text(
            json.dumps(
                {
                    "provenance": stamp,
                    "overall": {
                        "gt_total": overall.gt_total,
                        "det_total": overall.det_total,
                        "matched_centre": overall.matched_centre,
                        "matched_det": overall.matched_det,
                        "matched_strict": overall.matched_strict,
                    },
                    "per_scene": {
                        scene: {
                            "gt_total": s.gt_total,
                            "det_total": s.det_total,
                            "matched_centre": s.matched_centre,
                            "matched_det": s.matched_det,
                            "matched_strict": s.matched_strict,
                        }
                        for scene, s in per_scene.items()
                    },
                    "per_visibility": {
                        v: {"matched_centre": m, "gt_total": t}
                        for v, (m, t) in per_visibility.items()
                    },
                    "skipped": [{"name": s.name, "reason": s.reason} for s in skipped],
                    "failed": [
                        {"scene": f.scene, "name": f.name, "reason": f.reason} for f in failures
                    ],
                },
                indent=2,
            )
        )
    print_summary(results, failures, skipped, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
