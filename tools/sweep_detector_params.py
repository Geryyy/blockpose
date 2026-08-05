#!/usr/bin/env python3
"""Diagnose the merged-cluster recall failure and sweep DBSCAN/extent gates.

Context: the PCL detector reaches 63% recall (39/62) / 75% precision (39/52)
on the 62 approved blocks in `config/annotated_snapshots_seibersdorf.yaml`.
9 of those 62 sit inside a post-ground-removal DBSCAN cluster whose XY extent
exceeds `DetectionParameters.cluster_max_extent_xy` (1.5 m), concentrated on
the worst-recall wall/partially-wet scenes -- the working hypothesis is that
blocks touching in a wall merge into one connected component that the gate
then rejects outright (see `cpp/include/blockpose/core/pipeline.hpp`'s
`wide_proposal_skips`: an unsplit component wider than the gate in X or Y is
dropped before any plane is ever fit to it).

This tool (a) independently measures that connected-component geometry per
GT block, in pure Python, since the C++ core does not expose per-DBSCAN-label
extents to Python (`blockpose._core.discover`'s result dict has no
`proposal_diagnostics`/`wide_proposal_skips` key -- see
`cpp/bindings/blockpose_core_module.cpp`), and (b) sweeps `cluster_max_extent_xy`
and `dbscan_eps` through the real C++ detector (calling
`detect_blocks_detailed_cpp` directly with custom `DetectionParameters`,
exactly as `annotation_candidates.py::_run_detection` does internally --
`compute_candidate_sources` hardcodes `DetectionParameters()` so it cannot be
swept) to see whether parameters alone can recover the 9 blocks without
losing precision elsewhere.

The DBSCAN clustering used for (a) is a Python-side proxy (scikit-learn's
`DBSCAN`, same eps/min_samples/Euclidean metric as the core's grid-based
DBSCAN in `proposals.hpp::dbscan_label_components`) run on a Python
replica of the core's voxel-downsample + ground-removal pipeline. It is not
bit-identical to the C++ core -- notably the "ground_thickness" cut used to
build the DBSCAN seed set is approximated with the *planar* ground model
(`remove_ground`'s returned normal/offset), since the core's local per-cell
terrain correction (`LocalGroundModel.cell_z`) is not exposed to Python
either. Component *membership* should match closely (same eps/min_samples on
nearly the same point set); reported extents may differ from the core's own
by a small amount at the flat-vs-local-terrain margin. This caveat is
reported, not hidden, in `experiments/detector_cluster_failures.md`.

Run: `python tools/sweep_detector_params.py [--json-out PATH]`
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import DBSCAN

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotation_preprocess import GroundRemovalParams, remove_ground  # noqa: E402
from blockpose.annotation_tool import AnnotationDocument  # noqa: E402
from blockpose.core_bridge import detect_blocks_detailed_cpp  # noqa: E402
from blockpose.core_types import DetectionParameters, SensorContext, TraceHypothesis  # noqa: E402
from blockpose.io import load_snapshot, load_tf_snapshot  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config/annotated_snapshots_seibersdorf.yaml"
DEFAULT_TRANSFORMS = REPO_ROOT / "config/transforms_pzs100_initial_pose_1.yaml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/seibersdorf_dataset"

# Matches the annotation tool's own detected-vs-GT criterion (see prompt/task).
MATCH_RADIUS_M = 0.30
# Mirrors annotation_candidates.py's _MIN_POINTS_FOR_DETECTION -- kept in sync
# by hand since that constant is private and this tool must not import it.
MIN_POINTS_FOR_DETECTION = 30

DEFAULT_CLUSTER_MAX_EXTENT_GRID = (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5)
DEFAULT_DBSCAN_EPS_GRID = (0.03, 0.05, 0.07, 0.08, 0.10)


# ---------------------------------------------------------------------------
# Pure helpers -- no dataset access, unit tested in test_detector_sweep.py.
# ---------------------------------------------------------------------------


def voxel_downsample_mean(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Average points into floor-indexed voxels.

    Replicates `cpp/include/blockpose/core/pipeline.hpp::voxel_downsample`
    (mean per floor-indexed voxel cell) since that step has no Python
    binding. Grouping by voxel index rather than iterating in input order
    makes the result independent of point order, matching the core's
    canonical-order guarantee.
    """
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape((0, 3))
    cell = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inverse.reshape(-1), points)
    return sums / counts[:, None]


def xy_extent(points: np.ndarray) -> tuple[float, float]:
    """(dx, dy) axis-aligned bounding-box extent.

    Matches `extent(proposal)[0]`/`[1]` in `proposals.hpp`, which is what the
    core's own `cluster_max_extent_xy` gate compares per-axis (not a
    diagonal) -- see `pipeline.hpp`'s `proposal_extent[0] > ... ||
    proposal_extent[1] > ...`.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return (0.0, 0.0)
    span = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
    return float(span[0]), float(span[1])


def nearest_component_label(
    points: np.ndarray, labels: np.ndarray, query: np.ndarray
) -> tuple[int, float]:
    """DBSCAN label of the point nearest `query`, and that distance (m).

    Defines "the connected component containing a GT block" as the component
    of whichever surviving point sits closest to the GT centre. There is no
    guarantee any point sits exactly at the annotated centre (the centre is a
    human-fit box centroid, not a LiDAR return), so nearest-point is the
    least assumption-laden way to still tie every GT block to a component
    whenever one exists nearby. Returns label -1 (DBSCAN noise) with
    `inf` distance when `points` is empty.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return -1, float("inf")
    distances = np.linalg.norm(points - np.asarray(query, dtype=np.float64), axis=1)
    index = int(np.argmin(distances))
    return int(labels[index]), float(distances[index])


@dataclass(frozen=True)
class ScoreResult:
    """Aggregate recall/precision counts under the many-to-many match rule."""

    detected_gt: int
    total_gt: int
    accepted: int
    true_positive_accepted: int

    @property
    def recall(self) -> float:
        return self.detected_gt / self.total_gt if self.total_gt else float("nan")

    @property
    def precision(self) -> float:
        return self.true_positive_accepted / self.accepted if self.accepted else float("nan")

    def __add__(self, other: "ScoreResult") -> "ScoreResult":
        return ScoreResult(
            self.detected_gt + other.detected_gt,
            self.total_gt + other.total_gt,
            self.accepted + other.accepted,
            self.true_positive_accepted + other.true_positive_accepted,
        )


EMPTY_SCORE = ScoreResult(0, 0, 0, 0)


def score_matches(
    gt_positions: list[np.ndarray],
    candidate_positions: list[np.ndarray],
    radius: float = MATCH_RADIUS_M,
) -> tuple[list[bool], list[bool]]:
    """Which GT positions were detected, and which candidates matched a GT.

    A GT counts as detected when ANY candidate centre is within `radius`; a
    candidate counts as a true positive when it is within `radius` of ANY GT.
    This many-to-many rule (not a one-to-one Hungarian assignment) is exactly
    what the task's verified baseline snippet computes -- reproduced here as
    a pure function so both the baseline and every sweep cell score
    identically, and so it is unit-testable without loading the dataset.
    """
    gt_hit = [False] * len(gt_positions)
    cand_hit = [False] * len(candidate_positions)
    for gi, g in enumerate(gt_positions):
        for ci, c in enumerate(candidate_positions):
            if np.linalg.norm(np.asarray(c) - np.asarray(g)) <= radius:
                gt_hit[gi] = True
                cand_hit[ci] = True
    return gt_hit, cand_hit


# ---------------------------------------------------------------------------
# Dataset-backed pipeline.
# ---------------------------------------------------------------------------


@dataclass
class SnapshotData:
    scene: str
    name: str
    points: np.ndarray
    sensor_origin: np.ndarray
    gt_ids: list[str]
    gt_positions: list[np.ndarray]


@dataclass(frozen=True)
class ClusterAssignment:
    """One GT block's nearest post-ground-removal DBSCAN component."""

    scene: str
    snapshot: str
    gt_id: str
    component_label: int  # -1: no surviving point found nearby (see nearest_component_label)
    nearest_distance_m: float
    extent_x: float
    extent_y: float
    point_count: int
    detected_baseline: bool


def load_manifest(config_path: Path) -> list[tuple[str, str]]:
    """Every (scene, snapshot name) pair under `scenes -> * -> annotated -> name`."""
    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pairs = []
    for scene, info in doc["scenes"].items():
        for entry in info["annotated"]:
            pairs.append((scene, entry["name"]))
    return pairs


def load_all_snapshots(
    dataset_root: Path, transforms_cfg: Path, manifest: list[tuple[str, str]]
) -> list[SnapshotData]:
    snapshots = []
    for scene, name in manifest:
        snapshot_dir = dataset_root / name
        tf = load_tf_snapshot(snapshot_dir / "tf.yaml")
        points, _ = load_snapshot(
            snapshot_dir,
            transforms_cfg,
            T_world_cloud=tf.T_world_cloud,
            require_registrations=False,
        )
        targets = [
            t
            for t in AnnotationDocument.load(snapshot_dir / "annotations.yaml").targets
            if t.review_status == "approved"
        ]
        snapshots.append(
            SnapshotData(
                scene=scene,
                name=name,
                points=points,
                sensor_origin=tf.T_world_cloud[:3, 3],
                gt_ids=[t.id for t in targets],
                gt_positions=[t.position for t in targets],
            )
        )
    return snapshots


def run_detection(
    points: np.ndarray, sensor_origin: np.ndarray | None, params: DetectionParameters
) -> tuple[TraceHypothesis, ...]:
    """Raw hypotheses for one point set, mirroring
    `annotation_candidates.py::_run_detection` exactly (same min-point guard,
    same SensorContext construction, same exception handling) but taking
    `params` as an argument -- that module hardcodes `DetectionParameters()`
    internally, so it cannot be swept, and must not be modified per the task.
    """
    if len(points) < MIN_POINTS_FOR_DETECTION:
        return ()
    sensor_context = (
        SensorContext.from_points(sensor_origin, points) if sensor_origin is not None else None
    )
    try:
        result = detect_blocks_detailed_cpp(points, params=params, sensor_context=sensor_context)
    except (RuntimeError, ValueError):
        return ()
    assert result.trace is not None
    return result.trace.raw_hypotheses


def accepted_positions(hypotheses: tuple[TraceHypothesis, ...]) -> list[np.ndarray]:
    """Accepted candidate centres, matching `_pcl_candidates`'s `fate == "final"` rule."""
    return [h.pose.position for h in hypotheses if h.fate == "final"]


def per_gt_detected(
    snapshots: list[SnapshotData], params: DetectionParameters
) -> dict[tuple[str, str], bool]:
    """detected flag per (snapshot name, gt id), running the real detector once per snapshot."""
    out: dict[tuple[str, str], bool] = {}
    for snap in snapshots:
        hypotheses = run_detection(snap.points, snap.sensor_origin, params)
        positions = accepted_positions(hypotheses)
        gt_hit, _ = score_matches(snap.gt_positions, positions)
        for gid, hit in zip(snap.gt_ids, gt_hit, strict=True):
            out[(snap.name, gid)] = hit
    return out


def score_snapshots(snapshots: list[SnapshotData], params: DetectionParameters) -> ScoreResult:
    total = EMPTY_SCORE
    for snap in snapshots:
        hypotheses = run_detection(snap.points, snap.sensor_origin, params)
        positions = accepted_positions(hypotheses)
        gt_hit, cand_hit = score_matches(snap.gt_positions, positions)
        cell = ScoreResult(sum(gt_hit), len(snap.gt_positions), len(positions), sum(cand_hit))
        total = total + cell
    return total


def per_scene_scores(
    snapshots: list[SnapshotData], params: DetectionParameters
) -> dict[str, ScoreResult]:
    agg: dict[str, ScoreResult] = {}
    for snap in snapshots:
        hypotheses = run_detection(snap.points, snap.sensor_origin, params)
        positions = accepted_positions(hypotheses)
        gt_hit, cand_hit = score_matches(snap.gt_positions, positions)
        cell = ScoreResult(sum(gt_hit), len(snap.gt_positions), len(positions), sum(cand_hit))
        agg[snap.scene] = agg.get(snap.scene, EMPTY_SCORE) + cell
    return agg


def analyze_clusters(
    snapshot: SnapshotData, baseline_detected: dict[tuple[str, str], bool]
) -> list[ClusterAssignment]:
    """Task 1: nearest post-ground-removal DBSCAN component per GT block.

    Uses `DetectionParameters()` defaults throughout (voxel_size, dbscan_eps,
    dbscan_min_points, ground_thickness) -- "the detector's own ground
    removal" / "the detector's voxel size" from the task, not a swept value.
    """
    defaults = DetectionParameters()
    downsampled = voxel_downsample_mean(snapshot.points, defaults.voxel_size)
    above_ground, ground = remove_ground(downsampled, GroundRemovalParams())
    normal = np.asarray(ground["normal"], dtype=np.float64)
    offset = float(ground["offset"])
    # Planar approximation of the core's local ground model (see module
    # docstring): height above the fitted ground plane, thresholded the same
    # way pipeline.hpp builds its DBSCAN seed set (`ground.height(point) >=
    # ground_thickness`), just without the local per-cell terrain correction.
    height = above_ground @ normal + offset
    seed = above_ground[height >= defaults.ground_thickness]

    if len(seed) == 0:
        labels = np.zeros((0,), dtype=np.int64)
    else:
        dbscan = DBSCAN(eps=defaults.dbscan_eps, min_samples=defaults.dbscan_min_points)
        labels = dbscan.fit_predict(seed)
    extents: dict[int, tuple[float, float]] = {}
    counts: dict[int, int] = {}
    for label in set(labels.tolist()):
        if label == -1:
            continue
        member_points = seed[labels == label]
        extents[label] = xy_extent(member_points)
        counts[label] = int(member_points.shape[0])

    assignments = []
    for gid, gpos in zip(snapshot.gt_ids, snapshot.gt_positions, strict=True):
        label, distance = nearest_component_label(seed, labels, gpos)
        ex, ey = extents.get(label, (0.0, 0.0))
        assignments.append(
            ClusterAssignment(
                scene=snapshot.scene,
                snapshot=snapshot.name,
                gt_id=gid,
                component_label=label,
                nearest_distance_m=distance,
                extent_x=ex,
                extent_y=ey,
                point_count=counts.get(label, 0),
                detected_baseline=baseline_detected[(snapshot.name, gid)],
            )
        )
    return assignments


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    return "n/a" if value != value else f"{100 * value:.1f}%"  # value != value: NaN check


def print_contingency_table(assignments: list[ClusterAssignment], gate: float) -> None:
    over_detected = over_missed = under_detected = under_missed = 0
    for a in assignments:
        over = max(a.extent_x, a.extent_y) > gate
        if over and a.detected_baseline:
            over_detected += 1
        elif over and not a.detected_baseline:
            over_missed += 1
        elif not over and a.detected_baseline:
            under_detected += 1
        else:
            under_missed += 1
    print(f"\n=== Task 1: extent-over-gate ({gate} m) vs detected, n={len(assignments)} ===")
    print(f"{'':22s}{'detected':>10s}{'missed':>10s}")
    print(f"{'over gate':22s}{over_detected:>10d}{over_missed:>10d}")
    print(f"{'under gate':22s}{under_detected:>10d}{under_missed:>10d}")


def print_sweep_grid(
    results: dict[tuple[float, float], ScoreResult],
    cluster_max_extent_grid: tuple[float, ...],
    dbscan_eps_grid: tuple[float, ...],
) -> None:
    print("\n=== Task 2: recall / precision grid (cluster_max_extent_xy x dbscan_eps) ===")
    header = "cluster_max_extent_xy \\ dbscan_eps".ljust(26) + "".join(
        f"{eps:>16.2f}" for eps in dbscan_eps_grid
    )
    print(header)
    for cme in cluster_max_extent_grid:
        row = f"{cme:<26.2f}"
        for eps in dbscan_eps_grid:
            r = results[(cme, eps)]
            row += f"{_fmt_pct(r.recall) + '/' + _fmt_pct(r.precision):>16s}"
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS)
    parser.add_argument(
        "--cluster-max-extent-grid",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CLUSTER_MAX_EXTENT_GRID),
    )
    parser.add_argument(
        "--dbscan-eps-grid", type=str, default=",".join(str(v) for v in DEFAULT_DBSCAN_EPS_GRID)
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cluster_max_extent_grid = tuple(float(v) for v in args.cluster_max_extent_grid.split(","))
    dbscan_eps_grid = tuple(float(v) for v in args.dbscan_eps_grid.split(","))

    manifest = load_manifest(args.config)
    print(f"Loaded manifest: {len(manifest)} snapshots")
    snapshots = load_all_snapshots(args.dataset_root, args.transforms, manifest)
    total_gt = sum(len(s.gt_positions) for s in snapshots)
    print(f"Loaded {len(snapshots)} snapshots, {total_gt} approved GT blocks")

    defaults = DetectionParameters()
    baseline_detected = per_gt_detected(snapshots, defaults)
    baseline_score = score_snapshots(snapshots, defaults)
    print(
        f"\nBaseline (defaults): recall {baseline_score.detected_gt}/{baseline_score.total_gt} "
        f"({_fmt_pct(baseline_score.recall)}), precision {baseline_score.true_positive_accepted}/"
        f"{baseline_score.accepted} ({_fmt_pct(baseline_score.precision)})"
    )

    assignments: list[ClusterAssignment] = []
    for snap in snapshots:
        assignments.extend(analyze_clusters(snap, baseline_detected))
    print_contingency_table(assignments, defaults.cluster_max_extent_xy)

    header = (
        "snapshot".ljust(60) + "gt_id".ljust(14) + "extent_xy".ljust(12) + "n_pts".ljust(8) + "det"
    )
    print("\n" + header)
    for a in sorted(assignments, key=lambda a: -max(a.extent_x, a.extent_y)):
        if max(a.extent_x, a.extent_y) <= defaults.cluster_max_extent_xy:
            continue
        print(
            f"{a.snapshot:<60}{a.gt_id:<14}{max(a.extent_x, a.extent_y):<12.2f}"
            f"{a.point_count:<8d}{a.detected_baseline}"
        )

    sweep_results = {}
    for cme, eps in itertools.product(cluster_max_extent_grid, dbscan_eps_grid):
        params = DetectionParameters(cluster_max_extent_xy=cme, dbscan_eps=eps)
        sweep_results[(cme, eps)] = score_snapshots(snapshots, params)
    print_sweep_grid(sweep_results, cluster_max_extent_grid, dbscan_eps_grid)

    # Best cell: highest recall among cells that do not regress precision
    # below baseline; ties broken by precision, then by smaller eps/gate
    # (prefer the least aggressive change that achieves the recall gain).
    candidates = [
        (cme, eps, r)
        for (cme, eps), r in sweep_results.items()
        if r.precision >= baseline_score.precision
    ]
    pool = candidates or list((cme, eps, r) for (cme, eps), r in sweep_results.items())
    best_cme, best_eps, best_score = max(
        pool, key=lambda item: (item[2].recall, item[2].precision, -item[0], -item[1])
    )
    print(f"\n=== Task 3: best setting cluster_max_extent_xy={best_cme}, dbscan_eps={best_eps} ===")
    print(
        f"recall {best_score.detected_gt}/{best_score.total_gt} ({_fmt_pct(best_score.recall)}), "
        f"precision {best_score.true_positive_accepted}/{best_score.accepted} "
        f"({_fmt_pct(best_score.precision)})"
    )

    best_params = DetectionParameters(cluster_max_extent_xy=best_cme, dbscan_eps=best_eps)
    best_detected = per_gt_detected(snapshots, best_params)
    best_scene_scores = per_scene_scores(snapshots, best_params)
    baseline_scene_scores = per_scene_scores(snapshots, defaults)
    print("\nscene".ljust(45) + "baseline recall/prec".ljust(24) + "best recall/prec")
    for scene in sorted(baseline_scene_scores):
        b, n = baseline_scene_scores[scene], best_scene_scores[scene]
        print(
            f"{scene:<45}"
            f"{b.detected_gt}/{b.total_gt} {_fmt_pct(b.recall)}/{_fmt_pct(b.precision)}".ljust(24)
            + f"{n.detected_gt}/{n.total_gt} {_fmt_pct(n.recall)}/{_fmt_pct(n.precision)}"
        )

    gate = defaults.cluster_max_extent_xy
    over_gate_ids = {
        (a.snapshot, a.gt_id) for a in assignments if max(a.extent_x, a.extent_y) > gate
    }
    recovered = [key for key in over_gate_ids if not baseline_detected[key] and best_detected[key]]
    still_missed = [key for key in over_gate_ids if not best_detected[key]]
    lost = [key for key in baseline_detected if baseline_detected[key] and not best_detected[key]]
    print(f"\nOver-gate blocks recovered at best setting: {len(recovered)}/{len(over_gate_ids)}")
    for key in recovered:
        print(f"  recovered: {key}")
    print(f"Over-gate blocks still missed at best setting: {len(still_missed)}")
    for key in still_missed:
        print(f"  still missed: {key}")
    print(f"Previously-detected blocks lost at best setting: {len(lost)}")
    for key in lost:
        print(f"  lost: {key}")

    # Task 4: for merged components (>1 GT block sharing the nearest-component
    # label), report how many blocks share it and their pairwise centre
    # distances -- sizes a future splitter regardless of whether the sweep
    # recovered them.
    by_component: dict[tuple[str, int], list[ClusterAssignment]] = {}
    for a in assignments:
        if a.component_label == -1:
            continue
        by_component.setdefault((a.snapshot, a.component_label), []).append(a)
    print("\n=== Task 4: components shared by >1 GT block ===")
    shared = {k: v for k, v in by_component.items() if len(v) > 1}
    for (snapshot, label), members in sorted(shared.items()):
        snap = next(s for s in snapshots if s.name == snapshot)
        pos_by_id = dict(zip(snap.gt_ids, snap.gt_positions, strict=True))
        ids = [m.gt_id for m in members]
        dists = [
            float(np.linalg.norm(pos_by_id[a] - pos_by_id[b]))
            for a, b in itertools.combinations(ids, 2)
        ]
        print(
            f"{snapshot} label={label}: {len(members)} GT blocks {ids}, "
            f"extent={max(members[0].extent_x, members[0].extent_y):.2f} m, "
            f"pairwise centre distances={[round(d, 2) for d in dists]} m"
        )

    if args.json_out is not None:
        payload = {
            "baseline": {
                "recall": baseline_score.recall,
                "precision": baseline_score.precision,
                "detected_gt": baseline_score.detected_gt,
                "total_gt": baseline_score.total_gt,
                "accepted": baseline_score.accepted,
                "true_positive_accepted": baseline_score.true_positive_accepted,
            },
            "assignments": [vars(a) for a in assignments],
            "sweep": [
                {
                    "cluster_max_extent_xy": cme,
                    "dbscan_eps": eps,
                    "recall": r.recall,
                    "precision": r.precision,
                }
                for (cme, eps), r in sweep_results.items()
            ],
            "best": {"cluster_max_extent_xy": best_cme, "dbscan_eps": best_eps},
            "recovered": [list(k) for k in recovered],
            "still_missed": [list(k) for k in still_missed],
            "lost": [list(k) for k in lost],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
