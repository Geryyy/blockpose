#!/usr/bin/env python3
"""CLI: run the prototype block detector on one or more registration snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running without installation: add repo python/ to path
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

from blockpose.detection import DetectionParams, SensorContext, detect_blocks  # noqa: E402
from blockpose.evaluate import evaluate  # noqa: E402
from blockpose.io import load_snapshot, load_T_world_seyond, load_tf_snapshot  # noqa: E402

_DEFAULT_TRANSFORM = _REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
_DEFAULT_SNAPSHOTS = _REPO_ROOT / "data" / "registration_snapshots"


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    sep = "  "
    header_line = sep.join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(sep.join(str(row[h]).ljust(widths[h]) for h in headers))


def _run_snapshot(
    snap_dir: Path,
    config_path: Path,
    params: DetectionParams,
    visualize: bool,
) -> dict:
    tf_path = snap_dir / "tf.yaml"
    # Prefer the snapshot's own tf.yaml (crane joint state at capture time)
    # over the static reference config: they can differ by several metres,
    # which otherwise puts the cloud nowhere near the recorded ground truth.
    if tf_path.is_file():
        T_world_cloud = load_tf_snapshot(tf_path).T_world_cloud
    else:
        T_world_cloud = load_T_world_seyond(config_path)
    pts_world, gt = load_snapshot(snap_dir, config_path, T_world_cloud=T_world_cloud)

    # Ray/range evidence from the returns themselves (no true misses available
    # from a plain point cloud), which activates the detector's visibility and
    # free-space-violation scoring -- without it, real-data orientation
    # disambiguation for non-square (laid) blocks is markedly worse.
    sensor_context = SensorContext.from_points(T_world_cloud[:3, 3], pts_world)
    detections = detect_blocks(pts_world, params, sensor_context=sensor_context)
    result = evaluate(detections, gt)

    if visualize:
        _visualize(pts_world, detections, gt, snap_dir.name)

    return {
        "snapshot": snap_dir.name,
        "gt_blocks": len(gt),
        "detected": len(detections),
        "matched": result.matched,
        "fp": result.unmatched_det,
        "fn": result.unmatched_gt,
        "mean_t_err_m": f"{result.mean_translation_m:.3f}" if result.matched else "—",
        "mean_r_err_deg": f"{result.mean_rotation_deg:.1f}" if result.matched else "—",
    }


def _visualize(
    pts_world: np.ndarray,
    detections,
    gt,
    title: str,
) -> None:
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print(
            "[visualize] No display detected — skipping Open3D GUI. "
            "Use visualize_snapshot.py --save instead.",
            file=sys.stderr,
        )
        return
    try:
        import open3d as o3d
    except ImportError:
        print("[visualize] open3d not available", file=sys.stderr)
        return

    geoms = []

    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geoms.append(pcd)

    # Detected poses (blue)
    for pose in detections:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        T = np.eye(4)
        T[:3, :3] = pose.rotation
        T[:3, 3] = pose.position
        frame.transform(T)
        frame.paint_uniform_color([0.0, 0.3, 0.8])
        geoms.append(frame)

    # GT poses (green)
    for pose in gt:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        T = np.eye(4)
        T[:3, :3] = pose.rotation
        T[:3, 3] = pose.position
        frame.transform(T)
        frame.paint_uniform_color([0.0, 0.7, 0.2])
        geoms.append(frame)

    o3d.visualization.draw_geometries(geoms, window_name=title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect concrete blocks in LiDAR registration snapshots."
    )
    parser.add_argument(
        "snapshots",
        nargs="*",
        type=Path,
        help="Snapshot directories. If omitted, all snapshots under "
        "data/registration_snapshots/ are used.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_TRANSFORM,
        help="Path to transforms YAML (default: config/transforms_pzs100_initial_pose_1.yaml)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="Voxel downsample size in meters (default: 0.04)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.4,
        help="Minimum evidence score to keep a hypothesis (default: 0.4). Real "
        "snapshots can contain non-block clutter (e.g. the crane/gripper hardware) "
        "that still produces a plausible-shaped hypothesis; measured on real "
        "snapshots, spurious hypotheses like that score under ~0.2 while every "
        "confirmed block scores over ~0.65, so 0.4 is a conservative cut. "
        "Pass 0.0 to disable (matches every prior/synthetic default).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open Open3D viewer for each snapshot.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout.",
    )
    args = parser.parse_args()

    if args.snapshots:
        snap_dirs = [Path(p) for p in args.snapshots]
    else:
        snap_dirs = sorted(_DEFAULT_SNAPSHOTS.iterdir())

    params = DetectionParams(voxel_size=args.voxel_size, min_score=args.min_score)

    rows = []
    for snap_dir in snap_dirs:
        if not snap_dir.is_dir():
            print(f"[warn] {snap_dir} is not a directory, skipping", file=sys.stderr)
            continue
        try:
            row = _run_snapshot(snap_dir, args.config, params, args.visualize)
            rows.append(row)
        except Exception as exc:
            print(f"[error] {snap_dir.name}: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"\n{'='*60}")
        print("Block Detection Results")
        print(f"{'='*60}")
        _print_table(rows)

        if rows:
            matched_total = sum(int(r["matched"]) for r in rows)
            gt_total = sum(int(r["gt_blocks"]) for r in rows)
            fp_total = sum(int(r["fp"]) for r in rows)
            fn_total = sum(int(r["fn"]) for r in rows)
            print(f"\nAggregate: {matched_total}/{gt_total} GT matched, "
                  f"{fp_total} FP, {fn_total} FN across {len(rows)} snapshots")


if __name__ == "__main__":
    main()
