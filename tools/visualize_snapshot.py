#!/usr/bin/env python3
"""Visualize a registration snapshot in world frame using Rerun.

Shows the world-frame point cloud, the sensor (seyond) origin, ground-truth
block boxes and (optionally) detected block boxes in a single 3D scene, so the
world transform can be inspected visually.

Examples:
    # Live viewer (needs a display / rerun viewer):
    python tools/visualize_snapshot.py data/registration_snapshots/1783428578_018899200_seq1

    # Headless: write an .rrd you can open later with `rerun scene.rrd`:
    python tools/visualize_snapshot.py <snap> --save scene.rrd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

from blockpose.detection import DetectionParams, detect_blocks  # noqa: E402
from blockpose.io import load_snapshot, load_T_world_seyond  # noqa: E402
from blockpose.pose import rotmat_to_quat_xyzw  # noqa: E402

_DEFAULT_TRANSFORM = _REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
BLOCK_DIMS = (0.6, 0.6, 0.9)  # x, y, z (m)


def _height_colors(z: np.ndarray) -> np.ndarray:
    """Map Z values to an RGB uint8 array via a simple viridis-like ramp."""
    from matplotlib import colormaps

    zmin, zmax = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    norm = np.clip((z - zmin) / max(zmax - zmin, 1e-6), 0.0, 1.0)
    rgba = colormaps["viridis"](norm)
    return (rgba[:, :3] * 255).astype(np.uint8)


def _log_box(path: str, position, quat_xyzw, color) -> None:
    rr.log(
        path,
        rr.Boxes3D(
            centers=[position],
            half_sizes=[[BLOCK_DIMS[0] / 2, BLOCK_DIMS[1] / 2, BLOCK_DIMS[2] / 2]],
            quaternions=[rr.Quaternion(xyzw=quat_xyzw)],
            colors=[color],
        ),
    )


def _log_axes(path: str, T: np.ndarray, length: float = 0.5) -> None:
    """Log a coordinate frame (3 RGB arrows) from a 4x4 transform."""
    origin = T[:3, 3]
    R = T[:3, :3]
    rr.log(
        path,
        rr.Arrows3D(
            origins=[origin, origin, origin],
            vectors=[R[:, 0] * length, R[:, 1] * length, R[:, 2] * length],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path, help="Snapshot directory")
    ap.add_argument("--config", type=Path, default=_DEFAULT_TRANSFORM)
    ap.add_argument("--detect", action="store_true", help="Also run and show the detector")
    ap.add_argument("--save", type=Path, default=Path("snapshot.rrd"),
                    help="Write an .rrd file (default: snapshot.rrd in CWD)")
    ap.add_argument("--spawn", action="store_true",
                    help="Launch the Rerun viewer GUI (requires a display).")
    args = ap.parse_args()

    pts, gt = load_snapshot(args.snapshot, args.config)
    T = load_T_world_seyond(args.config)

    rr.init("blockpose_snapshot", spawn=args.spawn)
    if not args.spawn:
        rr.save(str(args.save))

    # World frame: Z up, right-handed
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    _log_axes("world/origin_axes", np.eye(4), length=1.0)

    # Point cloud coloured by height
    rr.log("world/points", rr.Points3D(pts, colors=_height_colors(pts[:, 2]), radii=0.01))

    # Sensor (seyond) origin + axes
    _log_axes("world/sensor_axes", T, length=0.8)
    rr.log(
        "world/sensor_origin",
        rr.Points3D([T[:3, 3]], colors=[[255, 255, 0]], radii=0.1, labels=["seyond"]),
    )

    # Ground-truth boxes (green)
    for i, pose in enumerate(gt):
        q = rotmat_to_quat_xyzw(pose.rotation)
        _log_box(f"world/gt/block_{i}", pose.position, q, [0, 200, 60])
        rr.log(
            f"world/gt/block_{i}_label",
            rr.Points3D([pose.position], colors=[[0, 200, 60]], radii=0.06,
                        labels=[f"GT{i} z={pose.position[2]:.2f}"]),
        )

    # Detections (blue)
    if args.detect:
        det = detect_blocks(pts, DetectionParams())
        for i, pose in enumerate(det):
            q = rotmat_to_quat_xyzw(pose.rotation)
            _log_box(f"world/det/block_{i}", pose.position, q, [40, 120, 255])
        print(f"Logged {len(det)} detections")

    print(f"Point cloud: {len(pts)} pts, {len(gt)} GT blocks")
    print(f"  X [{pts[:,0].min():.1f},{pts[:,0].max():.1f}]  "
          f"Y [{pts[:,1].min():.1f},{pts[:,1].max():.1f}]  "
          f"Z [{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")
    print(f"  sensor origin (world): {T[:3,3]}")
    if not args.spawn:
        print(f"Saved -> {args.save}   (open with:  rerun {args.save})")


if __name__ == "__main__":
    main()
