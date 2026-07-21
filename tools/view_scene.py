#!/usr/bin/env python3
"""Quick 3D viewer for a YAML scene — iterate scene definitions fast.

Loads a scene from config/scenes/*.yaml, raycasts it, and shows the point cloud
(coloured by height) with ground-truth block boxes (green). Optionally overlays
the detector output (blue) with --detect.

Examples:
    python tools/view_scene.py config/scenes/wall.yaml            # interactive window
    python tools/view_scene.py config/scenes/wall.yaml --detect   # + detector overlay
    python tools/view_scene.py config/scenes/wall.yaml --save wall.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.detection import DetectionParams, detect_scene_sample  # noqa: E402
from blockpose.pose import BLOCK_DIMS, BlockPose  # noqa: E402
from blockpose.scenes import generate_from_config, load_scene_config  # noqa: E402

GT_COLOR = "#00e000"
DET_COLOR = "#1e78ff"


def _box_edges(center, R, dims):
    dims = np.asarray(dims, float)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (R @ (signs * dims / 2.0).T).T + np.asarray(center, float)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return [np.stack([corners[i], corners[j]]) for i, j in edges]


def _draw_boxes(ax, poses: list[BlockPose], color: str, label: str, dims) -> None:
    first = True
    for p in poses:
        for seg in _box_edges(p.position, p.rotation, dims):
            ax.plot(*seg.T, color=color, lw=2.0, label=label if first else None, zorder=5)
            first = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Scene YAML (e.g. config/scenes/wall.yaml)")
    parser.add_argument("--detect", action="store_true", help="Overlay detector output (blue).")
    parser.add_argument("--save", type=Path, default=None, help="Save a PNG instead of a window.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for noise/clutter placement.")
    parser.add_argument("--elev", type=float, default=32.0, help="View elevation (deg).")
    parser.add_argument("--azim", type=float, default=-72.0, help="View azimuth (deg).")
    parser.add_argument("--reach", type=float, default=None, help="Half-width of the view box (m).")
    parser.add_argument("--subsample", type=int, default=25000, help="Max points plotted.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_path = args.scene if args.scene.is_absolute() else REPO_ROOT / args.scene
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene not found: {scene_path}")

    import matplotlib

    if args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_scene_config(scene_path)
    sample = generate_from_config(cfg, np.random.default_rng(args.seed))
    pts, gt = sample.points, sample.ground_truth
    dims = cfg.specs[0].dims if cfg.specs else BLOCK_DIMS

    # Pass the full sample, not just points: without the sensor origin and rays
    # the detector loses range-adaptive density scaling and free-space evidence,
    # which on the storage_yard scene drops recall from 9/9 to 1/9.
    det = detect_scene_sample(sample, DetectionParams()) if args.detect else None
    msg = f"{cfg.name}: {len(pts)} pts, {len(gt)} blocks"
    if det is not None:
        msg += f", {len(det)} detected"
    print(msg)

    centers = np.array([p.position for p in gt])
    ctr = centers.mean(0) if len(centers) else pts.mean(0)
    if args.reach is not None:
        reach = args.reach
    else:
        spread = float(np.abs(centers[:, :2] - ctr[:2]).max()) if len(centers) else 1.0
        reach = spread + (5.0 if cfg.clutter is not None else 1.5)

    keep = (np.abs(pts[:, 0] - ctr[0]) < reach) & (np.abs(pts[:, 1] - ctr[1]) < reach)
    p = pts[keep]
    if len(p) > args.subsample:
        p = p[np.random.default_rng(0).choice(len(p), args.subsample, replace=False)]

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap="turbo", s=3, alpha=0.6, depthshade=False)
    _draw_boxes(ax, gt, GT_COLOR, "ground truth", dims)
    if det:
        _draw_boxes(ax, det, DET_COLOR, "detected", dims)

    zmax = max(3.6, float(p[:, 2].max()) if len(p) else 3.6)
    ax.set_xlim(ctr[0] - reach, ctr[0] + reach)
    ax.set_ylim(ctr[1] - reach, ctr[1] + reach)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((2 * reach, 2 * reach, zmax))
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title=msg)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if args.save is not None:
        out = args.save if args.save.is_absolute() else Path.cwd() / args.save
        plt.savefig(out, dpi=110)
        print(f"saved -> {out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
