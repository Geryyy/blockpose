#!/usr/bin/env python3
"""Measure the real sensor's noise and sampling characteristics from snapshots.

Produces the numbers hard-coded as `MEASURED_*` in `blockpose/synth.py`, so the
synthetic generator's parameters are traceable to data instead of guessed. Run
this after re-exporting snapshots and update the constants if they move.

Method:
  * Range noise -- fit a local plane in a 0.15 m neighbourhood, keep only
    locally planar patches, and take the MAD-sigma of point-to-plane residuals.
    Residuals are debiased by sqrt(k/(k-3)) because the plane is fitted to the
    same noisy points it is measured against.
  * Angular sampling -- reports BOTH the nearest-neighbour angular gap (spacing
    within a scan line) and the areal-equivalent grid pitch sqrt(1/density).
    These differ by ~2x for a non-repetitive pattern. The areal figure is the
    one that governs how many points land on a block face, so it is the one the
    generator's isotropic ray grid must match.

Examples:
    python tools/measure_sensor_model.py
    python tools/measure_sensor_model.py --glob 'data/**/cloud.pcd' --samples 12000
"""

from __future__ import annotations

import argparse
import glob as globlib

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

NEIGHBOUR_RADIUS = 0.15
MIN_NEIGHBOURS = 15
PLANARITY_MAX = 0.25  # smallest/middle singular value; lower is flatter


def _local_plane_residuals(
    points: np.ndarray, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (range, cos_incidence, debiased residual) for locally planar patches."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    ranges = np.linalg.norm(points, axis=1)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(points), size=min(samples, len(points)), replace=False)

    rows = []
    for index in picks:
        count, neighbours, _ = tree.search_radius_vector_3d(cloud.points[index], NEIGHBOUR_RADIUS)
        if count < MIN_NEIGHBOURS:
            continue
        patch = points[np.asarray(neighbours)]
        centroid = patch.mean(axis=0)
        _, singular, basis = np.linalg.svd(patch - centroid, full_matrices=False)
        if singular[2] / max(singular[1], 1e-9) > PLANARITY_MAX:
            continue
        normal = basis[2]
        direction = points[index] / max(ranges[index], 1e-9)
        residual = abs(float((points[index] - centroid) @ normal))
        rows.append(
            (
                ranges[index],
                abs(float(normal @ direction)),
                residual * np.sqrt(count / max(count - 3, 1)),
            )
        )
    if not rows:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    table = np.asarray(rows)
    return table[:, 0], table[:, 1], table[:, 2]


def _sigma(values: np.ndarray) -> float:
    return float(1.4826 * np.median(values)) if len(values) else float("nan")


def analyse(path: str, samples: int, seed: int) -> None:
    points = np.asarray(o3d.io.read_point_cloud(path).points)
    points = points[np.isfinite(points).all(axis=1)]
    ranges = np.linalg.norm(points, axis=1)
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(points[:, 2] / np.maximum(ranges, 1e-9), -1, 1)))

    print(f"\n=== {path} ===")
    print(f"points {len(points)}   range p05={np.percentile(ranges, 5):.1f} "
          f"p50={np.percentile(ranges, 50):.1f} p95={np.percentile(ranges, 95):.1f} m")
    print(f"fov    azimuth {np.ptp(azimuth):.1f} deg   elevation {np.ptp(elevation):.1f} deg "
          f"({elevation.min():.1f} .. {elevation.max():.1f})")

    # Areal density over the solid angle actually covered.
    hist, _, _ = np.histogram2d(
        azimuth,
        elevation,
        bins=[np.arange(azimuth.min(), azimuth.max(), 0.5),
              np.arange(elevation.min(), elevation.max(), 0.5)],
    )
    covered = float((hist > 0).sum()) * 0.25
    density = len(points) / covered if covered else float("nan")

    directions = points / np.maximum(ranges, 1e-9)[:, None]
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(points), size=min(6000, len(points)), replace=False)
    chord, _ = cKDTree(directions).query(directions[probe], k=2)
    gap = np.degrees(2 * np.arcsin(np.clip(chord[:, 1] / 2, 0, 1)))

    print(f"sampling nearest-neighbour gap p50={np.median(gap):.3f} deg "
          f"(within scan line)")
    print(f"         areal density {density:.1f} pts/deg^2 -> equivalent grid pitch "
          f"{1 / np.sqrt(density):.3f} deg  <-- match the generator to THIS")

    sample_range, cos_incidence, residual = _local_plane_residuals(points, samples, seed)
    if not len(residual):
        print("no locally planar patches found")
        return

    print("range noise (near-normal incidence, cos > 0.7):")
    near_normal = cos_incidence > 0.7
    for low in range(0, 30, 5):
        window = near_normal & (sample_range >= low) & (sample_range < low + 5)
        if window.sum() < 40:
            continue
        print(f"  {low:2d}-{low + 5:2d} m  n={window.sum():5d}  "
              f"sigma={_sigma(residual[window]) * 100:5.2f} cm")

    print("range noise vs incidence (4-12 m):")
    band = (sample_range >= 4) & (sample_range < 12)
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        window = band & (cos_incidence >= low) & (cos_incidence < low + 0.2)
        if window.sum() < 40:
            continue
        off_normal = np.degrees(np.arccos(np.clip(low + 0.1, 0, 1)))
        print(f"  ~{off_normal:4.1f} deg off-normal  n={window.sum():5d}  "
              f"sigma={_sigma(residual[window]) * 100:5.2f} cm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="data/registration_snapshots/*/cloud.pcd")
    parser.add_argument("--samples", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(globlib.glob(args.glob, recursive=True))
    if not paths:
        print(f"no clouds matched {args.glob!r}")
        return 1
    for path in paths:
        analyse(path, args.samples, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
