#!/usr/bin/env python3
"""Repeatable native C++ cuboid-refinement characterization benchmark.

This is a measurement gate, not a numeric regression threshold: the two
solvers intentionally use different trust-region implementations.  It samples
12 deterministic seeds per noise level by default and reports error/guard
statistics for fully observed isolated blocks, isolated blocks without their
long-axis end caps, and a wall-row target whose end caps are occluded by its
neighbours.  The latter two expose the known near-null long-axis direction.

Run after building the nanobind extension:
    /opt/venv/bin/python3 tools/benchmark_refinement.py
    /opt/venv/bin/python3 tools/benchmark_refinement.py --seeds 2 --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from blockpose.pose import BlockPose

DIMS = np.array((0.9, 0.6, 0.6), dtype=np.float64)
NOISE_LEVELS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)


@dataclass(frozen=True)
class Scenario:
    name: str
    visible_faces: tuple[str, ...]
    description: str


SCENARIOS = (
    Scenario("isolated_full", ("+x", "-x", "+y", "-y", "+z"), "isolated block"),
    Scenario(
        "isolated_missing_endcaps",
        ("+y", "-y", "+z"),
        "isolated block; both long-axis end caps hidden",
    ),
    Scenario(
        "wall_row_occluded",
        ("+y", "+z"),
        "middle wall-row block; neighbours occlude both end caps and rear face",
    ),
)


def _rotation(yaw_deg: float = 17.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    return np.array(
        ((np.cos(yaw), -np.sin(yaw), 0.0), (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0))
    )


def _face_points(face: str, steps: int = 13) -> np.ndarray:
    """Uniform, deterministic samples over a named local cuboid face."""
    half = DIMS / 2.0
    axes = {"x": 0, "y": 1, "z": 2}
    sign, axis_name = face[0], face[1]
    fixed_axis = axes[axis_name]
    free_axes = [axis for axis in range(3) if axis != fixed_axis]
    # Do not put samples on face boundaries: a point exactly on a missing
    # end-cap's perimeter would leak long-axis information through the SDF's
    # corner subgradient and hide the intended partial-visibility null space.
    grid = np.linspace(-0.75, 0.75, steps)
    points: list[np.ndarray] = []
    for first in grid:
        for second in grid:
            point = np.zeros(3)
            point[fixed_axis] = (1.0 if sign == "+" else -1.0) * half[fixed_axis]
            point[free_axes[0]] = first * half[free_axes[0]]
            point[free_axes[1]] = second * half[free_axes[1]]
            points.append(point)
    return np.asarray(points)


def scenario_points(
    scenario: Scenario, noise_std: float, seed: int
) -> tuple[np.ndarray, BlockPose]:
    """Return noisy visible target returns and its exact pose for one trial."""
    truth = BlockPose(np.array((1.1, -0.25, 0.3)), _rotation(), dims=tuple(DIMS))
    local = np.concatenate([_face_points(face) for face in scenario.visible_faces])
    points = truth.position + local @ truth.rotation.T
    if noise_std:
        points += np.random.default_rng(seed).normal(0.0, noise_std, size=points.shape)
    return points, truth


def _cpp_refine(
    points: np.ndarray, initial: BlockPose
) -> tuple[BlockPose, bool, dict[str, object]]:
    try:
        from blockpose import _core
    except ImportError as exc:  # pragma: no cover - command environment failure
        message = "build and install the blockpose._core nanobind extension first"
        raise RuntimeError(message) from exc
    result = _core.refine_pose(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(initial.position, dtype=np.float64),
        np.ascontiguousarray(initial.rotation, dtype=np.float64),
        np.ascontiguousarray(initial.dims, dtype=np.float64),
    )
    pose = BlockPose(result["position"], result["rotation"], dims=initial.dims)
    return pose, result["refined"], result["diagnostics"]


def _record(
    solver: str,
    pose: BlockPose,
    refined: bool,
    diagnostics: dict[str, object],
    truth: BlockPose,
) -> dict[str, object]:
    error = pose.position - truth.position
    long_axis_error = float(abs(error @ truth.rotation[:, 0]))
    return {
        "solver": solver,
        "refined": bool(refined),
        "translation_error_m": float(np.linalg.norm(error)),
        "long_axis_error_m": long_axis_error,
        "solver_translation_m": float(diagnostics["translation_norm"] or 0.0),
        "guard_accepted": bool(diagnostics["guard_accepted"]),
    }


def run_benchmark(
    seeds: int = 12, noise_levels: Iterable[float] = NOISE_LEVELS
) -> list[dict[str, object]]:
    """Run every scenario/seed/noise trial and return JSON-serializable records."""
    records: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for noise_std in noise_levels:
            for seed in range(seeds):
                points, truth = scenario_points(scenario, noise_std, seed)
                # The known failure is movement along local +x, so seed that exact
                # under-constrained direction rather than conflating it with yaw.
                initial = BlockPose(
                    truth.position + truth.rotation[:, 0] * 0.04,
                    truth.rotation,
                    dims=truth.dims,
                )
                cpp_pose, cpp_refined, cpp_diag = _cpp_refine(points, initial)
                record = _record("cpp_lm", cpp_pose, cpp_refined, cpp_diag, truth)
                record.update(scenario=scenario.name, noise_std_m=noise_std, seed=seed)
                records.append(record)
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate metrics without hiding rejected optimizations."""
    result: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for noise_std in NOISE_LEVELS:
            for solver in ("cpp_lm",):
                matches = [
                    entry
                    for entry in records
                    if entry["scenario"] == scenario.name
                    and entry["noise_std_m"] == noise_std
                    and entry["solver"] == solver
                ]
                if not matches:
                    continue
                result.append(
                    {
                        "scenario": scenario.name,
                        "noise_std_m": noise_std,
                        "solver": solver,
                        "trials": len(matches),
                        "refined": sum(bool(entry["refined"]) for entry in matches),
                        "long_axis_error_p95_m": float(
                            np.percentile([entry["long_axis_error_m"] for entry in matches], 95)
                        ),
                        "translation_error_p95_m": float(
                            np.percentile([entry["translation_error_m"] for entry in matches], 95)
                        ),
                    }
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--json", type=Path, help="write complete records and summary to this path")
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    records = run_benchmark(args.seeds)
    summary = summarize(records)
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"records": records, "summary": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
