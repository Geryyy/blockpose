"""Load the robot's real wall-assembly plans as detector test geometry.

`config/wall_plans.yaml` is exported from the ROS workspace and describes what
the robot is actually asked to build. Driving the synthetic suite from it rather
than from invented layouts keeps the benchmark honest about the target task.

Two things in that file contradict assumptions the rest of this package grew up
with, and both matter:

1. **Blocks are laid down.** `block_size` is ``[0.9, 0.6, 0.6]`` and course
   centres sit at z = 0.3 with a 0.6 m rise between courses, so the 0.9 m long
   axis is horizontal.

   `basic_pick_and_place` places its block at z = 0.45 rather than 0.3. That is
   **not** an upright block: it is the same laid block with 0.15 m of clearance,
   because the real ground is not at z = 0 and the terrain is uneven. Plan
   heights therefore carry a terrain allowance and are NOT exact multiples of the
   block height -- which is an assumption `detection._make_hypothesis` makes when
   it scores `height_consistency` against `layer * dims[2]`.

2. **Courses are not flush.** Neighbouring centres are 0.92 m apart for a 0.9 m
   block, i.e. a **2 cm gap**, and the second course is offset by 0.30 m rather
   than a half lap. 2 cm is far below the 10 cm gap the synthetic `assembly`
   family showed the detector can resolve, and well inside a voxel plus a
   clustering epsilon.

`gripper_yaw_offset_deg` is a manipulation concern and has no bearing on the
block's pose, so it is parsed and ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from blockpose.pose import BLOCK_DIMS
from blockpose.synth import BlockSpec

DEFAULT_PLANS = Path(__file__).resolve().parents[2] / "config" / "wall_plans.yaml"


@dataclass(frozen=True)
class WallPlan:
    """A resolved plan: block ids, their world placements, and the block size."""

    name: str
    block_size: tuple[float, float, float]
    ids: tuple[str, ...]
    specs: tuple[BlockSpec, ...]

    @property
    def centre(self) -> np.ndarray:
        """Centroid of the placed blocks, for aiming a sensor at the structure."""
        if not self.specs:
            return np.zeros(3)
        return np.array(
            [
                float(np.mean([spec.x for spec in self.specs])),
                float(np.mean([spec.y for spec in self.specs])),
                float(np.mean([spec.z for spec in self.specs])),
            ]
        )

    def sensor_params_kwargs(self, standoff: float = 6.0, height: float = 4.0) -> dict:
        """Sensor placed `standoff` metres back along -x, aimed at the structure.

        The plans live around x = -14, y = -3 in the ROS `world` frame, so a
        sensor cannot be left at the generator's default origin.
        """
        centre = self.centre
        return {
            "sensor_pos": (float(centre[0] - standoff), float(centre[1]), height),
            "look_at": (float(centre[0]), float(centre[1]), float(centre[2])),
            "dims": self.block_size,
        }


def _resolve_position(step: dict, placed: dict[str, np.ndarray]) -> np.ndarray:
    """World position for one sequence step.

    A step is either absolute, or relative to a previously placed block. The
    relative form carries `fallback_absolute_position` for when the world model
    has no estimate for the reference block; since this loader builds ground
    truth rather than running the robot, the reference is always available and
    the fallback is only used if the reference is genuinely missing.
    """
    if step.get("absolute_position") is not None:
        return np.asarray(step["absolute_position"], dtype=np.float64)
    reference = step.get("relative_to_world_model")
    if reference is not None and reference in placed:
        return placed[reference] + np.asarray(step.get("offset", (0, 0, 0)), dtype=np.float64)
    fallback = step.get("fallback_absolute_position")
    if fallback is None:
        raise ValueError(f"step {step.get('id')!r} has no resolvable position")
    return np.asarray(fallback, dtype=np.float64)


def load_wall_plans(path: str | Path = DEFAULT_PLANS) -> dict[str, WallPlan]:
    """Parse every plan in the YAML into resolved `WallPlan` objects."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    defaults = data.get("defaults") or {}
    block_size = tuple(float(v) for v in defaults.get("block_size", BLOCK_DIMS))
    if len(block_size) != 3:
        raise ValueError(f"defaults.block_size must have three entries, got {block_size}")

    plans: dict[str, WallPlan] = {}
    for name, plan in (data.get("wall_plans") or {}).items():
        placed: dict[str, np.ndarray] = {}
        ids: list[str] = []
        specs: list[BlockSpec] = []
        for step in plan.get("sequence") or []:
            identifier = str(step["id"])
            position = _resolve_position(step, placed)
            placed[identifier] = position
            ids.append(identifier)
            specs.append(
                BlockSpec(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    yaw_deg=float(step.get("yaw_deg", 0.0)),
                    dims=block_size,
                )
            )
        plans[name] = WallPlan(name, block_size, tuple(ids), tuple(specs))
    return plans


def neighbour_gaps(plan: WallPlan) -> list[float]:
    """Face-to-face gaps between blocks sharing a course, in metres.

    Reports what the detector actually has to resolve. For `example_wall` this
    is 0.02 m, which is the number that decides whether a wall is separable at
    all from a single scan.
    """
    gaps = []
    for index, first in enumerate(plan.specs):
        for second in plan.specs[index + 1 :]:
            if abs(first.z - second.z) > 1e-6:
                continue  # different course
            delta = np.array([second.x - first.x, second.y - first.y])
            axis = int(np.argmax(np.abs(delta)))
            extent = plan.block_size[axis]
            gap = float(abs(delta[axis]) - extent)
            if -1e-6 <= gap < extent:  # adjacent, not separated by another block
                gaps.append(gap)
    return sorted(gaps)
