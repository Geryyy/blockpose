"""Declarative YAML scene definitions for synthetic block scenarios.

Lets you describe storage/assembly scenarios (isolated staging blocks, stacks,
walls, ...) in a small YAML file and generate the raycast cloud + ground truth
from it. Iterate the YAML and re-render with `tools/view_scene.py`.

Schema (all sections optional except at least one of `blocks`/`patterns`):

    name: wall_partial
    dims: [0.6, 0.6, 0.9]          # default block size (x, y, z); long axis is z
    sensor:
      pos: [-6.0, -1.5, 4.5]
      look_at: [1.5, 0.0, 1.3]
      az_fov_deg: 120
      el_fov_deg: 80
      ang_res_deg: 0.25
      range_noise_std: 0.01
    clutter: {trees: true, container: true, logs: true}   # or `true`, or omit for none
    blocks:                        # explicit blocks
      - {x: 0.0, y: 2.0, yaw_deg: 30, pitch_deg: 2}
    patterns:                      # generators expanded into blocks
      - {type: row,   count: 4, start: [0, 0],  step: [0, 1.0], yaw_deg: 10}
      - {type: stack, count: 3, base:  [2, 0],  yaw_deg: 0}
      - {type: wall,  origin: [0, -2], courses: 3, length: 5, axis: y, running_bond: true}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from blockpose.pose import BlockPose
from blockpose.synth import (
    BLOCK_DIMS,
    BlockSpec,
    ClutterParams,
    SceneSample,
    SynthParams,
    generate_scene,
)


@dataclass
class SceneConfig:
    """A parsed scene: sensor params, optional clutter, and the expanded block specs."""

    name: str
    params: SynthParams
    clutter: ClutterParams | None
    specs: list[BlockSpec]


def _parse_sensor(d: dict, dims: tuple[float, float, float]) -> SynthParams:
    kw: dict = {"dims": dims}
    if "pos" in d:
        kw["sensor_pos"] = tuple(d["pos"])
    if "look_at" in d and d["look_at"] is not None:
        kw["look_at"] = tuple(d["look_at"])
    for key in (
        "az_fov_deg",
        "el_fov_deg",
        "ang_res_deg",
        "range_noise_std",
        "range_noise_slope",
        "dropout",
        "stripe_dropout",
        "outlier_fraction",
        "outlier_radius",
        "ground",
        "ground_size",
    ):
        if key in d:
            kw[key] = d[key]
    return SynthParams(**kw)


def _parse_clutter(c) -> ClutterParams | None:
    if not c:  # None, False, or empty dict -> no clutter
        return None
    if c is True:
        return ClutterParams()
    return ClutterParams(**c)


def _parse_blocks(items: list[dict], dims: tuple[float, float, float]) -> list[BlockSpec]:
    specs = []
    for item in items:
        b = dict(item)
        b["dims"] = tuple(b["dims"]) if "dims" in b else dims
        specs.append(BlockSpec(**b))
    return specs


def _expand_pattern(p: dict, dims: tuple[float, float, float]) -> list[BlockSpec]:
    """Expand a `row` / `stack` / `wall` pattern into explicit BlockSpecs."""
    t = p["type"]
    yaw = float(p.get("yaw_deg", 0.0))
    width, height = dims[0], dims[2]  # cross-section along the run, block height
    specs: list[BlockSpec] = []

    if t == "row":
        sx, sy = p["start"]
        dx, dy = p.get("step", [0.0, width])
        for i in range(int(p["count"])):
            specs.append(BlockSpec(x=sx + i * dx, y=sy + i * dy, yaw_deg=yaw, dims=dims))

    elif t == "stack":
        bx, by = p["base"]
        for i in range(int(p["count"])):
            specs.append(BlockSpec(x=bx, y=by, z=height * (i + 0.5), yaw_deg=yaw, dims=dims))

    elif t == "wall":
        ox, oy = p["origin"]
        courses = int(p["courses"])
        length = int(p["length"])
        axis = p.get("axis", "x")
        gap = float(p.get("gap", 0.0))
        bond = bool(p.get("running_bond", False))
        step = width + gap
        for c in range(courses):
            offset = step / 2.0 if (bond and c % 2 == 1) else 0.0
            for j in range(length):
                d = j * step + offset
                x = ox + (d if axis == "x" else 0.0)
                y = oy + (d if axis == "y" else 0.0)
                specs.append(BlockSpec(x=x, y=y, z=height * (c + 0.5), yaw_deg=yaw, dims=dims))

    else:
        raise ValueError(f"unknown pattern type: {t!r}")

    return specs


def load_scene_config(path: str | Path) -> SceneConfig:
    """Parse a scene YAML into a `SceneConfig` (does not raycast)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    dims = tuple(data.get("dims", BLOCK_DIMS))

    params = _parse_sensor(data.get("sensor") or {}, dims)
    clutter = _parse_clutter(data.get("clutter"))

    specs = _parse_blocks(data.get("blocks") or [], dims)
    for pattern in data.get("patterns") or []:
        specs.extend(_expand_pattern(pattern, dims))

    if not specs:
        raise ValueError(f"scene {path} defines no blocks (need `blocks` and/or `patterns`)")

    return SceneConfig(
        name=str(data.get("name", path.stem)), params=params, clutter=clutter, specs=specs
    )


def generate_from_config(
    cfg: SceneConfig, rng: np.random.Generator | None = None
) -> SceneSample:
    """Raycast a parsed scene -> (pts_world (N,3) float64, list[BlockPose])."""
    return generate_scene(cfg.specs, cfg.params, rng, clutter=cfg.clutter)


def load_and_generate(
    path: str | Path, rng: np.random.Generator | None = None
) -> tuple[SceneConfig, np.ndarray, list[BlockPose]]:
    """Convenience: parse a scene YAML and raycast it -> (cfg, pts, gt)."""
    cfg = load_scene_config(path)
    pts, gt = generate_from_config(cfg, rng)
    return cfg, pts, gt
