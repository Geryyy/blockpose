"""Deterministic synthetic scenario registry for detector diagnostics.

These scenarios broaden measurement without changing the small Phase-A gate.
Each case has a stable name and family so reports can be compared by failure
mode rather than hiding regressions in a single aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blockpose.pose import BLOCK_DIMS_UPRIGHT
from blockpose.synth import (
    BlockSpec,
    ClutterParams,
    SceneSample,
    SynthParams,
    generate_scene,
    seyond_measured,
)


def _upright_spec(x: float, y: float, **kwargs) -> BlockSpec:
    """A BlockSpec for the generic (non-wall-plan) registry below.

    Every scenario here models a free-standing/staging block, not a wall
    course, so it is pinned to the upright orientation explicitly rather than
    riding on whatever blockpose.pose.BLOCK_DIMS happens to default to.
    """
    kwargs.setdefault("dims", BLOCK_DIMS_UPRIGHT)
    return BlockSpec(x, y, **kwargs)


def _upright_params(**kwargs) -> SynthParams:
    """SynthParams for the generic registry, pinned upright (see _upright_spec)."""
    kwargs.setdefault("dims", BLOCK_DIMS_UPRIGHT)
    return SynthParams(**kwargs)


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    specs: tuple[BlockSpec, ...]
    params: SynthParams
    clutter: ClutterParams | None = None

    def generate(self, seed: int) -> SceneSample:
        return generate_scene(
            list(self.specs), self.params, np.random.default_rng(seed), self.clutter
        )


def scenario_registry() -> tuple[Scenario, ...]:
    """Return the stable diagnostic matrix definition.

    Every scenario below (except `wall_plan_scenarios()`) models a
    free-standing/staging block, so it is built via `_upright_spec`/
    `_upright_params`, pinned to BLOCK_DIMS_UPRIGHT explicitly.
    """
    base = _upright_spec(0.0, 0.0, yaw_deg=23.0)
    return (
        Scenario("isolated_near", "isolated_range", (base,), _upright_params()),
        Scenario(
            "isolated_far",
            "isolated_range",
            (_upright_spec(8.0, 0.0, yaw_deg=61.0),),
            _upright_params(sensor_pos=(-5.0, 0.0, 4.0), look_at=(8.0, 0.0, 0.45)),
        ),
        Scenario(
            "isolated_tilt_7deg",
            "tilt",
            (_upright_spec(0.0, 0.0, yaw_deg=37.0, pitch_deg=7.0, roll_deg=-5.0),),
            _upright_params(),
        ),
        Scenario(
            "isolated_noise_3cm",
            "noise",
            (base,),
            _upright_params(range_noise_std=0.03, range_noise_slope=0.001),
        ),
        # Two dropout modes, because they fail differently. Random thinning
        # degrades density uniformly; a contiguous stripe punches a structured
        # gap that fragments a fixed-eps DBSCAN cluster into sub-threshold
        # pieces. A single stripe scenario also makes the family metric bimodal
        # (the band either lands on the target or does not), so both are kept.
        Scenario(
            "isolated_dropout_60pct_random",
            "dropout",
            (base,),
            _upright_params(dropout=0.6),
        ),
        Scenario(
            "isolated_dropout_40pct",
            "dropout",
            (base,),
            _upright_params(dropout=0.4, stripe_dropout=0.1),
        ),
        Scenario(
            "three_separated",
            "multi_block",
            tuple(
                _upright_spec(0.0, y, yaw_deg=15.0 + 20.0 * i)
                for i, y in enumerate((-1.3, 0.0, 1.3))
            ),
            _upright_params(az_fov_deg=100.0),
        ),
        Scenario(
            "touching_row",
            "touching",
            tuple(_upright_spec(0.0, 0.6 * i, yaw_deg=0.0) for i in range(3)),
            _upright_params(az_fov_deg=100.0),
        ),
        Scenario(
            "two_stack",
            "stack",
            (_upright_spec(0.0, 0.0, z=0.45), _upright_spec(0.0, 0.0, z=1.35)),
            _upright_params(look_at=(0.0, 0.0, 0.9)),
        ),
        Scenario(
            "offset_two_stack",
            "stack",
            (_upright_spec(0.0, 0.0, z=0.45), _upright_spec(0.25, 0.0, z=1.35)),
            _upright_params(look_at=(0.1, 0.0, 0.9)),
        ),
        Scenario(
            "offset_two_stack_yaw30",
            "stack",
            (
                _upright_spec(0.0, 0.0, z=0.45, yaw_deg=30.0),
                _upright_spec(0.34641016, 0.2, z=1.35, yaw_deg=30.0),
            ),
            _upright_params(look_at=(0.15, 0.1, 0.9)),
        ),
        Scenario(
            "near_field_hard_negative",
            "clutter",
            (base,),
            _upright_params(az_fov_deg=100.0),
            ClutterParams(
                trees=False,
                container=False,
                logs=False,
                near_field_boxes=2,
                near_field_offset=0.85,
            ),
        ),
        Scenario(
            "hard_negative_only",
            "hard_negative",
            (),
            _upright_params(az_fov_deg=100.0, look_at=(0.0, 0.0, 0.45)),
            ClutterParams(
                trees=False,
                container=False,
                logs=False,
                near_field_boxes=2,
                near_field_offset=0.7,
            ),
        ),
        Scenario(
            "elongated_full_height_negative",
            "hard_negative",
            (),
            _upright_params(az_fov_deg=100.0, look_at=(0.0, 0.0, 0.45)),
            ClutterParams(
                trees=False,
                container=False,
                logs=False,
                near_field_boxes=1,
                near_field_offset=0.0,
                near_field_dims=(0.6, 1.8, 0.9),
            ),
        ),
        # Standoff ladder. The real snapshots contain returns out to ~31 m, but
        # the detector was only ever measured to ~13 m ("isolated_far"), which
        # sits just inside the point-density cliff. These make the usable range
        # envelope an explicit, tracked number instead of an unknown.
        *(
            Scenario(
                f"standoff_{distance}m",
                "standoff",
                (_upright_spec(float(distance) - 5.0, 0.0, yaw_deg=41.0),),
                _upright_params(
                    sensor_pos=(-5.0, 0.0, 4.0),
                    look_at=(float(distance) - 5.0, 0.0, 0.45),
                ),
            )
            for distance in (10, 15, 20, 25)
        ),
        # Uneven ground. The detector fits a single RANSAC ground plane and
        # removes everything within 0.15 m of it; a flat synthetic plane never
        # tested that tolerance.
        Scenario(
            "sloped_ground_4deg",
            "uneven_ground",
            (_upright_spec(0.0, 0.0, yaw_deg=17.0),),
            _upright_params(ground_slope_deg=4.0),
        ),
        Scenario(
            "undulating_ground",
            "uneven_ground",
            tuple(
                _upright_spec(0.0, y, yaw_deg=25.0 + 30.0 * i)
                for i, y in enumerate((-1.3, 0.0, 1.3))
            ),
            _upright_params(az_fov_deg=100.0, ground_undulation=0.10, ground_roughness=0.015),
        ),
        Scenario(
            "sloped_rough_ground",
            "uneven_ground",
            (_upright_spec(0.0, 0.0, yaw_deg=52.0, pitch_deg=2.5),),
            _upright_params(ground_slope_deg=3.0, ground_undulation=0.06, ground_roughness=0.02),
        ),
        # Mixed pixels: the dominant real LiDAR artifact at depth edges.
        Scenario(
            "mixed_pixels",
            "sensor_artifacts",
            (_upright_spec(0.0, 0.0, yaw_deg=33.0),),
            _upright_params(mixed_pixel_fraction=0.25),
        ),
        Scenario(
            "mixed_pixels_and_noise",
            "sensor_artifacts",
            (_upright_spec(0.0, 0.0, yaw_deg=33.0),),
            _upright_params(mixed_pixel_fraction=0.35, range_noise_std=0.02, dropout=0.2),
        ),
        # Assembly geometry -- the actual target application. A wall course is a
        # row of blocks in contact, so it hits the same missing-interface problem
        # as `touching_row` but at the scale the robot actually works at. Tracked
        # explicitly because the 3-block touching case understated it: on the
        # config/scenes/wall.yaml scene the detector recovers 1 of 16 blocks.
        Scenario(
            "wall_course_single",
            "assembly",
            tuple(_upright_spec(0.0, 0.6 * i, yaw_deg=0.0) for i in range(5)),
            _upright_params(az_fov_deg=120.0, look_at=(0.0, 1.2, 0.45)),
        ),
        Scenario(
            "wall_two_courses",
            "assembly",
            tuple(
                _upright_spec(0.0, 0.6 * i + (0.3 if course else 0.0), z=0.9 * course + 0.45)
                for course in range(2)
                for i in range(4)
            ),
            _upright_params(az_fov_deg=120.0, look_at=(0.0, 1.0, 0.9)),
        ),
        Scenario(
            "spaced_row_10cm_gaps",
            "assembly",
            tuple(_upright_spec(0.0, 0.7 * i, yaw_deg=0.0) for i in range(4)),
            _upright_params(az_fov_deg=120.0, look_at=(0.0, 1.05, 0.45)),
        ),
        # The measured sensor profile, as opposed to the conservative default.
        Scenario(
            "measured_profile_near",
            "measured_sensor",
            (_upright_spec(0.0, 0.0, yaw_deg=23.0),),
            seyond_measured(dims=BLOCK_DIMS_UPRIGHT),
        ),
        Scenario(
            "measured_profile_far",
            "measured_sensor",
            (_upright_spec(15.0, 0.0, yaw_deg=61.0),),
            seyond_measured(
                dims=BLOCK_DIMS_UPRIGHT, sensor_pos=(-5.0, 0.0, 4.0), look_at=(15.0, 0.0, 0.45)
            ),
        ),
    )


def generate_scenario_matrix(seed: int) -> list[tuple[Scenario, SceneSample]]:
    """Generate every registered scenario with independent deterministic RNGs."""
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(scenario_registry()))
    return [
        (scenario, scenario.generate(int(child.generate_state(1)[0])))
        for scenario, child in zip(scenario_registry(), children, strict=True)
    ]


# ---------------------------------------------------------------------------
# Refine-request families: one requested block per scene, as the world model sends it.
# Site frame: the seyond mount of the 11-09 lab-day snapshot, cast with the recorded Robin W
# pattern gz sim uses, so the sim's sampling of a block is what the bench sees.

ROBIN_W_PATTERN = str(
    Path(__file__).resolve().parents[4]
    / "sensor_simulation/seyond_simulation/scan_mode/robin_w.csv"
)
SENSOR_POS = (-7.219, -0.242, 3.639)
SENSOR_ROTATION = (-0.7582, -0.4099, 0.5070, 0.5297, 0.0662, 0.8456, -0.3802, 0.9097, 0.1669)
VIEWS = {"r6": 6.0, "r7": 7.0, "r8": 8.0}  # row centre this far from the column towards -x (m)
ROW_ANGLES_DEG = (0, 45, 90)  # row axis vs line of sight; 0 = end-on, the wall-row case


@dataclass(frozen=True)
class RequestCase:
    scenario: Scenario
    target: int  # index into the scene's ground truth


def _row(n: int, pitch: float, view: str, angle_deg: float, z: float = 0.3, shift: float = 0.0):
    """n blocks, long axis along the row, row centre VIEWS[view] m from the sensor towards -x."""
    centre = np.array([SENSOR_POS[0] - VIEWS[view], SENSOR_POS[1]])
    u = np.array([np.cos(np.radians(180 + angle_deg)), np.sin(np.radians(180 + angle_deg))])
    return tuple(
        BlockSpec(
            *(centre + ((i - (n - 1) / 2) * pitch + shift) * u), z=z, yaw_deg=180.0 + angle_deg
        )
        for i in range(n)
    )


def _site_params(view: str) -> SynthParams:
    return seyond_measured(
        sensor_pos=SENSOR_POS,
        sensor_rotation=SENSOR_ROTATION,
        ray_pattern=ROBIN_W_PATTERN,
        range_noise_std=0.008,
        range_noise_slope=0.0,  # as gz sim
    )


def _no_clutter(**kwargs) -> ClutterParams:
    return ClutterParams(trees=False, container=False, logs=False, **kwargs)


def request_families() -> dict[str, tuple[RequestCase, ...]]:
    """pickup_touching_row, placed_beside_neighbour, occluded_by_gripper; names are stable."""
    pickup = []
    for n in (3, 4, 5):
        for gap_cm in (0, 2, 5):
            for angle in ROW_ANGLES_DEG:
                for view in VIEWS:
                    specs = _row(n, 0.9 + gap_cm / 100, view, angle)
                    name = f"pickup_n{n}_gap{gap_cm}_a{angle}_{view}"
                    scenario = Scenario(name, "pickup_touching_row", specs, _site_params(view))
                    pickup.append(RequestCase(scenario, n // 2))  # an inner block
    placed = []
    for pitch_mm in (900, 910, 920):
        pitch = pitch_mm / 1000
        for course in (0, 1):
            for angle in ROW_ANGLES_DEG:
                for view in VIEWS:
                    specs = _row(3, pitch, view, angle)
                    if course:  # course 1 rides 0.3 m along, as in example_wall
                        specs += _row(2, pitch, view, angle, z=0.9, shift=0.3 - pitch / 2)
                    name = f"placed_p{pitch_mm}_c{course}_a{angle}_{view}"
                    scenario = Scenario(name, "placed_beside_neighbour", specs, _site_params(view))
                    placed.append(RequestCase(scenario, len(specs) - 1))  # the fresh block
    occluded = []
    # (where, fraction of the line of sight, sideways offset m); centred it hides the block whole
    # hover: gripper bottom 0.4 m over the block top; hover<cm>: higher, as at the pickup hover
    spots = (
        ("los50", 0.5, 0.4),
        ("los50", 0.5, 0.8),
        ("los75", 0.75, 0.3),
        ("los75", 0.75, 0.6),
        ("hover", None, 0.0),
        ("hover150", None, 0.0),
        ("hover290", None, 0.0),
    )
    for n in (1, 3):
        for carrying in (False, True):
            for where, fraction, offset in spots:
                for angle in (0, 90):
                    for view in VIEWS:
                        specs = _row(n, 0.9, view, angle)
                        target = specs[n // 2]
                        sensor = np.array(_site_params(view).sensor_pos)
                        los = np.array([target.x, target.y, 0.3]) - sensor
                        if fraction is None:  # pre-descent, over the block top
                            lift = int(where[5:] or 40) / 100 - 0.4
                            p = np.array([target.x, target.y, lift + (1.3 if carrying else 1.25)])
                        else:
                            side = np.array([-los[1], los[0], 0.0]) / np.linalg.norm(los[:2])
                            p = sensor + fraction * los + offset * side
                        gripper = p + [0, 0, 0.55] if carrying else p
                        boxes = ((tuple(gripper), (1.2, 0.8, 0.5), target.yaw_deg),)  # PZS100
                        if carrying:
                            specs += (BlockSpec(*p, yaw_deg=target.yaw_deg),)
                        name = (
                            f"occluded_n{n}_{'carry' if carrying else 'empty'}_{where}"
                            f"_off{round(offset * 100)}_a{angle}_{view}"
                        )
                        scenario = Scenario(
                            name,
                            "occluded_by_gripper",
                            specs,
                            _site_params(view),
                            _no_clutter(robot_boxes=boxes),
                        )
                        occluded.append(RequestCase(scenario, n // 2))
    return {
        "pickup_touching_row": tuple(pickup),
        "placed_beside_neighbour": tuple(placed),
        "occluded_by_gripper": tuple(occluded),
    }


def world_model_prior(truth, rng: np.random.Generator, stale_share: float = 0.1):
    """Believed pose of `truth`: sigma 1-3 cm in xy, `stale_share` 0.15 m off -> xyz, yaw, stale."""
    stale = bool(rng.random() < stale_share)
    if stale:
        phi = rng.uniform(0, 2 * np.pi)
        offset = np.array([0.15 * np.cos(phi), 0.15 * np.sin(phi), 0.0])
    else:
        offset = np.append(rng.normal(0.0, rng.uniform(0.01, 0.03), 2), rng.normal(0.0, 0.005))
    yaw = np.arctan2(truth.rotation[1, 0], truth.rotation[0, 0]) + rng.normal(0, np.radians(2))
    return truth.position + offset, float(yaw), stale
