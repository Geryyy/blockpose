"""Deterministic synthetic scenario registry for detector diagnostics.

These scenarios broaden measurement without changing the small Phase-A gate.
Each case has a stable name and family so reports can be compared by failure
mode rather than hiding regressions in a single aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass

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
