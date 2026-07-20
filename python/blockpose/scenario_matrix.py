"""Deterministic synthetic scenario registry for detector diagnostics.

These scenarios broaden measurement without changing the small Phase-A gate.
Each case has a stable name and family so reports can be compared by failure
mode rather than hiding regressions in a single aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blockpose.synth import BlockSpec, ClutterParams, SceneSample, SynthParams, generate_scene


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
    """Return the stable diagnostic matrix definition."""
    base = BlockSpec(0.0, 0.0, yaw_deg=23.0)
    return (
        Scenario("isolated_near", "isolated_range", (base,), SynthParams()),
        Scenario(
            "isolated_far",
            "isolated_range",
            (BlockSpec(8.0, 0.0, yaw_deg=61.0),),
            SynthParams(sensor_pos=(-5.0, 0.0, 4.0), look_at=(8.0, 0.0, 0.45)),
        ),
        Scenario(
            "isolated_tilt_7deg",
            "tilt",
            (BlockSpec(0.0, 0.0, yaw_deg=37.0, pitch_deg=7.0, roll_deg=-5.0),),
            SynthParams(),
        ),
        Scenario(
            "isolated_noise_3cm",
            "noise",
            (base,),
            SynthParams(range_noise_std=0.03, range_noise_slope=0.001),
        ),
        Scenario(
            "isolated_dropout_40pct",
            "dropout",
            (base,),
            SynthParams(dropout=0.4, stripe_dropout=0.1),
        ),
        Scenario(
            "three_separated",
            "multi_block",
            tuple(
                BlockSpec(0.0, y, yaw_deg=15.0 + 20.0 * i)
                for i, y in enumerate((-1.3, 0.0, 1.3))
            ),
            SynthParams(az_fov_deg=100.0),
        ),
        Scenario(
            "touching_row",
            "touching",
            tuple(BlockSpec(0.0, 0.6 * i, yaw_deg=0.0) for i in range(3)),
            SynthParams(az_fov_deg=100.0),
        ),
        Scenario(
            "two_stack",
            "stack",
            (BlockSpec(0.0, 0.0, z=0.45), BlockSpec(0.0, 0.0, z=1.35)),
            SynthParams(look_at=(0.0, 0.0, 0.9)),
        ),
        Scenario(
            "near_field_hard_negative",
            "clutter",
            (base,),
            SynthParams(az_fov_deg=100.0),
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
            SynthParams(az_fov_deg=100.0, look_at=(0.0, 0.0, 0.45)),
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
            SynthParams(az_fov_deg=100.0, look_at=(0.0, 0.0, 0.45)),
            ClutterParams(
                trees=False,
                container=False,
                logs=False,
                near_field_boxes=1,
                near_field_offset=0.0,
                near_field_dims=(0.6, 1.8, 0.9),
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
