#!/usr/bin/env python3
"""Diagnostic-only VerifyPlaced run on leakage-safe snapshot plan priors.

Ground truth is loaded only after inference and is used solely to report error.
These 15 correlated development priors are not an acceptance or tuning set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from blockpose.annotations import load_annotations  # noqa: E402
from blockpose.detection import SensorContext  # noqa: E402
from blockpose.evaluate import rotation_error_deg  # noqa: E402
from blockpose.io import load_snapshot, load_tf_snapshot  # noqa: E402
from blockpose.snapshot_plan_fixtures import load_snapshot_plan_fixtures  # noqa: E402
from blockpose.snapshot_plan_topology import load_snapshot_plan_topology  # noqa: E402
from blockpose.verify_placed import verify_placed_blocks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/registration_snapshots")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Topology and priors are loaded before annotations by construction.
    topology = load_snapshot_plan_topology()
    fixtures = load_snapshot_plan_fixtures()

    rows = []
    for snapshot_id, priors in sorted(topology.priors_by_snapshot.items()):
        entries = [
            (prior.block_id, fixtures[f"{snapshot_id}/{prior.block_id}"]) for prior in priors
        ]
        directory = args.snapshots / snapshot_id
        tf = load_tf_snapshot(directory / "tf.yaml")
        points, _ = load_snapshot(directory, T_world_cloud=tf.T_world_cloud)
        context = SensorContext.from_points(tf.T_world_cloud[:3, 3], points)
        started = time.perf_counter()
        independent = verify_placed_blocks(
            points, [prior.pose for prior in priors], sensor_context=context
        )
        independent_runtime_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        inferred = verify_placed_blocks(
            points,
            list(priors),
            sensor_context=context,
            course_constraints=list(topology.constraints_by_snapshot.get(snapshot_id, ())),
        )
        course_runtime_ms = (time.perf_counter() - started) * 1000.0

        # Deliberately load annotations after inference: GT cannot affect points,
        # priors, assignment, thresholds, or fitted results.
        annotations = load_annotations(directory / "annotations.yaml")
        targets = {target.id: target.pose for target in annotations.targets}
        for (local_id, fixture), independent_result, result in zip(
            entries, independent, inferred, strict=True
        ):
            target = targets[local_id]
            prior_translation_error = float(
                np.linalg.norm(fixture.prior.position - target.position)  # type: ignore[union-attr]
            )
            prior_rotation_error = rotation_error_deg(fixture.prior, target)  # type: ignore[arg-type]
            final_translation_error = float(np.linalg.norm(result.pose.position - target.position))
            final_rotation_error = rotation_error_deg(result.pose, target)
            rows.append(
                {
                    "target_id": f"{snapshot_id}/{local_id}",
                    "stable_slot_id": fixture.stable_slot_id,
                    "status": result.status.value,
                    "observability": result.observability.value,
                    "prior_translation_error_m": prior_translation_error,
                    "prior_rotation_error_deg": prior_rotation_error,
                    "translation_error_m": final_translation_error,
                    "rotation_error_deg": final_rotation_error,
                    "independent_translation_error_m": float(
                        np.linalg.norm(independent_result.pose.position - target.position)
                    ),
                    "independent_rotation_error_deg": rotation_error_deg(
                        independent_result.pose, target
                    ),
                    "translation_improvement_m": (
                        prior_translation_error - final_translation_error
                    ),
                    "rotation_improvement_deg": prior_rotation_error - final_rotation_error,
                    "support_points": result.support_points,
                    "fit_diagnostics": result.diagnostics,
                    "independent_runtime_ms_per_snapshot_query": independent_runtime_ms,
                    "runtime_ms_per_snapshot_query": course_runtime_ms,
                    "prior_source_snapshots": list(fixture.source_snapshot_ids),
                    "topology_provenance": topology.provenance,
                    "acceptance_eligible": False,
                }
            )
    report = {
        "diagnostic_only": True,
        "no_real_threshold_tuning": True,
        "targets": rows,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
