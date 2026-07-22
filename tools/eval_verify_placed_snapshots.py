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
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from blockpose.annotations import load_annotations  # noqa: E402
from blockpose.detection import SensorContext  # noqa: E402
from blockpose.evaluate import rotation_error_deg  # noqa: E402
from blockpose.io import load_snapshot, load_tf_snapshot  # noqa: E402
from blockpose.snapshot_plan_fixtures import load_snapshot_plan_fixtures  # noqa: E402
from blockpose.verify_placed import verify_placed_blocks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/registration_snapshots")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fixtures = load_snapshot_plan_fixtures()
    grouped = defaultdict(list)
    for target_id, fixture in fixtures.items():
        if fixture.prior is None:
            continue
        snapshot_id, local_id = target_id.split("/", 1)
        assert snapshot_id not in fixture.source_snapshot_ids
        assert fixture.search_window_m == 0.10
        grouped[snapshot_id].append((local_id, fixture))

    rows = []
    for snapshot_id, entries in sorted(grouped.items()):
        directory = args.snapshots / snapshot_id
        tf = load_tf_snapshot(directory / "tf.yaml")
        points, _ = load_snapshot(directory, T_world_cloud=tf.T_world_cloud)
        priors = [entry.prior for _, entry in entries]
        assert all(prior is not None for prior in priors)
        context = SensorContext.from_points(tf.T_world_cloud[:3, 3], points)
        started = time.perf_counter()
        inferred = verify_placed_blocks(points, priors, sensor_context=context)  # type: ignore[arg-type]
        runtime_ms = (time.perf_counter() - started) * 1000.0

        # Deliberately load annotations after inference: GT cannot affect points,
        # priors, assignment, thresholds, or fitted results.
        annotations = load_annotations(directory / "annotations.yaml")
        targets = {target.id: target.pose for target in annotations.targets}
        for (local_id, fixture), result in zip(entries, inferred, strict=True):
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
                    "translation_improvement_m": (
                        prior_translation_error - final_translation_error
                    ),
                    "rotation_improvement_deg": prior_rotation_error - final_rotation_error,
                    "support_points": result.support_points,
                    "fit_diagnostics": result.diagnostics,
                    "runtime_ms_per_snapshot_query": runtime_ms,
                    "prior_source_snapshots": list(fixture.source_snapshot_ids),
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
