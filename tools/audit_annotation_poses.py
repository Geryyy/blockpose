#!/usr/bin/env python3
"""Measure where re-fitting a block from its own points is unreliable.

For each approved pose, refit a canonical block from the points inside it and
compare. The disagreement is a measure of OBSERVABILITY, not of annotation
quality, and it tracks the annotator's own visibility label almost perfectly:

    full             23 blocks   median  1.2 deg disagreement   0 over 30 deg
    partial          20 blocks   median  2.1 deg               3 over 30 deg
    heavy_occlusion   8 blocks   median 87.8 deg               7 over 30 deg

An earlier version of this file claimed the opposite -- that a large
disagreement meant the recorded pose was wrong -- because the flagged blocks
were mostly ones the annotator had placed with the PCA fallback. That was a
confound: the fallback is used precisely BECAUSE a block is too occluded for
the fitter, so `pca_fallback` is a proxy for occlusion, not a cause of error.
The operator checked the flagged snapshots and the ground truth is correct.

What the number is actually good for: it says which blocks carry too little
visible geometry to pin an orientation from the cloud alone. Those are the
blocks a detector will not recover by better fitting, and the ones where
occlusion-aware reasoning or RGB evidence has to supply what the geometry
cannot. Read a high value as "this block is hard", not "this label is wrong".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotation_snap import SnapParams, snap_block_to_cloud  # noqa: E402
from blockpose.annotation_tool import AnnotationDocument  # noqa: E402
from blockpose.evaluate import rotation_error_deg  # noqa: E402
from blockpose.experiment_record import experiment_stamp  # noqa: E402
from blockpose.io import load_snapshot, load_tf_snapshot  # noqa: E402
from blockpose.pose import BLOCK_DIMS, BlockPose  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "config" / "annotated_snapshots_seibersdorf.yaml"
DEFAULT_TRANSFORMS = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
# 30 degrees separates "the geometry pins this" from "the geometry does not":
# well-observed blocks sit near 2 degrees, heavily occluded ones near 90.
UNDERDETERMINED_ROTATION_DEG = 30.0
MIN_POINTS = 50


def audit(manifest: Path, transforms: Path) -> list[dict]:
    spec = yaml.safe_load(manifest.read_text())
    root = REPO_ROOT / spec["dataset"]
    rows: list[dict] = []
    for scene, entry in spec["scenes"].items():
        for snap in entry["annotated"]:
            directory = root / snap["name"]
            tf = load_tf_snapshot(directory / "tf.yaml")
            points, _ = load_snapshot(
                directory, transforms, T_world_cloud=tf.T_world_cloud, require_registrations=False
            )
            for target in AnnotationDocument.load(directory / "annotations.yaml").targets:
                if target.review_status != "approved":
                    continue
                local = (points - target.position) @ target.rotation
                half = np.asarray(target.dimensions) / 2 + 0.05
                inside = points[np.all(np.abs(local) <= half, axis=1)]
                row = {
                    "scene": scene,
                    "snapshot": snap["name"],
                    "block": target.id,
                    "provenance": str(target.provenance),
                    "points": int(len(inside)),
                    "refit": None,
                }
                if len(inside) >= MIN_POINTS:
                    centre = inside.mean(axis=0)
                    seed = inside[int(np.argmin(np.linalg.norm(inside - centre, axis=1)))]
                    result = snap_block_to_cloud(inside, seed, SnapParams(reject_ambiguous=False))
                    if result.success:
                        recorded = BlockPose(
                            target.position, target.rotation, dims=target.dimensions
                        )
                        fitted = BlockPose(
                            np.asarray(result.position),
                            np.asarray(result.rotation),
                            dims=BLOCK_DIMS,
                        )
                        drot = float(rotation_error_deg(fitted, recorded))
                        row["refit"] = {
                            "position_delta_m": float(
                                np.linalg.norm(fitted.position - recorded.position)
                            ),
                            "rotation_delta_deg": drot,
                            "underdetermined": bool(drot > UNDERDETERMINED_ROTATION_DEG),
                        }
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--transforms", type=Path, default=DEFAULT_TRANSFORMS)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows = audit(args.manifest, args.transforms)
    underdetermined = [r for r in rows if r["refit"] and r["refit"]["underdetermined"]]
    unfittable = [r for r in rows if r["refit"] is None]

    print(f"approved blocks audited: {len(rows)}")
    print(
        f"orientation NOT pinned by the visible geometry "
        f"(>{UNDERDETERMINED_ROTATION_DEG:.0f} deg): {len(underdetermined)}"
    )
    print(f"no refit possible (too few points or fit failed): {len(unfittable)}\n")
    for row in sorted(underdetermined, key=lambda r: -r["refit"]["rotation_delta_deg"]):
        print(
            f"  {row['snapshot'][:44]:<46}{row['block']:<10}{row['provenance']:<24}"
            f"drot {row['refit']['rotation_delta_deg']:>5.1f} deg  "
            f"dpos {row['refit']['position_delta_m']:.3f} m"
        )
    print("\nThese are blocks whose visible geometry does not determine an orientation.")
    print("They are NOT mislabelled -- the operator verified them. Treat them as the")
    print("cases a geometry-only detector cannot solve, needing occlusion-aware")
    print("reasoning or RGB evidence rather than a better fitter.")

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {"provenance": experiment_stamp(manifest=args.manifest), "blocks": rows}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
