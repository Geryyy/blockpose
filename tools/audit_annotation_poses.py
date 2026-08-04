#!/usr/bin/env python3
"""Flag annotated poses whose own points disagree with the recorded orientation.

Twelve of the 62 approved blocks were placed by the annotator's PCA fallback,
which pins a canonical box on a point selection when the native fitter cannot
find a top/side pair. Re-fitting each recorded pose from the points inside it
shows those twelve disagree with their own geometry by a median 88.4 degrees
AFTER symmetry-aware comparison -- i.e. the long axis is wrong, not a face
relabelling. Blocks created by the fitter or by a successful RGB-mask fit agree
to ~2 degrees, so this is specific to the fallback and not a property of the
comparison.

That matters beyond tidiness: strict-scored detector evaluation is measured
against these poses, so a wrong-orientation reference makes a correct detection
look like a failure. Excluding them changes the seeded fitter's strict recovery
on perfect crops from 5/16 to 4/7.

This tool only REPORTS. It never rewrites annotations: a disagreement means the
pose needs a human's eye in the annotator, not an automatic overwrite, and the
refit is exactly the mechanism that was already too weak to trust here.
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
# 30 degrees separates "same block, slightly off" from "wrong axis": agreeing
# poses sit near 2 degrees and the flagged ones near 90.
SUSPECT_ROTATION_DEG = 30.0
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
                            "suspect": bool(drot > SUSPECT_ROTATION_DEG),
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
    suspect = [r for r in rows if r["refit"] and r["refit"]["suspect"]]
    unfittable = [r for r in rows if r["refit"] is None]

    print(f"approved blocks audited: {len(rows)}")
    print(f"orientation disagrees by >{SUSPECT_ROTATION_DEG:.0f} deg: {len(suspect)}")
    print(f"no refit possible (too few points or fit failed): {len(unfittable)}\n")
    for row in sorted(suspect, key=lambda r: -r["refit"]["rotation_delta_deg"]):
        print(
            f"  {row['snapshot'][:44]:<46}{row['block']:<10}{row['provenance']:<24}"
            f"drot {row['refit']['rotation_delta_deg']:>5.1f} deg  "
            f"dpos {row['refit']['position_delta_m']:.3f} m"
        )
    print("\nRe-open these in tools/annotate_snapshot.py and correct the pose by hand.")
    print("Nothing is rewritten here: an automatic refit is the same mechanism that")
    print("produced the suspect poses, so it is not trustworthy as a replacement.")

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {"provenance": experiment_stamp(manifest=args.manifest), "blocks": rows}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
