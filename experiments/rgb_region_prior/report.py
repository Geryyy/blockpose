#!/usr/bin/env python3
"""Held-out region-recall report for the ``dino_proposer`` head.

This is **not** a pixel-IoU segmentation metric. Per the design doc,
``dino_proposer`` predicts a high-recall 2-D region that the detector
projects filtered PCL into and then runs its ordinary local fit -- so what
matters is: for each approved block, was there *some* predicted region loose
enough to plausibly contain it? Concretely, per held-out snapshot: threshold
the predicted probability map, extract connected components, and count a
block as **recovered** if any component's IoU with that block's own
convex-hull silhouette (``export.block_silhouette_mask`` -- the same
construction used to build the training target) exceeds ``--iou-threshold``
(default 0.3, deliberately loose). A predicted component matching no block is
a **false positive**. Reports recall and a rough false-positive rate per
held-out development fold, mirroring
``experiments.rgb_candidate_prior.report``'s development-only-pass scope and
held-out-honesty caveats.

**Region extraction.** Threshold + connected-component extraction
(``config.foreground_threshold``/``config.min_component_area_px``) is owned
by ``blockpose.rgb_region_proposer.extract_region_proposals``, the same
function the deployed ``DinoRegionProposalProvider`` uses, so eval and
deployment never diverge on post-processing.

    python -m experiments.rgb_region_prior.train --held-out-fold 1 \\
        --checkpoint-out out/dino_proposer_fold1_heldout
    python -m experiments.rgb_region_prior.report --held-out-fold 1 \\
        --checkpoint out/dino_proposer_fold1_heldout --export-root out/rgb_region_prior
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ablate_rgb_proposals import load_split_manifest  # noqa: E402
from blockpose.io import load_camera_model, load_tf_snapshot  # noqa: E402

from experiments.rgb_region_prior.dataset import (  # noqa: E402
    RegionDataset,
    RegionSample,
    development_fold_of_snapshot,
    select_development_groups,
)
from experiments.rgb_region_prior.export import (  # noqa: E402
    OCCLUDED_VISIBILITIES,
    approved_targets,
    block_silhouette_mask,
)

DEFAULT_EXPORT_ROOT = REPO_ROOT / "out" / "rgb_region_prior"
DEFAULT_SPLITS = REPO_ROOT / "config" / "rgb_ablation_splits.yaml"
DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"


def region_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("region masks must have the same shape")
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return int(np.logical_and(a, b).sum()) / union


def evaluate_snapshot_recall(
    predicted_regions: list[np.ndarray],
    block_masks: dict[str, np.ndarray],
    *,
    iou_threshold: float = 0.3,
) -> dict:
    """Per-block recovery + false-positive count for one snapshot's predictions.

    A block is "recovered" if any predicted region's IoU with that block's
    own silhouette clears ``iou_threshold`` -- a loose, high-recall region
    check, never a tight-segmentation IoU against the union mask. A predicted
    region matched to no block counts as a false positive; a region may only
    ever be credited toward one block (its single best IoU match) so one
    oversized region cannot "recover" every block in the scene for free.
    """
    recovered_by_block: dict[str, bool] = {}
    matched_region_indices: set[int] = set()
    for block_id, block_mask in block_masks.items():
        best_iou, best_index = 0.0, None
        for index, region in enumerate(predicted_regions):
            iou = region_iou(region, block_mask)
            if iou > best_iou:
                best_iou, best_index = iou, index
        recovered_by_block[block_id] = best_iou >= iou_threshold
        if recovered_by_block[block_id]:
            matched_region_indices.add(best_index)
    blocks_recovered = sum(recovered_by_block.values())
    false_positives = len(predicted_regions) - len(matched_region_indices)
    return {
        "blocks_total": len(block_masks),
        "blocks_recovered": blocks_recovered,
        "recall": blocks_recovered / len(block_masks) if block_masks else None,
        "predicted_regions": len(predicted_regions),
        "matched_regions": len(matched_region_indices),
        "false_positives": max(false_positives, 0),
        "recovered_by_block": recovered_by_block,
    }


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _block_masks_for_sample(
    sample: RegionSample,
    *,
    snapshots_dir: Path,
    camera,
    occluded_visibilities: tuple[str, ...],
) -> tuple[dict, dict]:
    """Recompute each approved block's own silhouette fresh from annotations.yaml.

    The exported record only stores the *union* mask (what the head trains
    on); per-block footprints for this per-block metric are re-derived
    directly from the source snapshot using the exact same construction
    ``export.py`` used to build that union, so train-time and eval-time
    targets can never silently diverge.

    ``occluded_visibilities`` must mirror whatever the checkpoint's export
    used -- this is the eval-side half of that same choice, and its default
    matching ``export.py``'s is what previously made this evaluation silently
    exclude heavy_occlusion/unknown blocks from recall regardless of which
    arm trained the checkpoint.

    Returns ``(masks, visibility_by_block_id)``.
    """
    from blockpose.annotations import load_annotations

    snapshot = snapshots_dir / sample.sample_id
    annotations = load_annotations(snapshot / "annotations.yaml")
    tf = load_tf_snapshot(snapshot / "tf.yaml")
    masks = {}
    visibility = {}
    for target in approved_targets(annotations, occluded_visibilities):
        mask = block_silhouette_mask(target.pose, tf.T_world_camera, camera)
        if mask is not None:
            masks[target.id] = mask
            visibility[target.id] = target.visibility
    return masks, visibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--occluded-visibilities",
        default="none",
        help="Comma-separated visibility labels to EXCLUDE from the recall denominator, or "
        "'none' (the default) to score every approved block. Must match whatever the "
        "checkpoint's `export.py --occluded-visibilities` used -- this script cannot verify "
        f"that from the checkpoint and trusts the caller's assertion, same as --held-out-fold. "
        f"Pass '{','.join(OCCLUDED_VISIBILITIES)}' to reproduce the old (v1) default.",
    )
    parser.add_argument(
        "--held-out-fold",
        type=int,
        default=None,
        help="Assert the checkpoint was trained excluding this development fold (produced by "
        "`train.py --held-out-fold`); restricts evaluation to that fold only, a genuinely "
        "held-out number. Without it, all development data is evaluated and labelled in-sample.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    # Deferred (not module scope) so this script's pure recall/false-positive
    # functions stay importable/testable without torch/transformers installed.
    from blockpose.rgb_region_proposer import extract_region_proposals, load_dino_region_proposer

    args = parse_args()
    occluded_visibilities = (
        ()
        if args.occluded_visibilities.strip().lower() == "none"
        else tuple(v.strip() for v in args.occluded_visibilities.split(",") if v.strip())
    )
    head, config = load_dino_region_proposer(args.checkpoint)
    camera = load_camera_model(args.calibration)

    def extract_regions(probability_map: np.ndarray) -> list[np.ndarray]:
        return [proposal.mask for proposal in extract_region_proposals(probability_map, config)]

    manifest = load_split_manifest(args.splits)
    snapshot_fold = development_fold_of_snapshot(manifest)
    dataset = RegionDataset.load(args.export_root)
    dev_dataset, group_fold = select_development_groups(dataset, snapshot_fold)
    if not dev_dataset.samples:
        raise ValueError(f"no exported sample resolves to a development fold in {args.splits}")

    if args.held_out_fold is not None:
        holdout_groups = {g for g, f in group_fold.items() if f == args.held_out_fold}
        if not holdout_groups:
            raise ValueError(f"fold {args.held_out_fold} has no development groups")
        eval_samples = [s for s in dev_dataset.samples if s.split_group in holdout_groups]
        held_out = True
        provenance_note = (
            f"Evaluated only on development fold {args.held_out_fold}'s snapshots. This is a "
            "genuinely held-out number ONLY IF the checkpoint was trained excluding this fold "
            f"(e.g. via `train.py --held-out-fold {args.held_out_fold}`) -- this script cannot "
            "verify that from the checkpoint file itself and trusts the caller's assertion."
        )
    else:
        eval_samples = list(dev_dataset.samples)
        held_out = False
        provenance_note = (
            "No --held-out-fold given: evaluated on ALL development-fold data. These are "
            "IN-SAMPLE numbers if the checkpoint was trained on any of it (true for the default "
            "`train.py` checkpoint, trained on the union of both development folds)."
        )

    per_snapshot = []
    totals = {
        "blocks_total": 0,
        "blocks_recovered": 0,
        "predicted_regions": 0,
        "false_positives": 0,
    }
    by_visibility: dict[str, dict[str, int]] = {}
    for sample in eval_samples:
        image = _load_image(sample.image_path)
        probability_map = np.asarray(head.predict_proposal_map(image))
        predicted_regions = extract_regions(probability_map)
        block_masks, visibility_by_block = _block_masks_for_sample(
            sample,
            snapshots_dir=args.snapshots_dir,
            camera=camera,
            occluded_visibilities=occluded_visibilities,
        )
        result = evaluate_snapshot_recall(
            predicted_regions, block_masks, iou_threshold=args.iou_threshold
        )
        per_snapshot.append({"sample_id": sample.sample_id, **result})
        totals["blocks_total"] += result["blocks_total"]
        totals["blocks_recovered"] += result["blocks_recovered"]
        totals["predicted_regions"] += result["predicted_regions"]
        totals["false_positives"] += result["false_positives"]
        for block_id, recovered in result["recovered_by_block"].items():
            visibility = visibility_by_block[block_id]
            bucket = by_visibility.setdefault(visibility, {"blocks_total": 0, "blocks_recovered": 0})
            bucket["blocks_total"] += 1
            bucket["blocks_recovered"] += int(recovered)

    groups_evaluated = sorted({sample.split_group for sample in eval_samples})
    warnings = []
    if len(groups_evaluated) < 3:
        warnings.append(
            f"only {len(groups_evaluated)} snapshot groups evaluated: held-out recall will be "
            "high-variance"
        )
    if totals["blocks_total"] == 0:
        warnings.append("no approved blocks in the evaluated snapshots: recall is undefined")

    report = {
        "development_only_pass": True,
        "held_out": held_out,
        "provenance_note": provenance_note,
        "export_root": str(args.export_root),
        "checkpoint": str(args.checkpoint),
        "iou_threshold": args.iou_threshold,
        "occluded_visibilities": list(occluded_visibilities),
        "snapshot_groups_evaluated": groups_evaluated,
        "warnings": warnings,
        "aggregate": {
            **totals,
            "recall": totals["blocks_recovered"] / totals["blocks_total"]
            if totals["blocks_total"]
            else None,
            "false_positives_per_snapshot": totals["false_positives"] / len(eval_samples)
            if eval_samples
            else None,
        },
        "recall_by_visibility": {
            visibility: {
                **counts,
                "recall": counts["blocks_recovered"] / counts["blocks_total"]
                if counts["blocks_total"]
                else None,
            }
            for visibility, counts in sorted(by_visibility.items())
        },
        "snapshots": per_snapshot,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
