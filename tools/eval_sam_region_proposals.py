#!/usr/bin/env python3
"""Measure how much SAM2 region proposals recover of the PCL detector's misses.

Context: on `config/annotated_snapshots_seibersdorf.yaml` (25 snapshots, 62
approved GT blocks) the whole-cloud PCL source reaches 39/62 = 63% recall. An
oracle test (crop to a missed block's own GT box, rerun the *existing*
fitter) showed 17 of the 23 misses have ample points and are recoverable at
2.6 cm median error -- i.e. this is a segmentation/region problem, not a
sensing problem. This script asks whether SAM2, run zero-shot with a grid of
point prompts and NO training, supplies region proposals that close any of
that gap in practice.

Pipeline per snapshot:
  1. Prompt SAM2 with a `--grid` x `--grid` point grid over the RGB image, one
     independent single-point object per grid cell, and keep each object's
     highest-IoU mask (of the 3 returned per prompt).
  2. Drop obviously useless masks by pixel area (`--mask-min-area-px`,
     `--mask-max-area-frac`) -- these are typically background reflections/
     ground rather than a candidate object.
  3. Project each surviving mask to 3-D with the SAME `rgb_annotation` code
     path the interactive annotator uses (`build_projection_cache`,
     `select_points_from_mask`), seeded at the mask centroid. This also
     enforces a minimum projected-point count via `MaskSelectionParams`
     (`select_points_from_mask` degrades to `success=False` when too few
     points survive its depth/connectivity filtering) -- reused rather than
     reimplemented, per the repo's single-source-of-truth convention.
  4. Run the EXISTING geometric fitter (`annotation_candidates._run_detection`,
     the same internal entry point used per DINO proposal region) on just
     that mask's points. No new detector, no training.
  5. Pool every SAM-sourced accepted hypothesis with the whole-cloud PCL
     accepted candidates, and de-duplicate by centre distance
     (`--dedup-radius`), keeping the higher-scoring candidate of any pair
     closer than that radius.
  6. Score recall/precision against the 62 GT blocks with the same 0.30 m
     centre-distance rule as the established baseline (`--match-dist`).

Usage:
    uv run python tools/eval_sam_region_proposals.py --grid 8 \
        --report-out experiments/sam_region_proposals.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.annotation_candidates import _run_detection  # noqa: E402
from blockpose.annotation_tool import AnnotationDocument, EditableTarget  # noqa: E402
from blockpose.evaluate import match_detections  # noqa: E402
from blockpose.io import (  # noqa: E402
    CameraModel,
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_tf_snapshot,
)
from blockpose.pose import BlockPose  # noqa: E402
from blockpose.rgb_annotation import (  # noqa: E402
    MaskSelectionParams,
    ProjectionCache,
    build_projection_cache,
    select_points_from_mask,
)

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "seibersdorf_dataset"
DEFAULT_MANIFEST = REPO_ROOT / "config" / "annotated_snapshots_seibersdorf.yaml"
DEFAULT_TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
_ORACLE_RECALL = "90%"  # oracle ceiling from the crop-to-GT-box experiment; see module docstring


# --------------------------------------------------------------------------
# Pure, GPU-free helpers (covered directly by tests/python/test_sam_region_proposals.py)
# --------------------------------------------------------------------------


def grid_points(width: int, height: int, grid: int, margin_frac: float = 0.02) -> list[list[float]]:
    """A `grid` x `grid` lattice of pixel coordinates, inset from the border.

    Prompts sitting exactly on row/column 0 or width-1/height-1 tend to land on
    frame edges/vignetting rather than scene content, so the lattice is inset
    by `margin_frac` of each dimension rather than spanning the full image.
    """
    if width <= 0 or height <= 0 or grid <= 0:
        raise ValueError("width, height, and grid must be positive")
    xs = np.linspace(margin_frac * width, (1.0 - margin_frac) * width, grid)
    ys = np.linspace(margin_frac * height, (1.0 - margin_frac) * height, grid)
    return [[float(x), float(y)] for y in ys for x in xs]


def mask_passes_area_filter(
    area_px: int, image_pixels: int, min_area_px: int, max_area_frac: float
) -> bool:
    """Cheap pixel-area screen for "obviously useless" SAM masks.

    A single grid prompt can return a whole-image or near-empty mask (sky,
    ground plane, a reflection); both are worth skipping before paying for
    projection + detection. `max_area_frac` is relative to image size because
    "too large" depends on resolution, `min_area_px` is absolute because a
    fixed floor is what actually rules out degenerate slivers.
    """
    if image_pixels <= 0:
        return False
    return min_area_px <= area_px <= max_area_frac * image_pixels


@dataclass(frozen=True)
class PooledCandidate:
    """One accepted detection hypothesis pooled from either source, pre-dedup."""

    id: str
    position: np.ndarray
    rotation: np.ndarray
    dims: tuple[float, float, float]
    score: float
    source: str  # "pcl" or "sam"

    def to_pose(self) -> BlockPose:
        return BlockPose(self.position, self.rotation, dims=self.dims)


def dedupe_candidates(candidates: list[PooledCandidate], radius_m: float) -> list[PooledCandidate]:
    """Greedy score-sorted centre-distance de-duplication.

    Highest geometry score first, so that when a PCL candidate and a SAM
    candidate agree on the same physical block within `radius_m`, the pooled
    result keeps exactly one -- the better-scored of the two -- rather than
    double-counting it as two detections (which would silently deflate
    precision without adding real recall). Order-independent (unlike a plain
    first-wins scan), because the whole point is comparing two sources that
    are appended in an arbitrary order.
    """
    ordered = sorted(candidates, key=lambda c: (-c.score, c.id))
    kept: list[PooledCandidate] = []
    for candidate in ordered:
        if all(np.linalg.norm(candidate.position - other.position) > radius_m for other in kept):
            kept.append(candidate)
    return kept


# --------------------------------------------------------------------------
# SAM2 grid proposal generation (GPU/model boundary; not exercised by tests)
# --------------------------------------------------------------------------


class Sam2GridProposer:
    """Lazy SAM2 backend that scores a whole point grid in one forward pass.

    Unlike `blockpose.rgb_annotation.TransformersSam2Backend` (one point per
    call, built for interactive single clicks), this batches every grid point
    as an independent single-point object in one `Sam2Model` call, which is
    what makes a `--grid 8` (64 prompts) pass take a fraction of a second
    instead of 64 sequential forward passes.
    """

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._torch = None

    def _load(self) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Sam2Processor.from_pretrained(self.model_id)
        self._model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()
        self._torch = torch

    def propose(self, rgb: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
        """Return `(masks, areas)`: masks is `(grid*grid, H, W)` bool, best-of-3 each."""
        if self._model is None:
            self._load()
        assert self._processor is not None and self._model is not None and self._torch is not None
        height, width = rgb.shape[:2]
        points = grid_points(width, height, grid)
        input_points = [[[point] for point in points]]  # (batch=1, objects=n, points=1, 2)
        input_labels = [[[1] for _ in points]]
        inputs = self._processor(
            images=rgb,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]  # (n, 3, H, W)
        scores = outputs.iou_scores.detach().cpu().numpy()[0]  # (n, 3)
        best = scores.argmax(axis=1)
        best_masks = masks.numpy()[np.arange(len(points)), best]
        areas = best_masks.reshape(len(points), -1).sum(axis=1)
        return best_masks, areas


def sam_candidates_for_snapshot(
    proposer: Sam2GridProposer,
    points: np.ndarray,
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    sensor_origin: np.ndarray,
    *,
    grid: int,
    mask_min_area_px: int,
    mask_max_area_frac: float,
    mask_params: MaskSelectionParams,
) -> tuple[list[PooledCandidate], dict]:
    """Run the SAM2-grid-proposal -> mask-select -> existing-fitter pipeline.

    Returns (accepted candidates, diagnostics). Diagnostics record how many
    masks were generated/survived each filter stage, for the wall-clock and
    grid-density reporting in `experiments/sam_region_proposals.md`.
    """
    masks, areas = proposer.propose(rgb, grid)
    image_pixels = rgb.shape[0] * rgb.shape[1]
    cache: ProjectionCache = build_projection_cache(points, T_world_camera, camera)

    survived_area = 0
    survived_projection = 0
    candidates: list[PooledCandidate] = []
    for i in range(masks.shape[0]):
        mask = masks[i]
        area = int(areas[i])
        if not mask_passes_area_filter(area, image_pixels, mask_min_area_px, mask_max_area_frac):
            continue
        survived_area += 1
        ys, xs = np.nonzero(mask)
        prompt_xy = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
        selection = select_points_from_mask(points, cache, mask, prompt_xy, mask_params)
        if not selection.success:
            continue
        survived_projection += 1
        hypotheses = _run_detection(points[selection.indices], sensor_origin)
        for hypothesis in hypotheses:
            if hypothesis.fate != "final":
                continue
            candidates.append(
                PooledCandidate(
                    f"sam:{i}:{hypothesis.id}",
                    hypothesis.pose.position,
                    hypothesis.pose.rotation,
                    hypothesis.pose.dims,
                    float(hypothesis.score),
                    "sam",
                )
            )
    diagnostics = {
        "masks_generated": int(masks.shape[0]),
        "masks_survived_area_filter": survived_area,
        "masks_survived_projection_filter": survived_projection,
        "sam_accepted_hypotheses": len(candidates),
    }
    return candidates, diagnostics


# --------------------------------------------------------------------------
# Snapshot manifest + GT loading
# --------------------------------------------------------------------------


def load_manifest_snapshots(manifest_path: Path) -> list[tuple[str, str]]:
    """Return `[(scene, snapshot_name), ...]` in manifest order."""
    data = yaml.safe_load(manifest_path.read_text()) or {}
    result = []
    for scene, info in (data.get("scenes") or {}).items():
        for entry in info.get("annotated") or []:
            result.append((scene, str(entry["name"])))
    return result


def approved_targets(doc: AnnotationDocument) -> list[EditableTarget]:
    return [t for t in doc.targets if t.review_status == "approved"]


def provenance_bucket(provenance: str) -> str:
    """Collapse a raw `target.provenance` string to its reporting category.

    `auto_fit` provenance strings carry a unique per-block lineage suffix
    (e.g. `auto_fit:component-5/region-0/dims-0/top-0/side-3/refined`) --
    every approved auto_fit block has a DIFFERENT exact string, so grouping
    by the raw value would put each one in its own singleton row instead of
    the three categories the report asks for (`auto_fit` vs `rgb_mask` vs
    `rgb_mask_pca_fallback`).
    """
    return "auto_fit" if provenance.startswith("auto_fit:") else provenance


# --------------------------------------------------------------------------
# Per-snapshot evaluation
# --------------------------------------------------------------------------


@dataclass
class SnapshotResult:
    scene: str
    name: str
    gt: list[EditableTarget]
    baseline_candidates: list[PooledCandidate]
    sam_candidates: list[PooledCandidate]
    pooled_candidates: list[PooledCandidate]
    baseline_matched_gt: set[int]
    pooled_matched_gt: set[int]
    baseline_matched_det: set[int]
    pooled_matched_det: set[int]
    sam_wall_time_s: float
    sam_diagnostics: dict


def evaluate_snapshot(
    scene: str,
    name: str,
    dataset_root: Path,
    proposer: Sam2GridProposer,
    camera: CameraModel,
    *,
    grid: int,
    mask_min_area_px: int,
    mask_max_area_frac: float,
    dedup_radius: float,
    match_dist: float,
    mask_params: MaskSelectionParams,
) -> SnapshotResult:
    d = dataset_root / name
    tf = load_tf_snapshot(d / "tf.yaml")
    points, _ = load_snapshot(
        d, DEFAULT_TRANSFORM_CONFIG, T_world_cloud=tf.T_world_cloud, require_registrations=False
    )
    doc = AnnotationDocument.load(d / "annotations.yaml")
    gt = approved_targets(doc)
    rgb = load_rgb(d / "rgb.png")
    sensor_origin = tf.T_world_cloud[:3, 3]

    # Equivalent to `compute_candidate_sources(...).pcl` with no gripper-exclusion
    # geometry: that helper's `pcl` output is exactly `_run_detection` filtered to
    # `fate == "final"`, gripper-rejection is a no-op when no boxes are supplied.
    raw_hypotheses = _run_detection(points, sensor_origin)
    baseline_candidates = [
        PooledCandidate(h.id, h.pose.position, h.pose.rotation, h.pose.dims, float(h.score), "pcl")
        for h in raw_hypotheses
        if h.fate == "final"
    ]

    started = time.perf_counter()
    sam_candidates, sam_diag = sam_candidates_for_snapshot(
        proposer,
        points,
        rgb,
        tf.T_world_camera,
        camera,
        sensor_origin,
        grid=grid,
        mask_min_area_px=mask_min_area_px,
        mask_max_area_frac=mask_max_area_frac,
        mask_params=mask_params,
    )
    sam_wall_time_s = time.perf_counter() - started

    pooled_candidates = dedupe_candidates(baseline_candidates + sam_candidates, dedup_radius)

    gt_poses = [BlockPose(t.position, t.rotation, dims=t.dimensions) for t in gt]
    baseline_pairs = match_detections(
        [c.to_pose() for c in baseline_candidates], gt_poses, max_dist=match_dist
    )
    pooled_pairs = match_detections(
        [c.to_pose() for c in pooled_candidates], gt_poses, max_dist=match_dist
    )

    return SnapshotResult(
        scene=scene,
        name=name,
        gt=gt,
        baseline_candidates=baseline_candidates,
        sam_candidates=sam_candidates,
        pooled_candidates=pooled_candidates,
        baseline_matched_gt={gi for _, gi in baseline_pairs},
        pooled_matched_gt={gi for _, gi in pooled_pairs},
        baseline_matched_det={di for di, _ in baseline_pairs},
        pooled_matched_det={di for di, _ in pooled_pairs},
        sam_wall_time_s=sam_wall_time_s,
        sam_diagnostics=sam_diag,
    )


# --------------------------------------------------------------------------
# Aggregation + reporting
# --------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.0f}%)"


def _aggregate(results: list[SnapshotResult]) -> dict:
    gt_total = sum(len(r.gt) for r in results)
    baseline_det = sum(len(r.baseline_candidates) for r in results)
    baseline_matched = sum(len(r.baseline_matched_gt) for r in results)
    pooled_det = sum(len(r.pooled_candidates) for r in results)
    pooled_matched = sum(len(r.pooled_matched_gt) for r in results)
    return {
        "gt_total": gt_total,
        "baseline_det": baseline_det,
        "baseline_matched": baseline_matched,
        "pooled_det": pooled_det,
        "pooled_matched": pooled_matched,
    }


def build_report(results: list[SnapshotResult], args: argparse.Namespace) -> str:
    overall = _aggregate(results)

    lines: list[str] = []
    lines.append("# SAM2 region proposals: does zero-shot segmentation close the recall gap?")
    lines.append("")
    lines.append(
        "No training, no fitter changes. SAM2 (`facebook/sam2.1-hiera-small`) is prompted with a "
        f"{args.grid}x{args.grid} point grid per image; surviving masks are projected to 3-D with "
        "`blockpose.rgb_annotation` (the same code the interactive annotator uses) and run through "
        "the existing geometric fitter's internal entry point "
        "(`annotation_candidates._run_detection`) exactly like a DINO proposal region. SAM-sourced "
        "accepted hypotheses are pooled with the whole-cloud PCL candidates and de-duplicated by "
        f"centre distance ({args.dedup_radius:.2f} m). A block counts as detected when an accepted "
        f"candidate's centre is within {args.match_dist:.2f} m of the GT centre "
        f"(Hungarian matching, "
        "same rule as the established baseline)."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| | Recall | Precision |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Baseline (whole-cloud PCL) | {_rate(overall['baseline_matched'], overall['gt_total'])} "
        f"| {_rate(overall['baseline_matched'], overall['baseline_det'])} |"
    )
    lines.append(
        f"| SAM2 region proposals + PCL, pooled | "
        f"{_rate(overall['pooled_matched'], overall['gt_total'])} "
        f"| {_rate(overall['pooled_matched'], overall['pooled_det'])} |"
    )
    lines.append(f"| Oracle ceiling (crop to GT box, same fitter) | {_ORACLE_RECALL} recall | -- |")
    lines.append("")

    # Miss recovery / new-FP accounting.
    baseline_misses: list[tuple[SnapshotResult, int]] = []
    recovered: list[tuple[SnapshotResult, int]] = []
    for r in results:
        for gi in range(len(r.gt)):
            if gi not in r.baseline_matched_gt:
                baseline_misses.append((r, gi))
                if gi in r.pooled_matched_gt:
                    recovered.append((r, gi))
    new_fp_count = 0
    for r in results:
        matched_ids = {r.pooled_candidates[di].id for di in r.pooled_matched_det}
        for c in r.pooled_candidates:
            if c.source == "sam" and c.id not in matched_ids:
                new_fp_count += 1
    lines.append("## Miss recovery")
    lines.append("")
    lines.append(
        f"Of {len(baseline_misses)} GT blocks missed by the whole-cloud PCL baseline, SAM2 region "
        f"proposals recovered {len(recovered)} once pooled and de-duplicated. "
        f"SAM-sourced candidates "
        f"that ended up unmatched to any GT block (new false positives beyond the baseline's own): "
        f"{new_fp_count}."
    )
    lines.append("")
    if recovered:
        lines.append("Recovered blocks:")
        for r, gi in recovered:
            bucket = provenance_bucket(r.gt[gi].provenance)
            lines.append(f"- {r.name} / {r.gt[gi].id} (provenance: {bucket})")
        lines.append("")

    # Per-scene breakdown.
    lines.append("## Per-scene breakdown")
    lines.append("")
    lines.append(
        "| Scene | GT | Baseline recall | Baseline precision | Pooled recall | Pooled precision |"
    )
    lines.append("|---|---|---|---|---|---|")
    scenes = sorted({r.scene for r in results})
    for scene in scenes:
        scene_results = [r for r in results if r.scene == scene]
        agg = _aggregate(scene_results)
        lines.append(
            f"| {scene} | {agg['gt_total']} | {_rate(agg['baseline_matched'], agg['gt_total'])} "
            f"| {_rate(agg['baseline_matched'], agg['baseline_det'])} "
            f"| {_rate(agg['pooled_matched'], agg['gt_total'])} "
            f"| {_rate(agg['pooled_matched'], agg['pooled_det'])} |"
        )
    lines.append("")

    # Per-provenance breakdown.
    lines.append("## Per-GT-provenance breakdown")
    lines.append("")
    lines.append(
        "`rgb_mask`/`rgb_mask_pca_fallback` are the human-annotated provenances for blocks the "
        "detector historically misses (see `config/annotated_snapshots_seibersdorf.yaml`); "
        "`auto_fit` is a block the detector's own geometric fit already found."
    )
    lines.append("")
    lines.append("| Provenance | GT count | Baseline recall | Pooled recall |")
    lines.append("|---|---|---|---|")
    provenances = sorted({provenance_bucket(t.provenance) for r in results for t in r.gt})
    for prov in provenances:
        gt_count = 0
        baseline_hit = 0
        pooled_hit = 0
        for r in results:
            for gi, t in enumerate(r.gt):
                if provenance_bucket(t.provenance) != prov:
                    continue
                gt_count += 1
                if gi in r.baseline_matched_gt:
                    baseline_hit += 1
                if gi in r.pooled_matched_gt:
                    pooled_hit += 1
        lines.append(
            f"| {prov} | {gt_count} | {_rate(baseline_hit, gt_count)} | "
            f"{_rate(pooled_hit, gt_count)} |"
        )
    lines.append("")

    # Wall clock.
    times = [r.sam_wall_time_s for r in results]
    lines.append("## Wall clock")
    lines.append("")
    lines.append(
        f"Grid density: {args.grid}x{args.grid} ({args.grid * args.grid} prompts/snapshot). "
        f"SAM2-region pipeline (grid inference + mask filtering + per-mask projection + fitter) "
        f"per snapshot: mean {np.mean(times):.2f} s, median {np.median(times):.2f} s, "
        f"min {np.min(times):.2f} s, max {np.max(times):.2f} s, over {len(times)} snapshots on "
        f"{args.device if args.device != 'auto' else 'cuda (auto-selected)'}."
    )
    lines.append("")
    total_masks = sum(r.sam_diagnostics["masks_generated"] for r in results)
    survived_area = sum(r.sam_diagnostics["masks_survived_area_filter"] for r in results)
    survived_proj = sum(r.sam_diagnostics["masks_survived_projection_filter"] for r in results)
    total_accepted = sum(r.sam_diagnostics["sam_accepted_hypotheses"] for r in results)
    lines.append(
        f"Mask funnel across all snapshots: {total_masks} masks generated -> "
        f"{survived_area} survived "
        f"the pixel-area filter (`--mask-min-area-px {args.mask_min_area_px}`, "
        f"`--mask-max-area-frac {args.mask_max_area_frac}`) -> {survived_proj} survived "
        f"`select_points_from_mask` projection -> {total_accepted} produced an accepted "
        '(`fate == "final"`) fitter hypothesis.'
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration", type=Path, default=None, help="camera calib YAML override")
    parser.add_argument(
        "--grid", type=int, default=8, help="grid density (grid x grid point prompts)"
    )
    parser.add_argument("--mask-min-area-px", type=int, default=400)
    parser.add_argument("--mask-max-area-frac", type=float, default=0.35)
    parser.add_argument(
        "--dedup-radius", type=float, default=0.25, help="pooling de-dup radius, meters"
    )
    parser.add_argument("--match-dist", type=float, default=0.30, help="GT match radius, meters")
    parser.add_argument("--model-id", type=str, default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--limit", type=int, default=None, help="only run the first N snapshots (debug)"
    )
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None, help="raw per-snapshot metrics")
    args = parser.parse_args()

    camera = load_camera_model(args.calibration)
    manifest = load_manifest_snapshots(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    proposer = Sam2GridProposer(model_id=args.model_id, device=args.device)
    mask_params = MaskSelectionParams()

    results: list[SnapshotResult] = []
    for scene, name in manifest:
        print(f"[{len(results) + 1}/{len(manifest)}] {name}", file=sys.stderr)
        results.append(
            evaluate_snapshot(
                scene,
                name,
                args.dataset_root,
                proposer,
                camera,
                grid=args.grid,
                mask_min_area_px=args.mask_min_area_px,
                mask_max_area_frac=args.mask_max_area_frac,
                dedup_radius=args.dedup_radius,
                match_dist=args.match_dist,
                mask_params=mask_params,
            )
        )

    report = build_report(results, args)
    print(report)
    if args.report_out is not None:
        args.report_out.write_text(report)
    if args.json_out is not None:
        overall = _aggregate(results)
        args.json_out.write_text(
            json.dumps(
                {
                    "overall": overall,
                    "snapshots": [
                        {
                            "scene": r.scene,
                            "name": r.name,
                            "gt": len(r.gt),
                            "baseline_matched": len(r.baseline_matched_gt),
                            "baseline_det": len(r.baseline_candidates),
                            "pooled_matched": len(r.pooled_matched_gt),
                            "pooled_det": len(r.pooled_candidates),
                            "sam_wall_time_s": r.sam_wall_time_s,
                            **r.sam_diagnostics,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
