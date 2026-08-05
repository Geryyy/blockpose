#!/usr/bin/env python3
"""Measure the actual composed pipeline: geometry -> +dino_proposer -> +dino_verifier.

Context: `experiments/pose_quality_and_partial_observation.md` and
`notebooks/blockpose_detector_showcase.ipynb` measure `dino_proposer` and
`dino_verifier` each in isolation (region-recall IoU metric; candidate
re-ranking accuracy) -- neither number describes what happens if you actually
run them together with the PCL detector and see how many *more real blocks*
get accepted. This script closes that gap using the exact same pooling
methodology as `tools/eval_sam_region_proposals.py` (SAM2 grid regions),
substituting the trained `dino_proposer` checkpoint for the region source,
and adding a `dino_verifier`-based rerank stage on top of the pooled set.

Pipeline per snapshot:
  1. `dino_proposer` predicts a dense foreground probability map for the RGB
     image (`DinoRegionProposerHead.predict_proposal_map`), thresholded into
     connected-component regions (`extract_region_proposals`) -- the same
     post-processing the deployed `DinoRegionProposalProvider` would use.
  2. Each region's mask is projected to 3-D with the SAME
     `blockpose.rgb_annotation` code path SAM2 proposals use
     (`build_projection_cache`, `select_points_from_mask`), and the retained
     points are run through the EXISTING geometric fitter's internal entry
     point (`annotation_candidates._run_detection`) -- no new detector.
  3. `dino_proposer`-sourced accepted hypotheses are pooled with the
     whole-cloud PCL accepted candidates and de-duplicated by centre distance
     (`--dedup-radius`), exactly as the SAM2 tool does. This produces the
     "geometry + dino_proposer" row.
  4. For the "+ dino_verifier" row, every pooled (post-dedup) candidate is
     scored with `score_candidate_with_dino` (its own per-candidate RGB
     crop + rendered conditioning image, `--crop-margin-frac`/`--out-size`
     matching training), and the pool is re-deduplicated using
     `geometry_score + verifier_contribution` as the tie-break instead of
     the raw geometry score alone -- i.e. the verifier's only effect is on
     WHICH of two competing candidates for the same physical block wins,
     never on generating or discarding candidates outright, matching
     `docs/dino-rgb-prior-ros-integration.md`'s "bounded reranking
     contribution ... cannot rescue a candidate that fails geometry checks."
  5. Score recall/precision against the 62 GT blocks with the same 0.30 m
     centre-distance rule as every other report in this repo.

**Honesty caveat, read before quoting a number from this script:** by
default this loads the checkpoints in `models/production/` (`dino_proposer`,
`dino_verifier`), which are trained on the union of all 25 annotated
snapshots -- these are **in-sample** numbers on this dataset, not a
held-out estimate. Pass `--proposer-checkpoint`/`--verifier-checkpoint`
pointing at a fold-held-out checkpoint pair (see
`experiments/rgb_region_prior/train.py --held-out-fold N` and
`experiments/rgb_candidate_prior/train.py --held-out-fold N`) for a genuinely
held-out number; this script does not do that sweep itself.

Usage:
    uv run python tools/eval_dino_proposer_pooled.py \
        --report-out experiments/dino_proposer_pooled_ablation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.annotation_candidates import _MIN_POINTS_FOR_DETECTION, _run_detection  # noqa: E402
from blockpose.annotation_tool import AnnotationDocument, EditableTarget  # noqa: E402
from blockpose.core_bridge import detect_blocks_detailed_cpp  # noqa: E402
from blockpose.core_types import DetectionParameters, SensorContext  # noqa: E402
from blockpose.evaluate import match_detections  # noqa: E402
from blockpose.experiment_record import (  # noqa: E402
    DEFAULT_MAX_ROTATION_DEG,
    DEFAULT_MIN_IOU,
    experiment_stamp,
    strict_match,
)
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
from blockpose.rgb_region_proposer import (  # noqa: E402
    extract_region_proposals,
    load_dino_region_proposer,
)
from blockpose.rgb_verifier import load_dino_verifier, score_candidate_with_dino  # noqa: E402

from experiments.rgb_candidate_prior import render  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "seibersdorf_dataset"
DEFAULT_MANIFEST = REPO_ROOT / "config" / "annotated_snapshots_seibersdorf.yaml"
DEFAULT_TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
DEFAULT_PROPOSER_CHECKPOINT = REPO_ROOT / "models" / "production" / "dino_proposer"
DEFAULT_VERIFIER_CHECKPOINT = REPO_ROOT / "models" / "production" / "dino_verifier"


# --------------------------------------------------------------------------
# Pooling primitives (shared shape with tools/eval_sam_region_proposals.py)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PooledCandidate:
    """One accepted detection hypothesis pooled from either source, pre-dedup."""

    id: str
    position: np.ndarray
    rotation: np.ndarray
    dims: tuple[float, float, float]
    score: float
    source: str  # "pcl" or "dino_proposer"

    def to_pose(self) -> BlockPose:
        return BlockPose(self.position, self.rotation, dims=self.dims)


def dedupe_candidates(
    candidates: list[PooledCandidate],
    radius_m: float,
    scores: dict[str, float] | None = None,
) -> list[PooledCandidate]:
    """Greedy score-sorted centre-distance de-duplication.

    `scores`, if given, overrides each candidate's tie-break score by id
    (used for the verifier-adjusted rerank pass); falls back to
    `candidate.score` (raw geometry score) otherwise. Order-independent,
    matching `eval_sam_region_proposals.py`'s rationale: whichever source's
    candidate is kept for one physical block should be the better-scored one,
    not whichever happened to be appended first.
    """
    key = (lambda c: scores.get(c.id, c.score)) if scores else (lambda c: c.score)
    ordered = sorted(candidates, key=lambda c: (-key(c), c.id))
    kept: list[PooledCandidate] = []
    for candidate in ordered:
        if all(np.linalg.norm(candidate.position - other.position) > radius_m for other in kept):
            kept.append(candidate)
    return kept


# --------------------------------------------------------------------------
# dino_proposer region generation (mirrors Sam2GridProposer's role)
# --------------------------------------------------------------------------


def fit_region_scored_against(
    region_points: np.ndarray,
    sensor_context: SensorContext | None,
    params: DetectionParameters | None,
):
    """Fit on `region_points`, but score the result against `sensor_context`.

    Mirrors `annotation_candidates._run_detection` (same min-point guard, same
    exception handling) except that the SensorContext is supplied by the
    caller instead of being rebuilt from the region's own points.

    **This distinction is the whole point of `--score-context`.** The
    detector's score is 60% ray-based evidence
    (`evidence.hpp::normalized_evidence_features` features 2/3/4: covered vs
    expected faces, supported rays vs free-space violations, supported vs
    incident rays). Every one of those terms is computed against whatever
    SensorContext it is handed. Build that context from a few-hundred-point
    region crop instead of the ~114k-point whole cloud and the hypothesis
    simply cannot accrue the free-space violations the whole cloud would have
    charged it -- so its score comes out systematically HIGHER than a
    whole-cloud candidate's, for reasons that have nothing to do with which
    is the better fit. Measured on one snapshot: identical crop, identical
    fit, identical pose, score 0.7714 against its own crop context vs 0.7205
    against the full cloud.

    Pooling two sources and keeping "whichever scored higher" is therefore
    invalid unless both scores are computed against the SAME context.
    """
    if len(region_points) < _MIN_POINTS_FOR_DETECTION:
        return ()
    try:
        result = detect_blocks_detailed_cpp(
            region_points, params=params or DetectionParameters(), sensor_context=sensor_context
        )
    except (RuntimeError, ValueError):
        return ()
    assert result.trace is not None
    return result.trace.raw_hypotheses


class DinoProposerRegions:
    """Lazy dino_proposer backend: one dense probability map per image."""

    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self._head = None
        self._config = None

    def _load(self) -> None:
        self._head, self._config = load_dino_region_proposer(self.checkpoint)

    def propose(self, rgb: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one bool mask per predicted region, already at `rgb`'s resolution."""
        if self._head is None:
            self._load()
        assert self._head is not None and self._config is not None
        probability_map = self._head.predict_proposal_map(rgb)
        proposals = extract_region_proposals(probability_map, self._config)
        return tuple(p.mask for p in proposals)


def dino_candidates_for_snapshot(
    proposer: DinoProposerRegions,
    points: np.ndarray,
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    sensor_origin: np.ndarray,
    *,
    mask_params: MaskSelectionParams,
    detection_params: DetectionParameters | None = None,
    score_context: str = "full",
) -> tuple[list[PooledCandidate], dict]:
    """dino_proposer region -> mask-select -> existing-fitter pipeline.

    Structurally identical to `eval_sam_region_proposals.sam_candidates_for_snapshot`,
    minus the pixel-area filter (`extract_region_proposals` already drops
    components under `config.min_component_area_px`).

    `score_context` selects which SensorContext the region fit is SCORED
    against -- `"full"` (the whole cloud, making scores comparable with the
    PCL candidates they are pooled against) or `"crop"` (the region's own
    points, reproducing the original, invalid behaviour). See
    `fit_region_scored_against`.
    """
    masks = proposer.propose(rgb)
    cache: ProjectionCache = build_projection_cache(points, T_world_camera, camera)
    full_context = SensorContext.from_points(sensor_origin, points)

    survived_projection = 0
    candidates: list[PooledCandidate] = []
    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        prompt_xy = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
        selection = select_points_from_mask(points, cache, mask, prompt_xy, mask_params)
        if not selection.success:
            continue
        survived_projection += 1
        if score_context == "full":
            hypotheses = fit_region_scored_against(
                points[selection.indices], full_context, detection_params
            )
        else:
            hypotheses = _run_detection(points[selection.indices], sensor_origin, detection_params)
        for hypothesis in hypotheses:
            if hypothesis.fate != "final":
                continue
            candidates.append(
                PooledCandidate(
                    f"dino:{i}:{hypothesis.id}",
                    hypothesis.pose.position,
                    hypothesis.pose.rotation,
                    hypothesis.pose.dims,
                    float(hypothesis.score),
                    "dino_proposer",
                )
            )
    diagnostics = {
        "regions_generated": len(masks),
        "regions_survived_projection_filter": survived_projection,
        "dino_accepted_hypotheses": len(candidates),
    }
    return candidates, diagnostics


# --------------------------------------------------------------------------
# dino_verifier scoring of a pooled candidate (new -- no SAM2-tool equivalent)
# --------------------------------------------------------------------------


def verifier_adjusted_scores(
    candidates: list[PooledCandidate],
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    verifier_head,
    verifier_config,
    calibration_temperature: float,
    *,
    crop_margin_frac: float,
    out_size: int,
) -> dict[str, float]:
    """`{candidate.id: geometry_score + verifier_contribution}` for every candidate.

    Renders each candidate's own crop + conditioning image (no shared union
    crop -- there is no "one annotated target" grouping at inference time,
    unlike training export), matching `crop_margin_frac`/`out_size` to
    `experiments/rgb_candidate_prior/export.py`'s training-time defaults so
    the crop distribution the head sees matches what it was trained on.
    """
    adjusted: dict[str, float] = {}
    for candidate in candidates:
        pose = candidate.to_pose()
        _, uv, in_front = render.project_corners(pose, T_world_camera, camera)
        bbox = render.crop_bbox(uv, in_front, camera, margin_frac=crop_margin_frac)
        rgb_crop = conditioning = None
        if bbox is not None:
            rgb_crop = render.render_crop(rgb, bbox, out_size=out_size)
            conditioning = render.render_conditioning(pose, T_world_camera, camera, bbox, out_size=out_size)
        result = score_candidate_with_dino(
            verifier_head,
            verifier_config,
            rgb_crop,
            conditioning,
            calibration_temperature=calibration_temperature,
        )
        adjusted[candidate.id] = candidate.score + result.contribution
    return adjusted


# --------------------------------------------------------------------------
# Snapshot manifest + GT loading (identical to eval_sam_region_proposals.py)
# --------------------------------------------------------------------------


def load_manifest_snapshots(manifest_path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(manifest_path.read_text()) or {}
    result = []
    for scene, info in (data.get("scenes") or {}).items():
        for entry in info.get("annotated") or []:
            result.append((scene, str(entry["name"])))
    return result


def approved_targets(doc: AnnotationDocument) -> list[EditableTarget]:
    return [t for t in doc.targets if t.review_status == "approved"]


# --------------------------------------------------------------------------
# Per-snapshot evaluation
# --------------------------------------------------------------------------


@dataclass
class SnapshotResult:
    scene: str
    name: str
    gt: list[EditableTarget]
    baseline_candidates: list[PooledCandidate]
    dino_candidates: list[PooledCandidate]
    pooled_candidates: list[PooledCandidate]
    reranked_candidates: list[PooledCandidate]
    baseline_matched_gt: set[int]
    pooled_matched_gt: set[int]
    reranked_matched_gt: set[int]
    baseline_strict_gt: set[int]
    pooled_strict_gt: set[int]
    reranked_strict_gt: set[int]
    dino_diagnostics: dict


def evaluate_snapshot(
    scene: str,
    name: str,
    dataset_root: Path,
    proposer: DinoProposerRegions,
    verifier_head,
    verifier_config,
    verifier_temperature: float,
    camera: CameraModel,
    *,
    dedup_radius: float,
    match_dist: float,
    crop_margin_frac: float,
    out_size: int,
    mask_params: MaskSelectionParams,
    detection_params: DetectionParameters | None = None,
    score_context: str = "full",
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

    raw_hypotheses = _run_detection(points, sensor_origin, detection_params)
    baseline_candidates = [
        PooledCandidate(h.id, h.pose.position, h.pose.rotation, h.pose.dims, float(h.score), "pcl")
        for h in raw_hypotheses
        if h.fate == "final"
    ]

    dino_candidates, dino_diag = dino_candidates_for_snapshot(
        proposer,
        points,
        rgb,
        tf.T_world_camera,
        camera,
        sensor_origin,
        mask_params=mask_params,
        detection_params=detection_params,
        score_context=score_context,
    )

    all_candidates = baseline_candidates + dino_candidates
    pooled_candidates = dedupe_candidates(all_candidates, dedup_radius)

    # Score the PRE-dedup pool: the rerank must decide which of two competing
    # candidates for one block survives, so both must still be present. Scoring
    # `pooled_candidates` (already pairwise separated by `dedup_radius`) and
    # re-deduping it at the same radius is a mathematical no-op -- every
    # candidate trivially passes the `> radius_m` test against every other, so
    # the result is identical to the input regardless of score.
    adjusted = verifier_adjusted_scores(
        all_candidates,
        rgb,
        tf.T_world_camera,
        camera,
        verifier_head,
        verifier_config,
        verifier_temperature,
        crop_margin_frac=crop_margin_frac,
        out_size=out_size,
    )
    reranked_candidates = dedupe_candidates(all_candidates, dedup_radius, scores=adjusted)

    gt_poses = [BlockPose(t.position, t.rotation, dims=t.dimensions) for t in gt]
    baseline_pairs = match_detections(
        [c.to_pose() for c in baseline_candidates], gt_poses, max_dist=match_dist
    )
    pooled_pairs = match_detections(
        [c.to_pose() for c in pooled_candidates], gt_poses, max_dist=match_dist
    )
    reranked_pairs = match_detections(
        [c.to_pose() for c in reranked_candidates], gt_poses, max_dist=match_dist
    )

    def strict_pairs(pairs, candidates):
        kept = set()
        for di, gi in pairs:
            if strict_match(candidates[di].to_pose(), gt_poses[gi])["strict_match"]:
                kept.add(gi)
        return kept

    return SnapshotResult(
        scene=scene,
        name=name,
        gt=gt,
        baseline_candidates=baseline_candidates,
        dino_candidates=dino_candidates,
        pooled_candidates=pooled_candidates,
        reranked_candidates=reranked_candidates,
        baseline_matched_gt={gi for _, gi in baseline_pairs},
        pooled_matched_gt={gi for _, gi in pooled_pairs},
        reranked_matched_gt={gi for _, gi in reranked_pairs},
        baseline_strict_gt=strict_pairs(baseline_pairs, baseline_candidates),
        pooled_strict_gt=strict_pairs(pooled_pairs, pooled_candidates),
        reranked_strict_gt=strict_pairs(reranked_pairs, reranked_candidates),
        dino_diagnostics=dino_diag,
    )


# --------------------------------------------------------------------------
# Aggregation + reporting
# --------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def _row(label: str, matched: int, gt_total: int, det_total: int, strict: int) -> str:
    return (
        f"| {label} | {_rate(matched, gt_total)} | {_rate(matched, det_total)} "
        f"| {_rate(strict, gt_total)} |"
    )


def build_report(results: list[SnapshotResult], args: argparse.Namespace) -> str:
    gt_total = sum(len(r.gt) for r in results)
    baseline_det = sum(len(r.baseline_candidates) for r in results)
    baseline_matched = sum(len(r.baseline_matched_gt) for r in results)
    baseline_strict = sum(len(r.baseline_strict_gt) for r in results)
    pooled_det = sum(len(r.pooled_candidates) for r in results)
    pooled_matched = sum(len(r.pooled_matched_gt) for r in results)
    pooled_strict = sum(len(r.pooled_strict_gt) for r in results)
    reranked_det = sum(len(r.reranked_candidates) for r in results)
    reranked_matched = sum(len(r.reranked_matched_gt) for r in results)
    reranked_strict = sum(len(r.reranked_strict_gt) for r in results)

    lines: list[str] = []
    lines.append("# Composed pipeline ablation: geometry -> +dino_proposer -> +dino_verifier")
    lines.append("")
    lines.append(
        "**In-sample numbers** (see module docstring): the loaded checkpoints are trained on "
        "the union of all 25 annotated snapshots this reports on, unless `--proposer-checkpoint`/"
        "`--verifier-checkpoint` point at a held-out-fold pair instead."
    )
    lines.append("")
    lines.append("| Stage | Recall (centre) | Precision (centre) | Recall (strict) |")
    lines.append("|---|---|---|---|")
    lines.append(_row("geometry (whole-cloud PCL)", baseline_matched, gt_total, baseline_det, baseline_strict))
    lines.append(_row("+ dino_proposer (pooled)", pooled_matched, gt_total, pooled_det, pooled_strict))
    lines.append(
        _row("+ dino_proposer + dino_verifier (reranked)", reranked_matched, gt_total, reranked_det, reranked_strict)
    )
    lines.append("")
    lines.append(
        "**Strict** additionally requires symmetry-aware rotation error <= "
        f"{DEFAULT_MAX_ROTATION_DEG:.0f} deg and oriented-box IoU >= {DEFAULT_MIN_IOU:.2f}."
    )
    lines.append("")

    baseline_misses = [
        (r, gi) for r in results for gi in range(len(r.gt)) if gi not in r.baseline_matched_gt
    ]
    recovered_pooled = [(r, gi) for r, gi in baseline_misses if gi in r.pooled_matched_gt]
    recovered_reranked = [(r, gi) for r, gi in baseline_misses if gi in r.reranked_matched_gt]

    def new_fps(matched_det_attr: str, candidates_attr: str) -> int:
        count = 0
        for r in results:
            candidates = getattr(r, candidates_attr)
            matched_det = getattr(r, matched_det_attr)
            matched_ids = {candidates[di].id for di in matched_det}
            for c in candidates:
                if c.source == "dino_proposer" and c.id not in matched_ids:
                    count += 1
        return count

    lines.append("## Miss recovery")
    lines.append("")
    lines.append(
        f"Of {len(baseline_misses)} GT blocks missed by whole-cloud PCL alone: "
        f"pooling `dino_proposer` regions recovers {len(recovered_pooled)}; "
        f"adding `dino_verifier` reranking recovers {len(recovered_reranked)}."
    )
    lines.append("")

    lines.append("## Per-scene breakdown")
    lines.append("")
    lines.append("| Scene | GT | geometry | + dino_proposer | + dino_proposer + verifier |")
    lines.append("|---|---|---|---|---|")
    for scene in sorted({r.scene for r in results}):
        scene_results = [r for r in results if r.scene == scene]
        gt_n = sum(len(r.gt) for r in scene_results)
        b = sum(len(r.baseline_matched_gt) for r in scene_results)
        p = sum(len(r.pooled_matched_gt) for r in scene_results)
        rr = sum(len(r.reranked_matched_gt) for r in scene_results)
        lines.append(f"| {scene} | {gt_n} | {b}/{gt_n} | {p}/{gt_n} | {rr}/{gt_n} |")
    lines.append("")

    total_regions = sum(r.dino_diagnostics["regions_generated"] for r in results)
    survived_proj = sum(r.dino_diagnostics["regions_survived_projection_filter"] for r in results)
    total_accepted = sum(r.dino_diagnostics["dino_accepted_hypotheses"] for r in results)
    lines.append("## dino_proposer region funnel")
    lines.append("")
    lines.append(
        f"Across all snapshots: {total_regions} regions generated -> {survived_proj} survived "
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
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--proposer-checkpoint", type=Path, default=DEFAULT_PROPOSER_CHECKPOINT)
    parser.add_argument("--verifier-checkpoint", type=Path, default=DEFAULT_VERIFIER_CHECKPOINT)
    parser.add_argument("--dedup-radius", type=float, default=0.25)
    parser.add_argument("--match-dist", type=float, default=0.30)
    parser.add_argument("--crop-margin-frac", type=float, default=0.25)
    parser.add_argument("--out-size", type=int, default=224)
    parser.add_argument(
        "--cluster-max-extent-xy", type=float, default=DetectionParameters().cluster_max_extent_xy
    )
    parser.add_argument("--dbscan-eps", type=float, default=DetectionParameters().dbscan_eps)
    parser.add_argument(
        "--score-context",
        choices=("full", "crop"),
        default="full",
        help="SensorContext used to SCORE region-sourced fits. 'full' (default) scores them "
        "against the whole cloud so their scores are comparable with the PCL candidates they "
        "are pooled against; 'crop' scores each against its own region points, which inflates "
        "them (60%% of the score is ray evidence a small crop cannot accrue violations against) "
        "and is kept only to reproduce the original, invalid measurement.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    camera = load_camera_model(args.calibration)
    manifest = load_manifest_snapshots(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    detection_params = replace(
        DetectionParameters(),
        cluster_max_extent_xy=args.cluster_max_extent_xy,
        dbscan_eps=args.dbscan_eps,
    )
    proposer = DinoProposerRegions(args.proposer_checkpoint, device=args.device)
    verifier_head, verifier_config, verifier_temperature = load_dino_verifier(args.verifier_checkpoint)
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
                verifier_head,
                verifier_config,
                verifier_temperature,
                camera,
                dedup_radius=args.dedup_radius,
                match_dist=args.match_dist,
                crop_margin_frac=args.crop_margin_frac,
                out_size=args.out_size,
                mask_params=mask_params,
                detection_params=detection_params,
                score_context=args.score_context,
            )
        )

    stamp = experiment_stamp(
        detector=detection_params,
        match_dist=args.match_dist,
        dedup_radius=args.dedup_radius,
        proposer_checkpoint=str(args.proposer_checkpoint),
        verifier_checkpoint=str(args.verifier_checkpoint),
    )
    report = build_report(results, args)
    report = f"<!-- provenance: commit {stamp['git_commit']} -->\n" + report
    print(report)
    if args.report_out is not None:
        args.report_out.write_text(report)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "provenance": stamp,
                    "snapshots": [
                        {
                            "scene": r.scene,
                            "name": r.name,
                            "gt": len(r.gt),
                            "baseline_matched": len(r.baseline_matched_gt),
                            "baseline_det": len(r.baseline_candidates),
                            "pooled_matched": len(r.pooled_matched_gt),
                            "pooled_det": len(r.pooled_candidates),
                            "reranked_matched": len(r.reranked_matched_gt),
                            "reranked_det": len(r.reranked_candidates),
                            **r.dino_diagnostics,
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
