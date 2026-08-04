"""Multi-source candidate blocks for the interactive snapshot annotator.

Wires together the four already-existing block detection/scoring sources so
an annotator can compare them for one snapshot, without duplicating any of
their detection or scoring logic:

- ``pcl``: pure PCL geometry -- the raw hypotheses from
  ``blockpose.detection.detect_blocks_detailed``.
- ``rgb_edge_prior``: the classical RGB edge validator/reranker
  (``blockpose.rgb_edge_prior``), scoring the same raw PCL hypotheses. It does
  not seed or detect blocks independently; the internal key is retained for
  API and on-disk compatibility.
- ``dino_proposer``: proposal-driven 3-D candidates from the learned
  full-image DINOv2 region proposer (``blockpose.rgb_region_proposer`` /
  ``blockpose.rgb_proposals``).
- ``dino_verifier_geometry_only`` / ``dino_verifier_reranked``: bounded
  DINOv2 verifier reranking (``blockpose.rgb_verifier``), mirroring
  ``tools/ablate_rgb_verifier.py``'s ``_rank`` band-bounded reordering rule
  (RGB/DINO evidence may only reorder candidates within one geometry score
  band, never override geometry across bands).

This module does no file I/O and no GUI/Open3D rendering: callers supply
already-loaded arrays/objects and get back only plain poses/scores/dataclasses.
Optional (DINO) sources degrade to an explicit warning, never a crash or a
silent no-op, and one source's failure never prevents computing the others.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blockpose.annotation_preprocess import filter_exclusion_boxes
from blockpose.core_bridge import detect_blocks_detailed_cpp
from blockpose.core_types import DetectionParameters, SensorContext, TraceHypothesis
from blockpose.io import CameraModel
from blockpose.pose import BlockPose
from blockpose.rgb_proposals import (
    DinoRegionProposalProvider,
    MaskAssociationParams,
    associate_mask_to_lidar,
    proposal_point_indices,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIN_POINTS_FOR_DETECTION = 30
_CROP_MARGIN_FRAC = 0.2
_CROP_OUT_SIZE = 224
_CANDIDATE_GROUP_RADIUS_M = 0.50
_DEFAULT_SCORE_BAND = 0.05
_DEFAULT_DINO_MARGIN = 0.08


@dataclass(frozen=True)
class SourceCandidate:
    """One candidate block as seen by a single detection/scoring source."""

    id: str
    pose: BlockPose
    score: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class CandidateSourcesResult:
    pcl: tuple[SourceCandidate, ...]
    rgb_edge_prior: tuple[SourceCandidate, ...]
    dino_proposer: tuple[SourceCandidate, ...]
    dino_verifier_geometry_only: tuple[SourceCandidate, ...]
    dino_verifier_reranked: tuple[SourceCandidate, ...]
    warnings: list[str]


def _run_detection(
    points: np.ndarray,
    sensor_origin: np.ndarray | None,
    params: DetectionParameters | None = None,
) -> tuple[TraceHypothesis, ...]:
    """Raw hypotheses for one point set, or `()` for degenerate/failing input.

    `sensor_origin` (the LiDAR's world position, `tf.T_world_cloud[:3, 3]`)
    builds a real `SensorContext` exactly like `tools/ablate_rgb_*.py` does,
    so free-space/visible-face/ray-explained-surface evidence terms (0.60 of
    the total score weight, formerly in `detection.py::_SCORE_WEIGHTS`) are
    computed instead of hard-zeroed. Falls back to `sensor_context=None` only
    when no origin is available at all -- callers should always pass one.

    `params` defaults to `DetectionParameters()`. Experiments MUST pass it
    explicitly: this function used to hardcode the defaults, so when
    `cluster_max_extent_xy` shipped at 2.25 an already-written report measured
    at 1.5 silently stopped reproducing from its own documented command. A
    recorded result is only meaningful next to the parameters that produced it.
    """
    if len(points) < _MIN_POINTS_FOR_DETECTION:
        return ()
    sensor_context = (
        SensorContext.from_points(sensor_origin, points) if sensor_origin is not None else None
    )
    try:
        result = detect_blocks_detailed_cpp(
            points, params=params or DetectionParameters(), sensor_context=sensor_context
        )
    except (RuntimeError, ValueError):
        return ()
    assert result.trace is not None
    return result.trace.raw_hypotheses


def _reject_hypotheses_above_gripper(
    hypotheses: tuple[TraceHypothesis, ...],
    above_gripper_boxes: dict[str, np.ndarray] | None,
) -> tuple[TraceHypothesis, ...]:
    """Drop raw hypotheses centred inside the above-gripper exclusion volume.

    The point-cloud self-filter (`annotation_preprocess.filter_pzs100_gripper`)
    already removes points above the jaws before detection ever runs, but a
    partial geometric fit can still be built almost entirely from points
    outside that volume (e.g. hugging a boom edge) and end up CENTRED inside
    it -- the point filter alone cannot catch that. This is a second,
    independent guard: reject any hypothesis whose pose centre falls inside
    the same above-gripper volume.

    Reuses the native oriented-box point test (via `filter_exclusion_boxes`)
    rather than reimplementing the box geometry in Python, per the
    single-source-of-geometry-truth convention -- the C++ core owns what
    "inside the box" means.

    `above_gripper_boxes` is the `above_centers`/`above_rotations`/
    `above_sizes` triple returned by `filter_pzs100_gripper`. `None`, or zero
    boxes (feature off, or no gripper geometry available), is a no-op --
    callers that don't have gripper geometry are unaffected.
    """
    if not hypotheses or above_gripper_boxes is None:
        return hypotheses
    # `.get()`, not indexing: callers pass the whole `filter_pzs100_gripper`
    # diagnostics dict, and an older compiled `_core` predates these keys.
    # `compute_candidate_sources` documents that it never raises, so a stale
    # extension must degrade to "no above-gripper rejection", not KeyError.
    above_centers = np.asarray(above_gripper_boxes.get("above_centers", ()), dtype=np.float64)
    if above_centers.shape[0] == 0:
        return hypotheses
    above_rotations = np.asarray(above_gripper_boxes.get("above_rotations", ()), dtype=np.float64)
    above_sizes = np.asarray(above_gripper_boxes.get("above_sizes", ()), dtype=np.float64)
    if above_rotations.shape != (above_centers.shape[0], 3, 3) or (
        above_sizes.shape != above_centers.shape
    ):
        return hypotheses
    kept = []
    for hypothesis in hypotheses:
        position = np.ascontiguousarray(hypothesis.pose.position, dtype=np.float64).reshape(1, 3)
        _, removed = filter_exclusion_boxes(position, above_centers, above_rotations, above_sizes)
        if removed == 0:
            kept.append(hypothesis)
    return tuple(kept)


def _pcl_candidates(raw_hypotheses: tuple[TraceHypothesis, ...]) -> tuple[SourceCandidate, ...]:
    return tuple(
        SourceCandidate(h.id, h.pose, float(h.score), h.fate == "final", h.fate)
        for h in raw_hypotheses
    )


def _rgb_edge_prior_candidates(
    raw_hypotheses: tuple[TraceHypothesis, ...],
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
) -> tuple[SourceCandidate, ...]:
    from blockpose import _core

    if not getattr(_core, "vision_enabled", False):
        raise RuntimeError("C++ RGB edge validator/reranker is not enabled in blockpose._core")
    gray = np.ascontiguousarray(
        np.mean(np.asarray(rgb, dtype=np.float64)[..., :3], axis=2).clip(0, 255), dtype=np.uint8
    )
    scorer = _core.RgbEdgePrior(
        gray,
        np.ascontiguousarray(camera.K, dtype=np.float64),
        np.ascontiguousarray(camera.D, dtype=np.float64),
        np.ascontiguousarray(T_world_camera, dtype=np.float64),
    )
    candidates = []
    for h in raw_hypotheses:
        score = scorer.score(
            np.ascontiguousarray(h.pose.position, dtype=np.float64),
            np.ascontiguousarray(h.pose.rotation, dtype=np.float64),
            np.ascontiguousarray(h.pose.dims, dtype=np.float64),
        )
        candidates.append(
            SourceCandidate(h.id, h.pose, score["score"], score["available"], "native_rgb")
        )
    return tuple(candidates)


def _dino_proposer_candidates(
    points: np.ndarray,
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    checkpoint: Path,
    sensor_origin: np.ndarray | None,
    above_gripper_boxes: dict[str, np.ndarray] | None,
) -> tuple[tuple[SourceCandidate, ...], str | None]:
    """Proposal-driven 3-D candidates, or `((), warning)` if the source fails.

    Constructing/running the provider is a single external-dependency
    boundary (missing torch/transformers, an unreadable or corrupt
    checkpoint, ...); catching broadly here is deliberate so that any such
    failure degrades to a warning instead of preventing the other three
    sources from being computed.
    """
    try:
        provider = DinoRegionProposalProvider(checkpoint=checkpoint)
        batch = provider.propose(rgb)
    except Exception as error:  # noqa: BLE001 - see docstring
        return (), f"dino_proposer unavailable: {error!r}"

    params = MaskAssociationParams()
    candidates: list[SourceCandidate] = []
    for proposal_index, proposal in enumerate(batch.proposals):
        try:
            association = associate_mask_to_lidar(points, T_world_camera, camera, proposal, params)
            indices = proposal_point_indices(np.empty(0, dtype=int), association, mode="union")
        except ValueError:
            # A malformed single proposal (e.g. mask shape mismatch) must not
            # discard every other proposal's candidates.
            continue
        proposal_hypotheses = _reject_hypotheses_above_gripper(
            _run_detection(points[indices], sensor_origin), above_gripper_boxes
        )
        for hypothesis in proposal_hypotheses:
            candidates.append(
                SourceCandidate(
                    f"dino_proposer:{proposal_index}:{hypothesis.id}",
                    hypothesis.pose,
                    float(hypothesis.score),
                    hypothesis.fate == "final",
                    f"proposal {proposal_index} ({hypothesis.fate})",
                )
            )
    return tuple(candidates), None


def _rank_bounded(
    candidates: list[dict], *, score_band: float, dino_bands: set[int] | None
) -> list[dict]:
    """Geometry-band-bounded reordering.

    Mirrors `tools/ablate_rgb_verifier.py::_rank` (itself mirroring
    `tools/ablate_rgb_edge_prior.py::_rank`): geometry order is kept across
    score bands, DINO evidence can only reorder candidates within one band.
    Reimplemented here rather than imported because `tools/` has no
    `__init__.py` -- it is not an importable package from inside `blockpose/`,
    and importing the ablate script as a module would run its own
    `sys.path`/cross-import side effects just to reach this small helper.
    """
    best = max(item["geometry_score"] for item in candidates)
    for item in candidates:
        item["geometry_band"] = int(np.floor((best - item["geometry_score"]) / score_band))
    return sorted(
        candidates,
        key=lambda item: (
            item["geometry_band"],
            -(
                item["dino_contribution"]
                if dino_bands is not None
                and item["geometry_band"] in dino_bands
                and not item["dino_abstained"]
                else 0.0
            ),
            -item["geometry_score"],
            item["id"],
        ),
    )


def _group_by_proximity(
    hypotheses: tuple[TraceHypothesis, ...], radius_m: float
) -> list[list[TraceHypothesis]]:
    """Union-find grouping of raw hypotheses whose positions are mutually close.

    The DINO verifier must score every candidate competing for one physical
    target against identical pixels (see `ablate_rgb_verifier.py::evaluate_snapshot`'s
    shared crop), but a whole snapshot can contain several unrelated blocks
    far apart in the image; grouping every raw hypothesis into one shared crop
    would blur those together. Grouping by mutual position proximity recovers
    the same-target grouping the ablation gets from ground-truth radius
    matching, without requiring ground truth.
    """
    n = len(hypotheses)
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(hypotheses[i].pose.position - hypotheses[j].pose.position)
            if distance <= radius_m:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[max(root_i, root_j)] = min(root_i, root_j)
    groups: dict[int, list[TraceHypothesis]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(hypotheses[index])
    return list(groups.values())


def _dino_verifier_views(
    raw_hypotheses: tuple[TraceHypothesis, ...],
    rgb: np.ndarray | None,
    T_world_camera: np.ndarray | None,
    camera: CameraModel | None,
    checkpoint: Path | None,
    *,
    score_band: float,
    dino_margin: float,
) -> tuple[tuple[SourceCandidate, ...], tuple[SourceCandidate, ...], str | None]:
    """Return `(geometry_only, reranked, warning_if_reranked_unavailable)`.

    `geometry_only` needs no RGB/checkpoint at all -- it is the same raw
    hypotheses in the band-bounded rank order with no DINO contribution,
    i.e. what `reranked` degrades to whenever DINO evidence is unavailable.
    """
    records = [
        {
            "id": h.id,
            "pose": h.pose,
            "geometry_score": float(h.score),
            "dino_contribution": 0.0,
            "dino_abstained": True,
            "dino_reason": "not_scored",
        }
        for h in raw_hypotheses
    ]
    if not records:
        return (), (), None

    geometry_ranked = _rank_bounded(
        [dict(r) for r in records], score_band=score_band, dino_bands=None
    )
    geometry_only = tuple(
        SourceCandidate(item["id"], item["pose"], item["geometry_score"], True, "geometry_only")
        for item in geometry_ranked
    )

    if checkpoint is None:
        return geometry_only, (), "dino_verifier unavailable: no checkpoint given"
    if rgb is None or T_world_camera is None or camera is None:
        return (
            geometry_only,
            (),
            "dino_verifier unavailable: rgb/T_world_camera/camera not all provided",
        )

    # Loading the checkpoint and importing the (torch-free, but repo-root-relative)
    # conditioning renderer are both external-dependency boundaries; catching
    # broadly here mirrors `_dino_proposer_candidates` so a bad checkpoint or a
    # broken import never prevents the other sources from being returned.
    try:
        from blockpose.rgb_verifier import load_dino_verifier, score_candidate_with_dino

        head, config, calibration_temperature = load_dino_verifier(checkpoint)

        repo_root = str(_REPO_ROOT)
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from experiments.rgb_candidate_prior.render import (
            crop_bbox,
            project_corners,
            render_conditioning,
            render_crop,
            union_crop_bbox,
        )
    except Exception as error:  # noqa: BLE001 - see docstring
        return geometry_only, (), f"dino_verifier unavailable: {error!r}"

    scored: dict[str, dict] = {}
    for group in _group_by_proximity(raw_hypotheses, _CANDIDATE_GROUP_RADIUS_M):
        candidate_bboxes = []
        for hypothesis in group:
            _, uv, in_front = project_corners(hypothesis.pose, T_world_camera, camera)
            bbox = crop_bbox(uv, in_front, camera, margin_frac=_CROP_MARGIN_FRAC)
            if bbox is not None:
                candidate_bboxes.append(bbox)
        shared_bbox = union_crop_bbox(candidate_bboxes, camera)
        if shared_bbox is None:
            for hypothesis in group:
                scored[hypothesis.id] = {
                    "contribution": 0.0,
                    "abstained": True,
                    "reason": "not_projectable",
                }
            continue
        crop = render_crop(rgb, shared_bbox, out_size=_CROP_OUT_SIZE)
        for hypothesis in group:
            conditioning = render_conditioning(
                hypothesis.pose, T_world_camera, camera, shared_bbox, out_size=_CROP_OUT_SIZE
            )
            result = score_candidate_with_dino(
                head, config, crop, conditioning, calibration_temperature=calibration_temperature
            )
            scored[hypothesis.id] = {
                "contribution": result.contribution,
                "abstained": result.abstained,
                "reason": result.reason,
            }

    for record in records:
        info = scored.get(
            record["id"], {"contribution": 0.0, "abstained": True, "reason": "not_scored"}
        )
        record["dino_contribution"] = info["contribution"]
        record["dino_abstained"] = info["abstained"]
        record["dino_reason"] = info["reason"]

    banded = _rank_bounded([dict(r) for r in records], score_band=score_band, dino_bands=None)
    by_band: dict[int, list[dict]] = {}
    for item in banded:
        by_band.setdefault(item["geometry_band"], []).append(item)
    dino_bands = {
        band
        for band, members in by_band.items()
        if len(
            top_scores := sorted(
                (m["dino_contribution"] for m in members if not m["dino_abstained"]), reverse=True
            )
        )
        >= 2
        and top_scores[0] - top_scores[1] >= dino_margin
    }
    reranked_order = _rank_bounded(
        [dict(r) for r in records], score_band=score_band, dino_bands=dino_bands
    )
    reranked = tuple(
        SourceCandidate(
            item["id"],
            item["pose"],
            item["dino_contribution"],
            item["geometry_band"] in dino_bands and not item["dino_abstained"],
            item["dino_reason"],
        )
        for item in reranked_order
    )
    return geometry_only, reranked, None


def compute_candidate_sources(
    points: np.ndarray,
    rgb: np.ndarray | None,
    T_world_camera: np.ndarray | None,
    camera: CameraModel | None,
    *,
    T_world_cloud: np.ndarray | None = None,
    dino_proposer_checkpoint: Path | None = None,
    dino_verifier_checkpoint: Path | None = None,
    above_gripper_boxes: dict[str, np.ndarray] | None = None,
    detection_params: DetectionParameters | None = None,
) -> CandidateSourcesResult:
    """Compute candidate blocks from all four sources for one snapshot.

    `points` must already be in world frame (as returned by
    `blockpose.io.load_snapshot`). `T_world_cloud` (the loaded `tf.yaml`'s
    cloud transform) supplies the LiDAR's world position for real free-space/
    visibility evidence, exactly like `tools/ablate_rgb_*.py` -- omitting it
    silently zeroes 0.60 of the score weight and even changes which
    hypotheses survive, so callers that have a `tf.yaml` loaded should always
    pass it. RGB-dependent sources require `rgb`, `T_world_camera`, and
    `camera` together; DINO sources additionally require their checkpoint.
    Any unmet requirement or failure degrades to an empty tuple plus an
    explanatory entry in `warnings`, never an exception.

    `above_gripper_boxes` is the optional `above_centers`/`above_rotations`/
    `above_sizes` diagnostics triple from `annotation_preprocess.
    filter_pzs100_gripper` (the PZS100 above-jaws exclusion volume covering
    the carrier/boom/arm). When given, any raw hypothesis whose pose centre
    falls inside that volume is rejected before scoring, on top of whatever
    the point-cloud self-filter already removed -- a partial fit can be built
    from points outside the volume yet still land centred inside it. `None`
    (the default) keeps every source's behaviour unchanged for callers with
    no gripper geometry available.
    """
    warnings: list[str] = []
    points = np.asarray(points, dtype=np.float64)
    have_rgb_context = rgb is not None and T_world_camera is not None and camera is not None
    sensor_origin = T_world_cloud[:3, 3] if T_world_cloud is not None else None
    if sensor_origin is None:
        warnings.append(
            "sensor_context unavailable: no T_world_cloud given -- free-space/visibility "
            "evidence terms are zeroed, PCL/derived scores are not comparable to production"
        )

    raw_hypotheses = _reject_hypotheses_above_gripper(
        _run_detection(points, sensor_origin, detection_params), above_gripper_boxes
    )
    pcl = _pcl_candidates(raw_hypotheses)

    if have_rgb_context:
        assert rgb is not None and T_world_camera is not None and camera is not None
        # RGB vision is an optional native capability.  Keep this call behind
        # the same per-source failure boundary as the DINO backends: a
        # geometry-only build (or any failure while constructing/scoring the
        # prior) must not discard the already-computed PCL candidates.
        try:
            rgb_edge_prior = _rgb_edge_prior_candidates(raw_hypotheses, rgb, T_world_camera, camera)
        except Exception as error:  # noqa: BLE001 - optional backend boundary
            rgb_edge_prior = ()
            warnings.append(f"RGB edge validator/reranker unavailable: {error!r}")
    else:
        rgb_edge_prior = ()
        warnings.append(
            "RGB edge validator/reranker unavailable: rgb/T_world_camera/camera not all provided"
        )

    dino_proposer: tuple[SourceCandidate, ...]
    if dino_proposer_checkpoint is None:
        dino_proposer = ()
        warnings.append("dino_proposer unavailable: no checkpoint given")
    elif not have_rgb_context:
        dino_proposer = ()
        warnings.append("dino_proposer unavailable: rgb/T_world_camera/camera not all provided")
    else:
        assert rgb is not None and T_world_camera is not None and camera is not None
        dino_proposer, proposer_warning = _dino_proposer_candidates(
            points,
            rgb,
            T_world_camera,
            camera,
            dino_proposer_checkpoint,
            sensor_origin,
            above_gripper_boxes,
        )
        if proposer_warning is not None:
            warnings.append(proposer_warning)

    dino_verifier_geometry_only, dino_verifier_reranked, verifier_warning = _dino_verifier_views(
        raw_hypotheses,
        rgb,
        T_world_camera,
        camera,
        dino_verifier_checkpoint,
        score_band=_DEFAULT_SCORE_BAND,
        dino_margin=_DEFAULT_DINO_MARGIN,
    )
    if verifier_warning is not None:
        warnings.append(verifier_warning)

    return CandidateSourcesResult(
        pcl=pcl,
        rgb_edge_prior=rgb_edge_prior,
        dino_proposer=dino_proposer,
        dino_verifier_geometry_only=dino_verifier_geometry_only,
        dino_verifier_reranked=dino_verifier_reranked,
        warnings=warnings,
    )
