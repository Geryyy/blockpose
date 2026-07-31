"""Contract tests for replayable RGB proposals and conservative association."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from blockpose.io import CameraModel
from blockpose.rgb_proposals import (
    DinoRegionProposalProvider,
    GeometryOnlyProposalProvider,
    ImageProposal,
    MaskAssociationParams,
    ProposalBatch,
    Sam2PointPromptProvider,
    StoredMaskProposalProvider,
    associate_mask_to_lidar,
    load_proposal_cache,
    make_cache_provenance,
    proposal_cache_key,
    proposal_point_indices,
    save_proposal_cache,
)
from blockpose.rgb_region_proposer import DinoRegionProposerConfig


def _camera() -> CameraModel:
    return CameraModel(np.array([[20, 0, 10], [0, 20, 10], [0, 0, 1]]), 20, 20)


def _proposal(mask: np.ndarray) -> ImageProposal:
    ys, xs = np.nonzero(mask)
    return ImageProposal(
        mask,
        (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
        provider="test",
    )


def test_stored_replay_is_deterministic_and_does_not_need_annotations(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    masks = snapshot / "masks"
    masks.mkdir(parents=True)
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    plt.imsave(snapshot / "rgb.png", rgb)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:7, 4:9] = 255
    plt.imsave(masks / "candidate.png", mask, cmap="gray", vmin=0, vmax=255)
    (snapshot / "metadata.yaml").write_text(
        "candidate_masks:\n  - {detection_id: 7, file: masks/candidate.png}\n"
    )

    result = StoredMaskProposalProvider().propose(rgb, snapshot=snapshot, timestamp_ns=12)

    assert result.provider == "stored_mask_replay"
    assert len(result.proposals) == 1
    assert result.proposals[0].bbox_xywh == (4, 3, 5, 4)
    assert result.proposals[0].provenance["source"] == "metadata.candidate_masks"


def test_cache_roundtrip_is_keyed_and_serializable(tmp_path: Path) -> None:
    snapshot = tmp_path / "s"
    snapshot.mkdir()
    (snapshot / "rgb.png").write_bytes(b"rgb")
    (snapshot / "metadata.yaml").write_text("candidate_masks: []\n")
    mask = np.zeros((20, 20), bool)
    mask[5:7, 5:8] = True
    payload = ProposalBatch((_proposal(mask),), "unit", (20, 20), 1.5, {"device": "cpu"})
    provenance = make_cache_provenance(
        snapshot, provider="unit", provider_config={"threshold": 0.5}, calibration="fixed"
    )
    key = proposal_cache_key(snapshot, "unit", provenance=provenance)
    target = tmp_path / "cache" / "proposal"
    save_proposal_cache(target, payload, cache_key=key, provenance=provenance)

    loaded = load_proposal_cache(target, expected_key=key, expected_provenance=provenance)
    assert loaded is not None
    assert loaded.provider == "unit"
    np.testing.assert_array_equal(loaded.proposals[0].mask, mask)
    assert (
        load_proposal_cache(target, expected_key="different", expected_provenance=provenance)
        is None
    )


def test_cache_rejects_mask_or_calibration_or_prompt_changes(tmp_path: Path) -> None:
    snapshot = tmp_path / "s"
    masks = snapshot / "masks"
    masks.mkdir(parents=True)
    (snapshot / "rgb.png").write_bytes(b"rgb")
    (masks / "candidate.png").write_bytes(b"initial mask")
    (snapshot / "metadata.yaml").write_text(
        "candidate_masks:\n  - {detection_id: 1, file: masks/candidate.png}\n"
    )
    mask = np.zeros((20, 20), bool)
    mask[5:7, 5:8] = True
    batch = ProposalBatch((_proposal(mask),), "stored", (20, 20), 0.0)
    provenance = make_cache_provenance(
        snapshot,
        provider="stored",
        calibration={"K": [1]},
        extrinsics={"T": [1]},
        prompt_config={"source": "none"},
    )
    cache_key = proposal_cache_key(snapshot, "stored", provenance=provenance)
    target = tmp_path / "cache" / "proposal"
    save_proposal_cache(target, batch, cache_key=cache_key, provenance=provenance)

    (masks / "candidate.png").write_bytes(b"changed mask")
    changed_asset = make_cache_provenance(
        snapshot,
        provider="stored",
        calibration={"K": [1]},
        extrinsics={"T": [1]},
        prompt_config={"source": "none"},
    )
    assert load_proposal_cache(
        target,
        expected_key=proposal_cache_key(snapshot, "stored", provenance=changed_asset),
        expected_provenance=changed_asset,
    ) is None
    changed_calibration = make_cache_provenance(
        snapshot,
        provider="stored",
        calibration={"K": [2]},
        extrinsics={"T": [1]},
        prompt_config={"source": "other"},
    )
    assert changed_calibration != provenance


def test_sam_cache_requires_resolved_revision_and_weights(tmp_path: Path) -> None:
    snapshot = tmp_path / "sam"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="resolved model_id"):
        make_cache_provenance(snapshot, provider="sam2_point_prompt", model_id="sam2")
    provenance = make_cache_provenance(
        snapshot,
        provider="sam2_point_prompt",
        model_id="sam2",
        model_revision="commit-abc",
        model_weights_sha256="0" * 64,
    )
    assert provenance.model_revision == "commit-abc"


def test_calibration_margin_recovers_jittered_projection_and_soft_weights() -> None:
    # The point projects one pixel right of a one-pixel mask: a 1 px calibration
    # uncertainty must retain it, while zero slack must not.
    mask = np.zeros((20, 20), bool)
    mask[10, 10] = True
    point = np.array([[0.05, 0.0, 1.0]])  # u=11, v=10 for the test camera
    proposal = _proposal(mask)
    strict = associate_mask_to_lidar(
        point, np.eye(4), _camera(), proposal, MaskAssociationParams(0, 0, 0, 3)
    )
    tolerant = associate_mask_to_lidar(
        point, np.eye(4), _camera(), proposal, MaskAssociationParams(0, 0.6, 2, 3)
    )

    assert strict.hard_indices.size == 0
    assert tolerant.calibration_margin_px == 2
    assert tolerant.hard_indices.tolist() == [0]
    assert tolerant.soft_weights.tolist() == [1.0]


def test_overlapping_masks_keep_independent_evidence_and_union_has_fallback() -> None:
    mask_a = np.zeros((20, 20), bool)
    mask_b = np.zeros((20, 20), bool)
    mask_a[9:12, 9:12] = True
    mask_b[10:13, 10:13] = True
    points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]])
    first = associate_mask_to_lidar(points, np.eye(4), _camera(), _proposal(mask_a))
    second = associate_mask_to_lidar(points, np.eye(4), _camera(), _proposal(mask_b))

    assert first.hard_indices.size > 0 and second.hard_indices.size > 0
    geometry = np.array([1])
    np.testing.assert_array_equal(proposal_point_indices(geometry, None, "union"), geometry)
    np.testing.assert_array_equal(
        proposal_point_indices(geometry, first, "union"), np.union1d(geometry, first.soft_indices)
    )


def test_image_dropout_is_explicit_geometry_fallback() -> None:
    batch = GeometryOnlyProposalProvider().propose(None)
    assert batch.proposals == ()
    assert batch.provenance["mode"] == "no_rgb"
    np.testing.assert_array_equal(proposal_point_indices(np.array([3, 1]), None, "union"), [1, 3])


def test_sam_provider_is_optional_and_reports_missing_runtime_only_on_use(monkeypatch) -> None:
    class UnavailableBackend:
        def segment(self, rgb: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
            raise RuntimeError("SAM2 unavailable")

    provider = Sam2PointPromptProvider(lambda image: [(1, 1)], backend=UnavailableBackend())
    with pytest.raises(RuntimeError, match="SAM2 unavailable"):
        provider.propose(np.zeros((4, 4, 3), np.uint8))


class _StubProposerHead:
    """Stands in for `DinoRegionProposerHead` in tests that must not require torch."""

    def __init__(self, probability_map: np.ndarray, config: DinoRegionProposerConfig):
        self.config = config
        self.device = "cpu"
        self._probability_map = probability_map
        self.calls: list[np.ndarray] = []

    def predict_proposal_map(self, rgb: np.ndarray) -> np.ndarray:
        self.calls.append(rgb)
        return self._probability_map


def _synthetic_probability_map(shape: tuple[int, int]) -> np.ndarray:
    probability_map = np.zeros(shape, dtype=np.float64)
    probability_map[2:9, 2:9] = 0.9  # 7x7 = 49 px, kept
    probability_map[20:22, 20:22] = 0.95  # 2x2 = 4 px, dropped by min_component_area_px
    return probability_map


def test_dino_region_proposal_provider_produces_valid_proposals_from_a_stub_head() -> None:
    shape = (40, 40)
    config = DinoRegionProposerConfig(min_component_area_px=10, foreground_threshold=0.5)
    head = _StubProposerHead(_synthetic_probability_map(shape), config)
    provider = DinoRegionProposalProvider(head)

    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    batch = provider.propose(rgb, timestamp_ns=42)

    assert batch.provider == "dino_region_proposer"
    assert batch.image_shape == shape  # ProposalBatch.__post_init__ enforces this per-proposal too
    assert len(batch.proposals) == 1
    proposal = batch.proposals[0]
    assert proposal.provider == "dino_region_proposer"
    assert proposal.model_id == config.model_id
    assert proposal.timestamp_ns == 42
    assert np.any(proposal.mask)
    x, y, w, h = proposal.bbox_xywh
    assert x >= 0 and y >= 0 and w >= 1 and h >= 1
    assert head.calls and head.calls[0] is rgb


def test_dino_region_proposal_provider_requires_head_or_checkpoint() -> None:
    with pytest.raises(ValueError, match="head or a checkpoint"):
        DinoRegionProposalProvider()


def test_dino_region_proposal_provider_rejects_missing_or_malformed_rgb() -> None:
    provider = DinoRegionProposalProvider(
        _StubProposerHead(np.zeros((4, 4)), DinoRegionProposerConfig())
    )
    with pytest.raises(ValueError, match="requires RGB"):
        provider.propose(None)
    with pytest.raises(ValueError, match="H, W, 3"):
        provider.propose(np.zeros((4, 4)))


def test_dino_region_proposal_provider_loads_checkpoint_lazily(tmp_path, monkeypatch) -> None:
    calls: list[object] = []
    stub_config = DinoRegionProposerConfig(min_component_area_px=1)
    stub_head = _StubProposerHead(np.full((10, 10), 0.9), stub_config)

    def fake_load(path):
        calls.append(path)
        return stub_head, stub_config

    monkeypatch.setattr("blockpose.rgb_region_proposer.load_dino_region_proposer", fake_load)
    checkpoint = tmp_path / "ckpt"
    provider = DinoRegionProposalProvider(checkpoint=checkpoint)
    assert calls == []  # nothing loaded until propose() is called

    batch = provider.propose(np.zeros((10, 10, 3), dtype=np.uint8))

    assert calls == [checkpoint]
    assert batch.provider == "dino_region_proposer"
    assert len(batch.proposals) == 1


def test_association_rejects_non_rigid_or_nonfinite_camera_transforms() -> None:
    mask = np.zeros((20, 20), bool)
    mask[10, 10] = True
    for transform in (np.full((4, 4), np.nan), np.diag([2.0, 1.0, 1.0, 1.0])):
        with pytest.raises(ValueError, match="T_world_camera|rotation"):
            associate_mask_to_lidar(
                np.array([[0.0, 0.0, 1.0]]), transform, _camera(), _proposal(mask)
            )
