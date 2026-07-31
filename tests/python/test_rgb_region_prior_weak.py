"""Tests for the weak-anchor exporter (``export_weak.py``) and ``train.py``'s
weak-sample weight blending (``sample_weights`` in ``train_head``).

Offline unit tests: no GPU, no network, synthetic captures/anchors only,
mirroring ``test_rgb_region_prior_export.py``'s synthetic-snapshot pattern and
``test_rgb_region_prior_training.py``'s ``_FakeRegionHead``/``_FakeDenseHead``
stub pattern.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.io import CameraModel  # noqa: E402
from blockpose.pose import BlockPose, rotmat_to_quat_xyzw  # noqa: E402

from experiments.rgb_region_prior import export_weak  # noqa: E402
from experiments.rgb_region_prior.dataset import RegionDataset, RegionSample  # noqa: E402
from experiments.rgb_region_prior.train import (  # noqa: E402
    mass_normalized_weak_weights,
    train_head,
)

_WIDTH, _HEIGHT = 64, 64


def _camera() -> CameraModel:
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    return CameraModel(K=K, width=_WIDTH, height=_HEIGHT)


def _identity_extrinsics() -> np.ndarray:
    return np.eye(4)


def _anchor(
    anchor_id: str,
    *,
    source_capture: str,
    scenario_group: str,
    position: tuple[float, float, float],
    dims: tuple[float, float, float] = (0.2, 0.2, 0.2),
    score: float = 0.6,
    agreement_count: int = 2,
) -> dict:
    return {
        "id": anchor_id,
        "scenario_group": scenario_group,
        "source_capture": source_capture,
        "hypothesis_id": anchor_id.rsplit("/", 1)[-1],
        "pose_world": {
            "translation_m": list(position),
            "quaternion_xyzw": rotmat_to_quat_xyzw(np.eye(3)).tolist(),
            "dims_m": list(dims),
        },
        "pcl_evidence": {"score": score, "min_score": 0.4},
        "multiframe_agreement": {
            "position_tolerance_m": 0.25,
            "rotation_tolerance_deg": 30.0,
            "agreement_count": agreement_count,
            "agreeing_captures": [],
        },
        "projected_region": {"crop_bbox_xyxy": [0, 0, 1, 1], "projection": {}},
        "source": {
            "image_stamp_ns": 0,
            "cloud_stamp_ns": 0,
            "sync_delta_s": 0.0,
            "rgb_checksum": "x",
            "tf_checksum": "x",
            "calibration_checksum": "x",
        },
        "provenance": "weak_pcl_multiframe_anchor",
    }


# ---------------------------------------------------------------------------
# anchor_pose / anchor_union_mask
# ---------------------------------------------------------------------------


def test_anchor_pose_reconstructs_position_rotation_dims() -> None:
    anchor = _anchor(
        "g/c/h0",
        source_capture="c",
        scenario_group="g",
        position=(1.0, 2.0, 3.0),
        dims=(0.9, 0.6, 0.6),
    )
    pose = export_weak.anchor_pose(anchor)
    assert isinstance(pose, BlockPose)
    assert np.allclose(pose.position, [1.0, 2.0, 3.0])
    assert np.allclose(pose.rotation, np.eye(3), atol=1e-6)
    assert pose.dims == pytest.approx((0.9, 0.6, 0.6))


def test_anchor_union_mask_combines_two_in_front_anchors() -> None:
    camera = _camera()
    anchors = [
        _anchor("g/c/h0", source_capture="c", scenario_group="g", position=(-0.6, 0.0, 2.0)),
        _anchor("g/c/h1", source_capture="c", scenario_group="g", position=(0.6, 0.0, 2.0)),
    ]
    mask, contributing, dropped = export_weak.anchor_union_mask(
        anchors, _identity_extrinsics(), camera
    )
    assert set(contributing) == {"g/c/h0", "g/c/h1"}
    assert dropped == []
    assert np.any(mask)

    single_mask, _, _ = export_weak.anchor_union_mask(anchors[:1], _identity_extrinsics(), camera)
    assert np.any(single_mask & mask)
    assert not np.array_equal(single_mask, mask)  # union is strictly bigger


def test_anchor_union_mask_drops_behind_camera_anchor_but_keeps_others() -> None:
    camera = _camera()
    anchors = [
        _anchor("g/c/visible", source_capture="c", scenario_group="g", position=(0.0, 0.0, 2.0)),
        _anchor("g/c/behind", source_capture="c", scenario_group="g", position=(0.0, 0.0, -2.0)),
    ]
    mask, contributing, dropped = export_weak.anchor_union_mask(
        anchors, _identity_extrinsics(), camera
    )
    assert contributing == ["g/c/visible"]
    assert dropped == ["g/c/behind"]
    assert np.any(mask)


def test_anchor_union_mask_empty_for_no_anchors() -> None:
    camera = _camera()
    mask, contributing, dropped = export_weak.anchor_union_mask([], _identity_extrinsics(), camera)
    assert not np.any(mask)
    assert contributing == []
    assert dropped == []


# ---------------------------------------------------------------------------
# export_capture end-to-end on a synthetic capture directory
# ---------------------------------------------------------------------------


def _write_calibration(path: Path) -> None:
    K = [100.0, 0.0, 32.0, 0.0, 100.0, 32.0, 0.0, 0.0, 1.0]
    calib = {"image_width": _WIDTH, "image_height": _HEIGHT, "K": K, "D": []}
    path.write_text(yaml.safe_dump(calib))


def _write_tf(path: Path) -> None:
    identity = np.eye(4).tolist()
    path.write_text(
        yaml.safe_dump(
            {
                "reference_frame": "world",
                "image_frame": "camera",
                "cloud_frame": "cloud",
                "transforms": [
                    {"name": "T_world_camera", "available": True, "matrix": identity},
                ],
            }
        )
    )


def _write_metadata(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "rgb": {"stamp": {"sec": 100, "nanosec": 0}},
                "cloud": {"stamp": {"sec": 100, "nanosec": 20_000_000}},
            }
        )
    )


def _build_capture(root: Path, name: str) -> Path:
    capture = root / name
    capture.mkdir(parents=True)
    Image.fromarray(np.full((_HEIGHT, _WIDTH, 3), 128, dtype=np.uint8)).save(capture / "rgb.png")
    _write_tf(capture / "tf.yaml")
    _write_metadata(capture / "metadata.yaml")
    return capture


def test_export_capture_writes_record_and_union_mask(tmp_path: Path) -> None:
    capture = _build_capture(tmp_path / "seibersdorf", "scenario_a__001")
    calibration_path = tmp_path / "calib.yaml"
    _write_calibration(calibration_path)
    export_root = tmp_path / "out"
    export_root.mkdir()
    anchors = [
        _anchor(
            "scenario_a/scenario_a__001/h0",
            source_capture="scenario_a__001",
            scenario_group="scenario_a",
            position=(0.0, 0.0, 2.0),
        )
    ]

    records: list[dict] = []
    status = export_weak.export_capture(
        capture,
        anchors,
        export_root=export_root,
        calibration_path=calibration_path,
        calibration_checksum="deadbeef",
        anchors_checksum="feedface",
        sync_tolerance_s=0.10,
        records=records,
    )
    assert status["status"] == "exported"
    assert status["split_group"] == "scenario_a"
    assert len(records) == 1
    record = records[0]
    assert record["sample_id"] == "scenario_a__001"
    assert record["split_group"] == "scenario_a"
    assert record["block_count"] == 1
    assert record["weak_supervision"] is True
    assert record["weak_evidence"]["anchor_count"] == 1

    mask_path = export_root / record["mask"]
    assert mask_path.is_file()
    mask = np.asarray(Image.open(mask_path)) > 0
    assert np.any(mask)


def test_export_capture_all_anchors_dropped_projection_is_not_exported(tmp_path: Path) -> None:
    capture = _build_capture(tmp_path / "seibersdorf", "scenario_a__002")
    calibration_path = tmp_path / "calib.yaml"
    _write_calibration(calibration_path)
    export_root = tmp_path / "out"
    export_root.mkdir()
    anchors = [
        _anchor(
            "scenario_a/scenario_a__002/h0",
            source_capture="scenario_a__002",
            scenario_group="scenario_a",
            position=(0.0, 0.0, -2.0),  # behind camera -> dropped
        )
    ]

    records: list[dict] = []
    status = export_weak.export_capture(
        capture,
        anchors,
        export_root=export_root,
        calibration_path=calibration_path,
        calibration_checksum="deadbeef",
        anchors_checksum="feedface",
        sync_tolerance_s=0.10,
        records=records,
    )
    assert status["status"] == "skipped_all_anchors_dropped_projection"
    assert records == []


# ---------------------------------------------------------------------------
# zero-anchor-capture skip behavior + accounting (main() end-to-end)
# ---------------------------------------------------------------------------


def test_main_skips_zero_anchor_captures_with_correct_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots_dir = tmp_path / "seibersdorf"
    snapshots_dir.mkdir()
    _build_capture(snapshots_dir, "scenario_a__with_anchor")
    _build_capture(snapshots_dir, "scenario_a__no_anchor")  # zero anchors -> must be skipped

    calibration_path = tmp_path / "calib.yaml"
    _write_calibration(calibration_path)

    anchors_path = tmp_path / "anchors.jsonl"
    anchor = _anchor(
        "scenario_a/scenario_a__with_anchor/h0",
        source_capture="scenario_a__with_anchor",
        scenario_group="scenario_a",
        position=(0.0, 0.0, 2.0),
    )
    anchors_path.write_text(json.dumps(anchor) + "\n")

    export_root = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_weak.py",
            "--anchors",
            str(anchors_path),
            "--snapshots-dir",
            str(snapshots_dir),
            "--calibration",
            str(calibration_path),
            "--export-root",
            str(export_root),
        ],
    )
    export_weak.main()

    manifest = json.loads((export_root / "manifest.json").read_text())
    health = manifest["health"]
    assert health["captures_considered"] == 2
    assert health["captures_with_anchors"] == 1
    assert health["captures_zero_anchors_skipped"] == 1
    assert health["zero_anchor_captures"] == ["scenario_a__no_anchor"]
    assert manifest["total_records"] == 1

    samples = (export_root / "samples.jsonl").read_text().splitlines()
    assert len(samples) == 1
    record = json.loads(samples[0])
    assert record["sample_id"] == "scenario_a__with_anchor"


# ---------------------------------------------------------------------------
# train.py's weak-sample weight blending: reduced loss contribution
# ---------------------------------------------------------------------------

_HIDDEN_SIZE = 6
_GRID = 4


class _FakeDenseHead(torch.nn.Module):
    """Conv2d-based, matching the real dense head's (N, C, H, W) layout."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class _FakeRegionHead:
    """Local stand-in for the assumed ``DinoRegionProposerHead`` training API."""

    def __init__(self) -> None:
        self.dense_head = _FakeDenseHead(_HIDDEN_SIZE)


def _region_sample(
    tmp_path: Path, sample_id: str, split_group: str, foreground: bool
) -> RegionSample:
    image_path = tmp_path / f"{sample_id}_image.png"
    mask_path = tmp_path / f"{sample_id}_mask.png"
    Image.fromarray(np.full((16, 16, 3), 30, dtype=np.uint8)).save(image_path)
    mask = np.zeros((16, 16), dtype=np.uint8)
    if foreground:
        mask[4:12, 4:12] = 255
    Image.fromarray(mask).save(mask_path)
    return RegionSample(
        sample_id=sample_id,
        split_group=split_group,
        image_path=image_path,
        mask_path=mask_path,
        block_count=1 if foreground else 0,
        raw={"source": {"snapshot": sample_id}, "weak_supervision": not foreground},
    )


def test_mass_normalized_weak_weights_caps_total_weak_mass(tmp_path: Path) -> None:
    """The regression this fixes: a flat per-sample weight lets weak data
    dominate purely by outnumbering real data. 210 weak samples at a flat 0.3
    weight actually outweighed 19 real samples 3.3x in total mass on the
    first real attempt at this. Mass normalization must not let that happen
    regardless of how lopsided the counts are.
    """
    real = [_region_sample(tmp_path, f"real_{i}", "g", True) for i in range(19)]
    weak = [_region_sample(tmp_path, f"weak_{i}", "g", True) for i in range(210)]
    weights = mass_normalized_weak_weights(real, weak, mass_fraction=0.3)
    total_weak_mass = sum(weights.values())
    total_real_mass = len(real) * 1.0
    assert total_weak_mass == pytest.approx(0.3 * total_real_mass)
    assert total_weak_mass < total_real_mass


def test_mass_normalized_weak_weights_scales_with_real_count_per_call(tmp_path: Path) -> None:
    """The same weak pool must get a *different* per-sample weight depending
    on how many real samples are in THIS training call (held-out-fold vs.
    cross-validation fold vs. final union all have different real counts).
    """
    weak = [_region_sample(tmp_path, f"weak_{i}", "g", True) for i in range(100)]
    few_real = [_region_sample(tmp_path, f"few_{i}", "g", True) for i in range(5)]
    many_real = [_region_sample(tmp_path, f"many_{i}", "g", True) for i in range(50)]
    few_weights = mass_normalized_weak_weights(few_real, weak, mass_fraction=0.3)
    many_weights = mass_normalized_weak_weights(many_real, weak, mass_fraction=0.3)
    few_weight = next(iter(few_weights.values()))
    many_weight = next(iter(many_weights.values()))
    assert few_weight == pytest.approx(0.3 * 5 / 100)
    assert many_weight == pytest.approx(0.3 * 50 / 100)
    assert many_weight > few_weight


def test_mass_normalized_weak_weights_empty_weak_or_real_returns_empty(tmp_path: Path) -> None:
    real = [_region_sample(tmp_path, "real_0", "g", True)]
    assert mass_normalized_weak_weights(real, [], mass_fraction=0.3) == {}
    assert mass_normalized_weak_weights([], real, mass_fraction=0.3) == {}


def test_train_head_default_sample_weights_none_matches_unweighted_mean(tmp_path: Path) -> None:
    """Passing sample_weights=None (the default) must reproduce the pre-weak behavior exactly."""
    # Same initial dense-head weights for both, so any difference in the
    # reports can only come from sample_weights, not random init.
    torch.manual_seed(123)
    head_a = _FakeRegionHead()
    torch.manual_seed(123)
    head_b = _FakeRegionHead()
    samples = [
        _region_sample(tmp_path, "s1", "g1", True),
        _region_sample(tmp_path, "s2", "g1", False),
    ]
    rng = np.random.default_rng(0)
    features = {
        s.sample_id: rng.normal(size=(_GRID, _GRID, _HIDDEN_SIZE)).astype(np.float32)
        for s in samples
    }
    report_default = train_head(head_a, samples, features, epochs=5, lr=0.05, device="cpu", seed=0)
    report_explicit_uniform = train_head(
        head_b,
        samples,
        features,
        epochs=5,
        lr=0.05,
        device="cpu",
        seed=0,
        sample_weights={s.sample_id: 1.0 for s in samples},
    )
    assert report_default["epoch_losses"] == pytest.approx(report_explicit_uniform["epoch_losses"])


def test_train_head_weak_sample_gets_reduced_weight_in_loss(tmp_path: Path) -> None:
    """A weak sample's gradient contribution should shrink as its weight shrinks.

    Freeze the head at its initial (random) weights and compare the raw batch
    loss for one real sample + one weak sample at weak weight 1.0 vs weak
    weight 0.1: since the weak sample's per-sample loss is fixed for a frozen
    head, only its contribution to the weighted mean should change, and it
    must move the batch loss *toward* the real sample's own loss as the weak
    weight shrinks.
    """
    real_sample = _region_sample(tmp_path, "real_s", "g1", True)
    weak_sample = _region_sample(tmp_path, "weak_s", "weak_group", False)
    samples = [real_sample, weak_sample]
    rng = np.random.default_rng(1)
    features = {
        s.sample_id: rng.normal(size=(_GRID, _GRID, _HIDDEN_SIZE)).astype(np.float32)
        for s in samples
    }

    from experiments.rgb_region_prior.model import resample_mask_to_patch_grid, weighted_bce_loss
    from experiments.rgb_region_prior.train import _load_mask

    def _raw_losses(head: _FakeRegionHead) -> dict[str, float]:
        losses = {}
        with torch.no_grad():
            for sample in samples:
                grid_h, grid_w = features[sample.sample_id].shape[:2]
                target = resample_mask_to_patch_grid(_load_mask(sample.mask_path), grid_h, grid_w)
                feature_tensor = torch.from_numpy(features[sample.sample_id]).float()
                feature_tensor = feature_tensor.permute(2, 0, 1).unsqueeze(0)
                logits = head.dense_head(feature_tensor).reshape(target.shape)
                losses[sample.sample_id] = float(weighted_bce_loss(logits, target).item())
        return losses

    def _first_batch_loss(weak_weight: float) -> float:
        head = _FakeRegionHead()
        head.dense_head.eval()
        raw = _raw_losses(head)
        # epochs=0 would never run the loop, so run exactly 1 epoch and read the
        # single recorded batch loss -- this is the loss BEFORE any optimizer
        # step touched the weights, since it's recorded right after backward().
        report = train_head(
            head,
            samples,
            features,
            epochs=1,
            lr=0.0,  # lr=0 -> weights never move, batch loss is a pure function of raw losses
            device="cpu",
            seed=0,
            sample_weights={weak_sample.sample_id: weak_weight},
        )
        observed = report["epoch_losses"][0]
        expected = (raw[real_sample.sample_id] * 1.0 + raw[weak_sample.sample_id] * weak_weight) / (
            1.0 + weak_weight
        )
        assert observed == pytest.approx(expected, rel=1e-4)
        return observed

    loss_full_weight = _first_batch_loss(1.0)
    loss_reduced_weight = _first_batch_loss(0.3)
    loss_near_zero_weight = _first_batch_loss(0.01)

    # As the weak sample's weight shrinks toward 0, the weighted-mean batch
    # loss must move monotonically toward the real sample's own raw loss
    # (i.e. away from the weak sample's raw loss).
    assert loss_near_zero_weight != pytest.approx(loss_full_weight)
    distances_from_zero_weight_loss = [
        abs(loss_full_weight - loss_near_zero_weight),
        abs(loss_reduced_weight - loss_near_zero_weight),
    ]
    assert distances_from_zero_weight_loss[1] < distances_from_zero_weight_loss[0]


def test_train_head_zero_weak_weight_recovers_real_only_loss(tmp_path: Path) -> None:
    """At weak weight 0, the batch loss must equal the real sample's own loss alone."""
    real_sample = _region_sample(tmp_path, "real_s", "g1", True)
    weak_sample = _region_sample(tmp_path, "weak_s", "weak_group", False)
    samples = [real_sample, weak_sample]
    rng = np.random.default_rng(2)
    features = {
        s.sample_id: rng.normal(size=(_GRID, _GRID, _HIDDEN_SIZE)).astype(np.float32)
        for s in samples
    }

    head = _FakeRegionHead()
    head.dense_head.eval()

    from experiments.rgb_region_prior.model import resample_mask_to_patch_grid, weighted_bce_loss
    from experiments.rgb_region_prior.train import _load_mask

    with torch.no_grad():
        grid_h, grid_w = features[real_sample.sample_id].shape[:2]
        target = resample_mask_to_patch_grid(_load_mask(real_sample.mask_path), grid_h, grid_w)
        feature_tensor = torch.from_numpy(features[real_sample.sample_id]).float()
        feature_tensor = feature_tensor.permute(2, 0, 1).unsqueeze(0)
        logits = head.dense_head(feature_tensor).reshape(target.shape)
        real_only_loss = float(weighted_bce_loss(logits, target).item())

    report = train_head(
        head,
        samples,
        features,
        epochs=1,
        lr=0.0,
        device="cpu",
        seed=0,
        sample_weights={weak_sample.sample_id: 0.0},
    )
    assert report["epoch_losses"][0] == pytest.approx(real_only_loss, rel=1e-4)


# ---------------------------------------------------------------------------
# RegionDataset.load reads the weak record shape (weak_supervision marker present)
# ---------------------------------------------------------------------------


def test_region_dataset_load_reads_weak_export_shape(tmp_path: Path) -> None:
    export_root = tmp_path / "weak_out"
    export_root.mkdir()
    record = {
        "sample_id": "scenario_a__001",
        "split_group": "scenario_a",
        "image": "images/scenario_a__001.png",
        "mask": "masks/scenario_a__001.png",
        "block_count": 1,
        "weak_supervision": True,
        "weak_evidence": {"anchor_count": 1},
    }
    (export_root / "samples.jsonl").write_text(json.dumps(record) + "\n")
    dataset = RegionDataset.load(export_root)
    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert sample.sample_id == "scenario_a__001"
    assert sample.raw["weak_supervision"] is True
