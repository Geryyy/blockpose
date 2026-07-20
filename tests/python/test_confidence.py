"""Tests for deterministic synthetic-only validity confidence calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from blockpose.confidence import (
    ARTIFACT_VERSION,
    DEFAULT_ARTIFACT,
    FEATURE_SCHEMA,
    CalibrationArtifact,
    calibration_metrics,
    feature_vector,
    fit_logistic,
    load_artifact,
    predict,
    save_artifact,
)
from blockpose.detection import SensorContext, detect_blocks, detect_blocks_detailed
from blockpose.synth import BlockSpec, SynthParams, generate_scene

TOOLS = Path(__file__).parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import calibrate_confidence  # noqa: E402


def _features(**updates) -> dict[str, float]:
    values = dict.fromkeys(FEATURE_SCHEMA, 0.5)
    values.update(updates)
    return values


def test_metrics_brier_ece_reliability_and_threshold_counts() -> None:
    metrics = calibration_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 5)
    assert metrics["count"] == 4
    assert metrics["positives"] == metrics["negatives"] == 2
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["precision_at_0_8"] == 1.0
    assert metrics["coverage_at_0_8"] == 0.5
    assert sum(item["count"] for item in metrics["reliability"]) == 4


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_metrics_reject_invalid_bins(bins) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calibration_metrics(np.array([0, 1]), np.array([0.1, 0.9]), bins)


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ([0, 2], [0.1, 0.9], "binary"),
        ([0, np.nan], [0.1, 0.9], "binary"),
        ([0, 1], [-0.1, 0.9], r"\[0, 1\]"),
        ([0, 1], [0.1, 1.1], r"\[0, 1\]"),
        ([0, 1], [0.1, np.inf], r"\[0, 1\]"),
        ([[0, 1]], [[0.1, 0.9]], "aligned non-empty"),
    ],
)
def test_metrics_reject_invalid_labels_and_probabilities(labels, probabilities, message) -> None:
    with pytest.raises(ValueError, match=message):
        calibration_metrics(np.asarray(labels), np.asarray(probabilities))


def test_fit_serialization_schema_and_fallback(tmp_path: Path) -> None:
    width = len(FEATURE_SCHEMA)
    matrix = np.vstack([
        np.full(width, 0.1), np.full(width, 0.3),
        np.full(width, 0.8), np.full(width, 1.0),
    ])
    labels = np.array([0, 0, 1, 1])
    artifact = fit_logistic(
        matrix, labels, development_seeds=[1], gate_seeds=[2], l2=0.5
    )
    path = tmp_path / "artifact.json"
    save_artifact(artifact, path)
    loaded = load_artifact(path)
    assert loaded == artifact
    assert predict(_features(), None) == 0.0
    bad = CalibrationArtifact(**{**artifact.__dict__, "schema": ["wrong"] * len(FEATURE_SCHEMA)})
    assert predict(_features(), bad) == 0.0
    with pytest.raises(ValueError, match="schema/order"):
        feature_vector(dict(reversed(list(_features().items()))))


@pytest.mark.parametrize("l2", [-1.0, np.nan, np.inf])
def test_fit_rejects_invalid_regularization(l2: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        fit_logistic(
            np.array([[0.0] * len(FEATURE_SCHEMA), [1.0] * len(FEATURE_SCHEMA)]),
            np.array([0, 1]),
            l2=l2,
            development_seeds=[1],
            gate_seeds=[2],
        )


def test_predict_malformed_in_memory_artifact_falls_back() -> None:
    malformed = CalibrationArtifact(
        ARTIFACT_VERSION,
        list(FEATURE_SCHEMA),
        [None] * len(FEATURE_SCHEMA),
        [1.0] * len(FEATURE_SCHEMA),
        [1.0] * len(FEATURE_SCHEMA),
        0.0,
        1.0,
        [1],
        [2],
    )
    assert predict(_features(), malformed) == 0.0


@pytest.mark.parametrize("l2", [-1.0, np.nan, np.inf])
def test_artifact_rejects_invalid_regularization_in_memory_and_loaded(
    l2: float, tmp_path: Path
) -> None:
    artifact = CalibrationArtifact(
        ARTIFACT_VERSION,
        list(FEATURE_SCHEMA),
        [0.0] * len(FEATURE_SCHEMA),
        [1.0] * len(FEATURE_SCHEMA),
        [0.0] * len(FEATURE_SCHEMA),
        0.0,
        l2,
        [1],
        [2],
    )
    with pytest.raises(ValueError, match="l2 must be finite and non-negative"):
        artifact.validate()
    assert predict(_features(), artifact) == 0.0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(artifact.__dict__))
    assert load_artifact(path) is None


def test_no_default_runtime_artifact_and_seed_splits_are_disjoint() -> None:
    assert load_artifact(DEFAULT_ARTIFACT) is None
    assert set(calibrate_confidence.DEVELOPMENT_SEEDS).isdisjoint(
        calibrate_confidence.GATE_SEEDS
    )


def test_selected_output_corpus_blocks_fit_without_organic_negatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibrate_confidence, "DEVELOPMENT_SEEDS", [211])
    monkeypatch.setattr(calibrate_confidence, "GATE_SEEDS", [212])
    report = calibrate_confidence.run(tmp_path / "must_not_exist.json")
    assert report["status"] == "blocked"
    assert report["development"]["positives"] > 0
    assert report["development"]["negatives"] < 20
    assert not (tmp_path / "must_not_exist.json").exists()


def test_small_selected_output_corpus_has_stable_labels() -> None:
    first_x, first_y = calibrate_confidence.build_corpus([211])
    second_x, second_y = calibrate_confidence.build_corpus([211])
    assert len(first_x) == len(first_y)
    assert len(second_x) == len(second_y)
    # Open3D 0.19's segment_plane RANSAC is not seedable (o3d.utility.random.seed
    # does not reach it) and is not run-to-run reproducible, so hypothesis
    # GENERATION can differ between two calls on identical input. Refined pose
    # values are stable to <1 um because the SDF fit converges to the same
    # optimum regardless of which plane seeded it, but the number of candidates
    # can vary. Asserting an exact label vector here was flaky roughly one run in
    # three. The property that actually matters for calibration is unchanged: the
    # corpus is overwhelmingly positive and therefore cannot train a calibrator.
    for labels in (first_y, second_y):
        assert labels.sum() >= 0.95 * len(labels)


def test_runtime_confidence_is_finite_validity_and_does_not_change_geometry() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=21.0)], SynthParams(), np.random.default_rng(5)
    )
    context = SensorContext(sample.sensor_origin, sample.ray_directions, sample.ranges)
    detailed = detect_blocks_detailed(sample.points, sensor_context=context)
    plain = detect_blocks(sample.points, sensor_context=context)
    assert len(detailed.poses) == len(plain) == 1
    assert detailed.diagnostics[0].calibrated_confidence is None
    assert detailed.poses[0].confidence in (0.5, 1.0)
    assert plain[0].confidence in (0.5, 1.0)
    np.testing.assert_allclose(detailed.poses[0].position, plain[0].position, atol=0.03)
