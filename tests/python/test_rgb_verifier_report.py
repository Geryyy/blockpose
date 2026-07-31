"""Geometry-baseline comparator, calibration summary, and the has_valid_pair
guard for report.py's held-out reporting -- no GPU, no network, no checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.rgb_candidate_prior.dataset import Sample
from experiments.rgb_candidate_prior.model import first_correct_rank, has_valid_pair
from experiments.rgb_candidate_prior.report import calibration_summary, geometry_baseline_result

# ---------------------------------------------------------------------------
# geometry_baseline_result
# ---------------------------------------------------------------------------


def _sample(label: str, provenance: str, sample_id: str = "snap/block_0/x") -> Sample:
    return Sample(
        sample_id=sample_id,
        split_group="snap",
        label=label,
        provenance=provenance,
        rgb_path=Path("rgb.png"),
        conditioning_path=Path("cond.png"),
        raw={},
    )


def test_geometry_baseline_correct_when_first_raw_hypothesis_is_positive() -> None:
    samples = [
        _sample("positive", "raw_hypothesis"),
        _sample("negative", "raw_hypothesis"),
    ]
    assert geometry_baseline_result(samples) == "correct"


def test_geometry_baseline_incorrect_when_first_raw_hypothesis_is_negative() -> None:
    samples = [
        _sample("negative", "raw_hypothesis"),
        _sample("positive", "raw_hypothesis"),
    ]
    assert geometry_baseline_result(samples) == "incorrect"


def test_geometry_baseline_reports_no_raw_hypothesis_candidate() -> None:
    samples = [_sample("positive", "positive_perturbation"), _sample("negative", "yaw_sibling")]
    assert geometry_baseline_result(samples) == "no_raw_hypothesis_candidate"


def test_geometry_baseline_ignores_ignore_labelled_raw_hypothesis() -> None:
    samples = [_sample("ignore", "raw_hypothesis")]
    assert geometry_baseline_result(samples) == "no_raw_hypothesis_candidate"


# ---------------------------------------------------------------------------
# calibration_summary
# ---------------------------------------------------------------------------


def test_calibration_summary_empty_input() -> None:
    summary = calibration_summary([], [])
    assert summary == {"bins": [], "expected_calibration_error": None, "brier_score": None}


def test_calibration_summary_perfect_calibration_has_zero_ece() -> None:
    # Perfect calibration means predicted probability == empirical frequency
    # *within each occupied bin*, not perfect separation: two samples at 0.5
    # (one positive, one negative) land in one bin whose mean_predicted (0.5)
    # exactly matches its empirical_positive_rate (0.5).
    calibrated = [0.5, 0.5]
    labels = ["positive", "negative"]
    summary = calibration_summary(calibrated, labels, n_bins=10)
    assert summary["expected_calibration_error"] == pytest.approx(0.0)
    assert summary["brier_score"] == pytest.approx(0.25)


def test_calibration_summary_flags_overconfident_wrong_predictions() -> None:
    calibrated = [0.95, 0.95]
    labels = ["negative", "negative"]
    summary = calibration_summary(calibrated, labels, n_bins=10)
    assert summary["expected_calibration_error"] > 0.9
    assert summary["brier_score"] > 0.9


# ---------------------------------------------------------------------------
# has_valid_pair guard (regression: an all-positive set must not report a
# vacuous top-1 "correct" -- see model.py's first_correct_rank, which returns
# index 0 for a lone positive with no negative to actually rank against)
# ---------------------------------------------------------------------------


def test_all_positive_set_is_excluded_from_ranking_like_report_py_does() -> None:
    labels = ["positive", "positive"]
    scores = [0.5, 0.5]
    assert first_correct_rank(scores, labels) == 0  # vacuously "correct" if used unguarded
    assert not has_valid_pair(labels)  # so report.py's has_valid_pair guard must skip this set


def test_all_negative_set_is_excluded_from_ranking() -> None:
    labels = ["negative", "negative"]
    assert not has_valid_pair(labels)


def test_mixed_set_passes_the_guard() -> None:
    labels = ["positive", "negative"]
    assert has_valid_pair(labels)
