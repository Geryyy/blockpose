"""Tests for the synthetic detector eval harness (tools/eval_detector.py).

This is the Phase-A regression gate: it runs the real detector over a synthetic
suite and asserts the checked-in thresholds are met. It shares config/detector_eval.yaml
with the CLI so there is a single source of truth for the thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import eval_detector  # noqa: E402

THRESHOLDS_YAML = REPO_ROOT / "config" / "detector_eval.yaml"


def test_gate_passes_on_current_detector() -> None:
    metrics = eval_detector.run_suite(seed=0)
    thresholds = yaml.safe_load(THRESHOLDS_YAML.read_text())["thresholds"]
    failures = eval_detector.check_thresholds(metrics, thresholds)
    assert not failures, f"Phase-A gate regressed: {failures}"


def test_metrics_report_diagnostics() -> None:
    metrics = eval_detector.run_suite(seed=0)
    # Isolated-block gate metrics.
    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["false_positives_empty"] == 0
    # Multi-block diagnostics are reported (ungated Phase-C tracking).
    assert "diagnostics_multiblock" in metrics
    assert "recall" in metrics["diagnostics_multiblock"]
