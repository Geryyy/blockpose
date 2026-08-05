"""Structural smoke tests for the checked-in detector evidence-stage showcase.

The notebook compares the four evidence stages (PCL geometric detector, FK/
gripper prior, dino_proposer, dino_verifier) and is regenerated from
committed snapshot data and calibration only -- no trained checkpoint or GPU
required. See `notebooks/blockpose_detector_showcase.ipynb`.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "blockpose_detector_showcase.ipynb"


def test_showcase_is_executed_and_covers_every_evidence_stage() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    # Markdown hard-wraps within a sentence, so compare against whitespace-
    # collapsed text rather than requiring exact line breaks to match.
    flat_text = " ".join(text.split())

    # Review-critical claims, not marketing language: keep every evidence
    # stage, its honesty caveats, and the wiring status visible whenever the
    # notebook is regenerated.
    for required in (
        "PCL geometric detector",
        "dino_proposer",
        "dino_verifier",
        "FK / gripper",
        "cluster_max_extent_xy",
        "heavy_occlusion",
        "not yet wired into the C++ detector",
        "in-sample",
        "Retired as a candidate",
    ):
        assert required in flat_text

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_showcase_retains_rendered_review_figures() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    rendered = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    # Architecture diagram, the 4-panel metrics comparison, the wall-scene 3-D
    # plot, and the occluded-block RGB overlay.
    assert len(rendered) >= 4
