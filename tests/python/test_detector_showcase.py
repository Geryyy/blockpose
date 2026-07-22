"""Structural smoke tests for the checked-in detector-mode showcase."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "blockpose_detector_showcase.ipynb"


def test_showcase_is_executed_and_reports_all_modes_honestly() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    rendered_notebook = json.dumps(notebook)

    # These are review-critical claims, not marketing language.  Keep all
    # modes, full-cloud/ROI semantics, provenance, and withheld evidence
    # visible whenever the notebook is regenerated.
    for required in (
        "SCENE_DISCOVERY",
        "REFINE_BLOCK",
        "REFINE_GRASPED",
        "full cloud",
        "scoring boundary only",
        "baseline_comparison",
        "source_worktree_dirty",
        "timeout/query-failure details",
        "seamless touching row",
        "buried lower stack block",
    ):
        assert required in text
    assert "NotRun" in rendered_notebook

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
    # Discovery aggregate + one discovery overlay + two refine overlays + two
    # retained synthetic limits.  This ensures a reader can inspect evidence.
    assert len(rendered) >= 6
