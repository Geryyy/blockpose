"""Structural smoke test for the checked-in detector showcase notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "blockpose_detector_showcase.ipynb"


def test_showcase_is_executed_and_documents_honest_baseline() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    # The showcase must keep naming its failures, not just its successes. These
    # assertions track intent; update them when a gap genuinely closes, not to
    # make the test pass after a rewrite.
    assert "evaluation-only label" in text
    assert "diagnostics_stratified" in text
    assert "buried lower stack block" in text
    assert "seamless touching row" in text
    assert "blocks in contact" in text  # the assembly / wall gap
    assert "standoff" in text
    assert "calibrated_confidence" in text
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
