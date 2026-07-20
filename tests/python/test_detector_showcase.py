"""Structural smoke test for the checked-in detector showcase notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "blockpose_detector_showcase.ipynb"


def test_showcase_is_executed_and_documents_honest_baseline() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "evaluation-only labels" in text
    assert "diagnostics_stratified" in text
    assert "Failure: buried lower stack block" in text
    assert "Failure: sparse far-range block" in text
    assert "Failure: seamless touching row" in text
    assert "CuboidHypothesis" in text
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
