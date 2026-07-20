"""Structural checks for physically equal 3-D notebook axes."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parents[2] / "notebooks"
SHOWCASE = "blockpose_detector_showcase.ipynb"
OPTIONAL_GALLERY = "scene_gallery.ipynb"


def _source(name: str) -> tuple[dict, str]:
    notebook = json.loads((NOTEBOOKS / name).read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def _notebooks_to_check(root: Path) -> list[str]:
    names = [SHOWCASE]
    if (root / OPTIONAL_GALLERY).is_file():
        names.append(OPTIONAL_GALLERY)
    return names


def test_relevant_notebooks_use_data_span_box_aspect_and_are_executed() -> None:
    for name in _notebooks_to_check(NOTEBOOKS):
        notebook, source = _source(name)
        assert "set_box_aspect(spans)" in source
        assert "np.isfinite" in source
        assert "min_span" in source
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        assert not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )


def test_clean_checkout_does_not_require_optional_gallery(tmp_path: Path) -> None:
    (tmp_path / SHOWCASE).touch()
    assert _notebooks_to_check(tmp_path) == [SHOWCASE]
