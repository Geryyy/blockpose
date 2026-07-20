"""Structural checks for physically equal 3-D notebook axes.

Both mechanisms in use are accepted. The gallery renders static matplotlib
projections and equalizes them with `set_box_aspect` over the data span; the
showcase renders interactive plotly figures via `blockpose.viz`, which sets
`aspectmode: cube` over a cubic range (covered numerically by
`test_viz.py::test_equal_aspect_ranges_are_cubic`). What matters is that a
notebook never displays a distorted cuboid, not which library produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parents[2] / "notebooks"
SHOWCASE = "blockpose_detector_showcase.ipynb"
OPTIONAL_GALLERY = "scene_gallery.ipynb"

MATPLOTLIB_MARKERS = ("set_box_aspect(spans)", "min_span", "np.isfinite")
VIZ_MARKERS = ("blockpose.viz", "scene_figure", "sample_figure")


def _source(name: str) -> tuple[dict, str]:
    notebook = json.loads((NOTEBOOKS / name).read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def _notebooks_to_check(root: Path) -> list[str]:
    names = [SHOWCASE]
    if (root / OPTIONAL_GALLERY).is_file():
        names.append(OPTIONAL_GALLERY)
    return names


def test_relevant_notebooks_equalize_3d_axes_and_are_executed() -> None:
    for name in _notebooks_to_check(NOTEBOOKS):
        notebook, source = _source(name)
        uses_matplotlib = all(marker in source for marker in MATPLOTLIB_MARKERS)
        uses_viz = any(marker in source for marker in VIZ_MARKERS)
        assert uses_matplotlib or uses_viz, f"{name} has no equal-aspect 3-D mechanism"
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        assert not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )


def test_showcase_ships_rendered_interactive_figures() -> None:
    """Interactive inspection is the point; the code alone is not enough."""
    notebook, _ = _source(SHOWCASE)
    rendered = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if "application/vnd.plotly.v1+json" in output.get("data", {})
    ]
    assert len(rendered) >= 4


def test_clean_checkout_does_not_require_optional_gallery(tmp_path: Path) -> None:
    (tmp_path / SHOWCASE).touch()
    assert _notebooks_to_check(tmp_path) == [SHOWCASE]
