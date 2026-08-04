"""Regression checks for the annotator's operator-facing workflow."""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


def test_annotation_panel_follows_the_five_stage_workflow() -> None:
    source = Path("tools/annotate_snapshot.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "WORKFLOW_SECTION_TITLES"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == (
        "1. Preprocess cloud",
        "2. Auto-fit candidates",
        "3. Add and adjust blocks",
        "4. Visibility and details",
        "5. Save snapshot",
    )
    assert "gui.ScrollableVert" in source
    assert source.index("Delete selected block...") < source.index("Position XYZ (m), Euler XYZ")
    assert "accepted suggestions become pending blocks" in source
    assert "Accepted results are added to the editable block list below." in source
    assert "Adopt selected suggestion as editable block" not in source
    assert "self.controller.adopt_candidates(" in source
    assert '"rgb_edge_prior", "+ RGB edge validator/reranker"' in source
    assert "+ RGB edge prior" not in source
    assert "*self.pose_step_buttons" in source
    # Pose corrections stay adjacent to the value they modify instead of in
    # detached +/- rows below the value grid.
    assert "for axis, (label, field) in enumerate(" in source
    assert "row.add_child(field)" in source
    assert "grid = gui.VGrid(2, 4)" not in source
    assert 'self._section("Snapshot", open_by_default=False)' in source
    assert "advanced_filters.set_is_open(False)" in source
    assert "parent=self.save_footer" in source
    assert "rect.get_bottom() - 210" in source


def test_ui_strings_avoid_glyphs_open3d_cannot_render() -> None:
    """Open3D's bundled font renders U+2014/U+2026/U+25C0/U+25B6 as "?" glyphs.

    "°" and "·" render fine and are deliberately not checked here.
    """
    source = Path("tools/annotate_snapshot.py").read_text(encoding="utf-8")
    for forbidden, name in (
        ("—", "em dash"),
        ("…", "ellipsis"),
        ("◀", "left-pointing triangle"),
        ("▶", "right-pointing triangle"),
    ):
        assert forbidden not in source, f"unrenderable {name} found in UI source"
