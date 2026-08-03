"""Regression checks for the annotator's operator-facing workflow."""

import ast
from pathlib import Path


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
    assert source.index("Delete selected block…") < source.index(
        "Position XYZ (m), Euler XYZ"
    )
    assert "Adopt selected suggestion as editable block" in source
    assert "self.controller.adopt_candidate(" in source
    assert "*self.pose_step_buttons" in source
    assert 'self._section("Snapshot", open_by_default=False)' in source
    assert 'advanced_filters.set_is_open(False)' in source
    assert 'parent=self.save_footer' in source
    assert "rect.get_bottom() - 210" in source
