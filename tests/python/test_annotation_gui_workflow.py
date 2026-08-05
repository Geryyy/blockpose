"""Regression checks for the annotator's operator-facing workflow."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2]))
import tools.annotate_snapshot as annotator  # noqa: E402


def test_workflow_section_titles_name_the_five_operator_stages() -> None:
    """The panel is one top-to-bottom workflow; the titles are what the operator reads.

    Checked against the module constant rather than the rendered panel because
    building a real window needs a display. Everything else about the panel is
    covered behaviourally in `test_annotation_gui_state.py`.
    """
    assert annotator.WORKFLOW_SECTION_TITLES == (
        "1. Mask the gripper",
        "2. Auto-fit candidates",
        "3. Add and adjust blocks",
        "4. Visibility and details",
        "5. Save snapshot",
    )


def test_candidate_sources_are_chosen_automatically_not_by_the_operator() -> None:
    """Rerankers score the SAME hypotheses, so only the most refined one is used.

    Measured over this repo's 23 snapshots the RGB edge reranker strictly
    dominates raw PCL (same 75% recall, 88% precision against 84%), so there is
    nothing for an operator to choose. `dino_proposer` runs its own detection
    and is therefore additive rather than a substitute.
    """
    window = SimpleNamespace(
        candidate_sources_result=SimpleNamespace(
            pcl=("p1", "p2"), rgb_edge_prior=("r1",), dino_proposer=(), dino_verifier_reranked=()
        )
    )
    active = annotator.AnnotatorWindow._active_candidate_sources(window)
    assert active == [("rgb_edge_prior", ("r1",))], "a reranker must replace raw PCL, not add to it"

    window.candidate_sources_result = SimpleNamespace(
        pcl=("p1",), rgb_edge_prior=(), dino_proposer=("d1",), dino_verifier_reranked=()
    )
    active = annotator.AnnotatorWindow._active_candidate_sources(window)
    assert active == [("pcl", ("p1",)), ("dino_proposer", ("d1",))], (
        "the DINO proposer detects independently, so it must be added to the base source"
    )

    window.candidate_sources_result = None
    assert annotator.AnnotatorWindow._active_candidate_sources(window) == []


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


def test_segmentation_is_enabled_by_default_and_opt_out() -> None:
    """SAM recovers the blocks auto-fit never proposes, so it must not be opt-in.

    Measured over this repo's 23 snapshots, auto-fit reaches 75% recall; the
    remaining quarter is recovered by clicking the block in RGB. Making that
    path require a flag hid the tool's most useful control.
    """
    import tools.annotate_snapshot as tool

    parser = tool.build_arg_parser()
    assert parser.parse_args([]).sam_model == tool.DEFAULT_SAM_MODEL
    assert parser.parse_args(["--no-sam"]).sam_model is None
    assert parser.parse_args(["--sam-model", "facebook/other"]).sam_model == "facebook/other"


def test_build_panel_only_uses_widgets_init_has_already_created() -> None:
    """Catch `_build_panel` reaching for an attribute `__init__` never assigned.

    The GUI tests all drive real methods against SimpleNamespace fakes, which
    supply every attribute by construction, so none of them exercise the real
    `__init__` -> `_build_panel` ordering. Constructing a genuine
    `AnnotatorWindow` needs a display and cannot run here, so this checks the
    same invariant statically: every `self.<attr>` that `_build_panel` reads
    must be assigned in `__init__` above the `self._build_panel()` call.

    A regression here is not subtle -- it is an AttributeError on launch -- but
    it is invisible to every other test in this suite.
    """
    import ast

    source = Path("tools/annotate_snapshot.py").read_text(encoding="utf-8")
    klass = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "AnnotatorWindow"
    )
    methods = {n.name: n for n in klass.body if isinstance(n, ast.FunctionDef)}
    init, build = methods["__init__"], methods["_build_panel"]

    build_call_line = next(
        node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_build_panel"
    )
    assigned = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno < build_call_line
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    # Attributes _build_panel assigns itself are fine; so is anything callable
    # on the class (methods, helpers) rather than an instance widget.
    assigned |= {
        target.attr
        for node in ast.walk(build)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    } | set(methods)

    read = {
        node.attr
        for node in ast.walk(build)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    missing = sorted(read - assigned)
    assert not missing, f"_build_panel uses attributes __init__ never creates: {missing}"
