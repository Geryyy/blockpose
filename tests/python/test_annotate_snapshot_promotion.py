"""Coverage for `AnnotatorWindow._promote_accepted_candidates`'s de-duplication.

Regression tests for the auto-adoption defect: `dino_proposer` candidate ids
are namespaced (`dino_proposer:{proposal_index}:{hypothesis.id}`, see
`annotation_candidates.py::_dino_proposer_candidates`) while the active base
source's (pcl/rgb_edge_prior/dino_verifier_reranked) ids are the bare
hypothesis id, so de-duplicating only by `candidate.id` let a proposer
candidate describing the SAME physical block as an already-accepted base
candidate slip through and get promoted a second time right beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / "python"))
from blockpose.annotation_candidates import CandidateSourcesResult, SourceCandidate  # noqa: E402
from blockpose.annotation_tool import AnnotationController, AnnotationDocument  # noqa: E402
from blockpose.pose import BlockPose  # noqa: E402

import tools.annotate_snapshot as annotator  # noqa: E402

_IDENTITY = np.eye(3)
_DIMS = (0.9, 0.6, 0.6)


def _pose(position) -> BlockPose:
    return BlockPose(np.asarray(position, dtype=np.float64), _IDENTITY.copy(), dims=_DIMS)


def _candidate(candidate_id: str, position, *, accepted: bool = True) -> SourceCandidate:
    return SourceCandidate(candidate_id, _pose(position), 0.9, accepted, "final")


def _result(**sources) -> CandidateSourcesResult:
    defaults = {
        "pcl": (),
        "rgb_edge_prior": (),
        "dino_proposer": (),
        "dino_verifier_geometry_only": (),
        "dino_verifier_reranked": (),
        "warnings": [],
    }
    defaults.update(sources)
    return CandidateSourcesResult(**defaults)


def _window(document: AnnotationDocument, result: CandidateSourcesResult) -> SimpleNamespace:
    controller = AnnotationController(document)
    window = SimpleNamespace(
        controller=controller,
        candidate_sources_result=result,
        _load_document_widgets=lambda: None,
    )
    window._active_candidate_sources = MethodType(
        annotator.AnnotatorWindow._active_candidate_sources, window
    )
    window._promote_accepted_candidates = MethodType(
        annotator.AnnotatorWindow._promote_accepted_candidates, window
    )
    return window


def test_a_proposer_twin_of_an_accepted_base_candidate_is_not_double_promoted() -> None:
    """The measured 8-snapshot bug: 18/23 proposer candidates within 0.25 m of
    an already-promoted base candidate, several axis-swapped twins, all
    auto-adopted alongside the base one because id-only de-dup never collides.
    """
    document = AnnotationDocument(snapshot="scene-1")
    result = _result(
        pcl=(_candidate("hyp-1", [1.0, 2.0, 0.3]),),
        dino_proposer=(
            # Same physical block as "hyp-1", 0.1 m away -- well inside the
            # 0.25 m dedup radius, but a different, namespaced id.
            _candidate("dino_proposer:0:hyp-1", [1.08, 2.0, 0.3]),
            # A genuinely different block, 2 m away -- must still be promoted.
            _candidate("dino_proposer:0:hyp-2", [3.0, 2.0, 0.3]),
        ),
    )
    window = _window(document, result)

    promoted = window._promote_accepted_candidates()

    assert len(promoted) == 2
    positions = [target.position for target in document.targets]
    assert len(positions) == 2
    # The base candidate's position must be the one kept, not the proposer twin's.
    assert any(np.allclose(position, [1.0, 2.0, 0.3]) for position in positions)
    assert not any(np.allclose(position, [1.08, 2.0, 0.3]) for position in positions)
    assert any(np.allclose(position, [3.0, 2.0, 0.3]) for position in positions)


def test_dedup_checks_targets_already_in_the_document_not_just_the_batch() -> None:
    """`adopted_candidate_ids` only suppresses re-adoption by id, so a fresh
    recompute's proposer twin of an earlier-adopted (different-id) target
    must still be caught by the centre-distance check, not just an
    id-collision check local to the current batch.
    """
    document = AnnotationDocument(snapshot="scene-1")
    controller = AnnotationController(document)
    controller.adopt_candidates([("hyp-1", np.array([1.0, 2.0, 0.3]), _IDENTITY.copy(), _DIMS)])
    assert len(document.targets) == 1

    # Recompute: a namespaced proposer id near the already-adopted target, and
    # not itself in `adopted_candidate_ids` (that set is keyed on "hyp-1").
    result = _result(
        dino_proposer=(_candidate("dino_proposer:0:hyp-1", [1.05, 2.0, 0.3]),),
    )
    window = SimpleNamespace(
        controller=controller,
        candidate_sources_result=result,
        _load_document_widgets=lambda: None,
    )
    window._active_candidate_sources = MethodType(
        annotator.AnnotatorWindow._active_candidate_sources, window
    )
    window._promote_accepted_candidates = MethodType(
        annotator.AnnotatorWindow._promote_accepted_candidates, window
    )

    promoted = window._promote_accepted_candidates()

    assert promoted == []
    assert len(document.targets) == 1


def test_two_blocks_farther_apart_than_the_dedup_radius_are_both_kept() -> None:
    """The de-dup must not become an accidental cap on legitimate distinct blocks."""
    document = AnnotationDocument(snapshot="scene-1")
    result = _result(
        pcl=(
            _candidate("hyp-1", [0.0, 0.0, 0.3]),
            _candidate("hyp-2", [1.0, 0.0, 0.3]),
        ),
    )
    window = _window(document, result)

    promoted = window._promote_accepted_candidates()

    assert len(promoted) == 2
    assert len(document.targets) == 2


def test_target_provenance_label_separates_region_proposals_from_whole_cloud() -> None:
    """The operator-facing half of the same defect.

    De-duplication removes the twins it can prove are twins; the label is
    what lets a human catch the ones it cannot, so it has to distinguish the
    three origins the target list can contain -- and it has to keep doing so
    off `provenance` alone, which is the only origin record that survives a
    save/reload round trip.
    """
    document = AnnotationDocument(snapshot="scene-1")
    controller = AnnotationController(document)
    controller.adopt_candidates(
        [
            ("hyp-1", np.array([0.0, 0.0, 0.3]), _IDENTITY.copy(), _DIMS),
            ("dino_proposer:0:hyp-2", np.array([3.0, 0.0, 0.3]), _IDENTITY.copy(), _DIMS),
        ]
    )
    controller.add_manual(np.array([6.0, 0.0, 0.3]))

    labels = [annotator.target_provenance_label(target) for target in document.targets]
    assert labels == ["whole cloud", "region proposal", "manual"]


def test_rejected_candidates_are_never_promoted_or_counted_for_dedup() -> None:
    document = AnnotationDocument(snapshot="scene-1")
    result = _result(
        pcl=(_candidate("hyp-1", [0.0, 0.0, 0.3], accepted=False),),
    )
    window = _window(document, result)

    promoted = window._promote_accepted_candidates()

    assert promoted == []
    assert document.targets == []
