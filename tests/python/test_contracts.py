import json

import pytest
from blockpose.contracts import (
    BlockPriorContract,
    BlockResultContract,
    ConfidenceContract,
    ConfidenceKind,
    CovarianceContract,
    CovarianceKind,
    DetectorMode,
    DetectorRequestContract,
    DetectorResponseContract,
    EvidenceContract,
    EvidenceKind,
    FrameContract,
    ImageProposalContract,
    Observability,
    PoseContract,
    PriorSource,
    ResultStatus,
    canonical_json,
    normalize_unavailable_evidence,
)
from blockpose.engine import QueryResult
from blockpose.frames import PointFrame


def _pose() -> PoseContract:
    return PoseContract("world", (1.0, 2.0, 0.3), (0.0, 0.0, 0.0, 1.0))


def _frame() -> FrameContract:
    return FrameContract(
        "seyond", 100, 110, 42, "cloud:100", ("relative_time", "intensity")
    )


def _prior(source: PriorSource = PriorSource.PLAN) -> BlockPriorContract:
    return BlockPriorContract("block-1", source, _pose(), (0.9, 0.6, 0.6))


def test_modes_map_explicitly_and_enforce_prior_preconditions() -> None:
    discovery = DetectorRequestContract("q1", DetectorMode.SCENE_DISCOVERY, _frame())
    refine = DetectorRequestContract(
        "q2", DetectorMode.REFINE_BLOCK, _frame(), (_prior(),), target_block_id="block-1"
    )
    grasped = DetectorRequestContract(
        "q3", DetectorMode.REFINE_GRASPED, _frame(), (_prior(PriorSource.FK_GRASP),)
    )
    assert (discovery.engine_method, refine.engine_method, grasped.engine_method) == (
        "detect_free",
        "verify_placed",
        "estimate_gripped",
    )
    with pytest.raises(ValueError, match="requires exactly 1 prior"):
        DetectorRequestContract("bad", DetectorMode.REFINE_BLOCK, _frame())
    with pytest.raises(ValueError, match="FK_GRASP"):
        DetectorRequestContract("bad", DetectorMode.REFINE_GRASPED, _frame(), (_prior(),))
    with pytest.raises(ValueError, match="target_block_id"):
        DetectorRequestContract(
            "bad",
            DetectorMode.REFINE_GRASPED,
            _frame(),
            (_prior(PriorSource.FK_GRASP),),
            target_block_id="different-block",
        )


def test_canonical_json_is_byte_stable_and_finite_only() -> None:
    proposal = ImageProposalContract(
        "p1", "image:100", "camera", 100, (10.0, 20.0, 30.0, 40.0), "block", 0.75
    )
    request = DetectorRequestContract(
        "q1", DetectorMode.SCENE_DISCOVERY, _frame(), image_proposals=(proposal,)
    )
    encoded = request.to_json()
    assert encoded == request.to_json()
    assert list(json.loads(encoded)) == sorted(json.loads(encoded))
    assert json.loads(encoded)["frame"]["unavailable_evidence"] == [
        "intensity",
        "relative_time",
    ]
    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json({"blockpose.value": float("nan")})
    with pytest.raises(ValueError, match="must be numeric"):
        PoseContract("world", ("1.0", 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def test_quaternion_sign_and_extensions_are_canonical() -> None:
    assert PoseContract("world", (0, 0, 0), (0, 0, 0, -1)).quaternion_xyzw[-1] == 1.0
    result = BlockResultContract(
        "block-1",
        _pose(),
        (0.9, 0.6, 0.6),
        ResultStatus.DETECTED,
        Observability.WELL_CONSTRAINED,
        EvidenceContract(EvidenceKind.GEOMETRY, support_points=20),
        CovarianceContract(CovarianceKind.UNAVAILABLE, None),
        ConfidenceContract(ConfidenceKind.UNCALIBRATED_SCORE, 0.8),
        extensions={"vendor.z": 1, "blockpose.a": 2},
    )
    assert list(json.loads(canonical_json(result))["extensions"]) == ["blockpose.a", "vendor.z"]
    with pytest.raises(ValueError, match="namespaced"):
        BlockResultContract(
            "block-1",
            _pose(),
            (0.9, 0.6, 0.6),
            ResultStatus.DETECTED,
            Observability.WELL_CONSTRAINED,
            EvidenceContract(EvidenceKind.GEOMETRY),
            CovarianceContract(CovarianceKind.UNAVAILABLE, None),
            ConfidenceContract(ConfidenceKind.UNAVAILABLE, None),
            extensions={"raw_score": 0.2},
        )


def test_prior_only_cannot_masquerade_as_measurement_and_missing_needs_rays() -> None:
    common = dict(
        block_id="block-1",
        pose=_pose(),
        dimensions_m=(0.9, 0.6, 0.6),
        covariance=CovarianceContract(CovarianceKind.UNAVAILABLE, None),
        confidence=ConfidenceContract(ConfidenceKind.UNAVAILABLE, None),
    )
    prior = BlockResultContract(
        **common,
        status=ResultStatus.NOT_OBSERVABLE,
        observability=Observability.UNOBSERVABLE,
        evidence=EvidenceContract(EvidenceKind.PRIOR_ONLY),
    )
    assert prior.evidence.kind is EvidenceKind.PRIOR_ONLY
    with pytest.raises(ValueError, match="prior-only"):
        BlockResultContract(
            **common,
            status=ResultStatus.CONFIRMED,
            observability=Observability.UNOBSERVABLE,
            evidence=EvidenceContract(EvidenceKind.PRIOR_ONLY),
        )
    with pytest.raises(ValueError, match="free-space"):
        BlockResultContract(
            **common,
            status=ResultStatus.MISSING,
            observability=Observability.UNOBSERVABLE,
            evidence=EvidenceContract(EvidenceKind.GEOMETRY, free_space_rays=0),
        )


def test_covariance_and_calibrated_confidence_are_typed() -> None:
    values = [0.0] * 36
    for index in range(6):
        values[index * 6 + index] = 0.01
    covariance = CovarianceContract(CovarianceKind.CALIBRATED, tuple(values), "world")
    assert covariance.to_dict()["tangent_order"] == ("x", "y", "z", "rx", "ry", "rz")
    with pytest.raises(ValueError, match="calibration_id"):
        ConfidenceContract(ConfidenceKind.CALIBRATED_VALIDITY, 0.9)
    indefinite = values.copy()
    indefinite[0] = indefinite[7] = 1.0
    indefinite[1] = indefinite[6] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        CovarianceContract(CovarianceKind.CALIBRATED, tuple(indefinite), "world")


def test_canonical_dimensions_and_response_invariants() -> None:
    with pytest.raises(ValueError, match="canonical local axes"):
        BlockPriorContract("block-1", PriorSource.PLAN, _pose(), (0.6, 0.6, 0.9))
    block = BlockResultContract(
        "block-1",
        _pose(),
        (0.9, 0.6, 0.6),
        ResultStatus.DETECTED,
        Observability.WELL_CONSTRAINED,
        EvidenceContract(EvidenceKind.GEOMETRY),
        CovarianceContract(CovarianceKind.UNAVAILABLE, None),
        ConfidenceContract(ConfidenceKind.UNAVAILABLE, None),
    )
    with pytest.raises(ValueError, match="only detected"):
        DetectorResponseContract(
            "q", DetectorMode.SCENE_DISCOVERY, (block.__class__(
                **{**block.__dict__, "status": ResultStatus.CONFIRMED}
            ),), (), 1.0
        )
    with pytest.raises(ValueError, match="unique"):
        DetectorResponseContract("q", DetectorMode.SCENE_DISCOVERY, (block, block), (), 1.0)

    missing = BlockResultContract(
        "block-1",
        _pose(),
        (0.9, 0.6, 0.6),
        ResultStatus.MISSING,
        Observability.UNOBSERVABLE,
        EvidenceContract(EvidenceKind.GEOMETRY, free_space_rays=3),
        CovarianceContract(CovarianceKind.UNAVAILABLE, None),
        ConfidenceContract(ConfidenceKind.UNAVAILABLE, None),
    )
    with pytest.raises(ValueError, match="cannot return missing"):
        DetectorResponseContract(
            "q",
            DetectorMode.REFINE_GRASPED,
            (missing,),
            (),
            1.0,
            target_block_id="block-1",
        )


def test_unavailable_evidence_maps_current_frame_and_engine_values_without_loss() -> None:
    frame = PointFrame([[0.0, 0.0, 0.0]], frame_id="world")
    query = QueryResult(
        (),
        (),
        0.0,
        (
            "per_point_sensor_origins",
            "no_return_ray_origins",
            "no_return_relative_time",
        ),
    )
    source = (*frame.unavailable_evidence, *query.unavailable_evidence)
    mapped = normalize_unavailable_evidence(source)
    assert {item.value for item in mapped} == set(source)
