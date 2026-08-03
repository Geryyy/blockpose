from __future__ import annotations

import numpy as np
from blockpose.annotation_preprocess import (
    filter_exclusion_boxes,
    filter_workspace,
    remove_ground,
)


def test_workspace_and_explicit_exclusion_filters_preserve_expected_points() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.2], [2.0, 0.0, 0.2]])
    workspace, removed = filter_workspace(points, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    assert removed == 1
    kept, removed = filter_exclusion_boxes(
        workspace, [[0.2, 0.0, 0.2]], [np.eye(3)], [[0.1, 0.1, 0.1]]
    )
    assert removed == 1
    np.testing.assert_allclose(kept, [[0.0, 0.0, 0.0]])


def test_ground_removal_returns_native_model_diagnostics() -> None:
    ground = np.array([[x, y, 0.0] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])
    kept, diagnostics = remove_ground(np.vstack((ground, [[0.0, 0.0, 0.4]])))
    assert diagnostics["input_count"] == 10
    assert diagnostics["removed_count"] == 9
    assert diagnostics["source"] in {"ransac", "percentile"}
    np.testing.assert_allclose(kept, [[0.0, 0.0, 0.4]])
