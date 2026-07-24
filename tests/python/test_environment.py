"""Python-side environment smoke tests for the future C++ core dependencies.

Verifies the uv venv provides working versions of the planned core dependencies
and that the Open3D wheel matches the C++ SDK version pinned for the upcoming
scaffolding. No C++ executable or binding is implemented yet.
"""

import numpy as np
import open3d as o3d
import pytest
import rerun as rr  # noqa: F401  (import is the test)
import yaml  # noqa: F401
from mcap.reader import make_reader  # noqa: F401

# Must match OPEN3D_VERSION in docker/Dockerfile.core (plan §2.3: identical
# tensor-API behavior between C++ core and Python tooling).
PINNED_OPEN3D = "0.19.0"


def test_open3d_version_matches_cpp_sdk() -> None:
    assert o3d.__version__ == PINNED_OPEN3D


def test_open3d_tensor_float32_pointcloud() -> None:
    pts = np.array(
        [[0, 0, 0], [0.9, 0, 0], [0, 0.6, 0], [0, 0, 0.6]], dtype=np.float32
    )
    cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(pts))
    assert cloud.point.positions.dtype == o3d.core.float32
    down = cloud.voxel_down_sample(0.05)
    assert down.point.positions.shape[0] == 4


def test_open3d_raycasting_scene() -> None:
    """Raycasting is the synthetic generator's core primitive (plan §8)."""
    box = o3d.t.geometry.TriangleMesh.create_box(0.9, 0.6, 0.6)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(box)
    rays = o3d.core.Tensor(
        [[0.45, 0.3, 5.0, 0.0, 0.0, -1.0]], dtype=o3d.core.float32
    )
    hit = scene.cast_rays(rays)
    assert np.isfinite(hit["t_hit"].numpy()).all()


def test_numpy_float32_row_major_convention() -> None:
    """Canonical point buffer: (N, 3) float32 C-contiguous (plan §2.2)."""
    pts = np.zeros((100, 3), dtype=np.float32)
    assert pts.flags["C_CONTIGUOUS"]


@pytest.mark.skip(reason="enable after the planned pybind11 scaffolding is implemented")
def test_bindings_zero_copy() -> None:
    import blockpose  # noqa: F401

    # Future binding gate: assert the C++ PointCloud maps to NumPy without copy
    # (compare data pointers via __array_interface__).
