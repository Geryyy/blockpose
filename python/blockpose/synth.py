"""Lightweight synthetic scene generator for block detection development.

Raycasts a set of box meshes from a sensor pose using Open3D's tensor
raycasting, producing a partial-visibility point cloud plus exact ground-truth
`BlockPose` objects. No physics engine — just geometry — so it stays fast and
faithful to the exact block shape while exercising the real detector failure
modes (self-occlusion, range noise, dropout, tilt).

The output ``(pts_world, gt_poses)`` matches ``io.load_snapshot`` so the same
detector + evaluator flow runs on synthetic and real data unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from blockpose.pose import BlockPose

# Block frame convention (see docs/detector-plan.md): the 0.9 m long axis is
# local Z, the 0.6 x 0.6 cross-section spans local X/Y.
BLOCK_DIMS = (0.6, 0.6, 0.9)


@dataclass
class BlockSpec:
    """One block placed in a synthetic scene."""

    x: float
    y: float
    z: float | None = None  # center height; defaults to resting on flat ground
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    dims: tuple[float, float, float] = BLOCK_DIMS


@dataclass
class SynthParams:
    """Scene + sensor + degradation parameters for synthetic generation."""

    sensor_pos: tuple[float, float, float] = (-5.0, 0.0, 4.0)
    # Ray fan aimed at the scene centroid (or `look_at` if set).
    look_at: tuple[float, float, float] | None = None
    az_fov_deg: float = 70.0
    el_fov_deg: float = 55.0
    ang_res_deg: float = 0.25
    range_noise_std: float = 0.01  # meters, along the ray
    dropout: float = 0.0  # fraction of hit points randomly removed
    ground: bool = True
    ground_size: float = 40.0
    dims: tuple[float, float, float] = BLOCK_DIMS


def _rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees -> (3,3) rotation."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def _box_mesh(center: np.ndarray, R: np.ndarray, dims: tuple[float, float, float]):
    mesh = o3d.geometry.TriangleMesh.create_box(dims[0], dims[1], dims[2])
    mesh.translate(-np.asarray(dims, dtype=np.float64) / 2.0)  # origin-centered
    mesh.rotate(R, center=(0.0, 0.0, 0.0))
    mesh.translate(center)
    return mesh


def _ray_directions(sensor_pos: np.ndarray, target: np.ndarray, params: SynthParams) -> np.ndarray:
    """Pinhole-style ray fan from sensor toward target -> (M, 3) unit directions."""
    forward = target - sensor_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    n_az = max(int(params.az_fov_deg / params.ang_res_deg), 1)
    n_el = max(int(params.el_fov_deg / params.ang_res_deg), 1)
    az = np.radians(np.linspace(-params.az_fov_deg / 2, params.az_fov_deg / 2, n_az))
    el = np.radians(np.linspace(-params.el_fov_deg / 2, params.el_fov_deg / 2, n_el))
    az_grid, el_grid = np.meshgrid(az, el)
    dirs = (
        forward[None, :]
        + np.tan(az_grid).ravel()[:, None] * right[None, :]
        + np.tan(el_grid).ravel()[:, None] * up[None, :]
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def generate_scene(
    specs: list[BlockSpec],
    params: SynthParams | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[BlockPose]]:
    """Raycast a scene of blocks -> (pts_world (N,3) float64, list[BlockPose]).

    GT poses use the block geometric center and R = Rz(yaw)Ry(pitch)Rx(roll).
    """
    if params is None:
        params = SynthParams()
    if rng is None:
        rng = np.random.default_rng()

    gt: list[BlockPose] = []
    scene = o3d.t.geometry.RaycastingScene()
    for spec in specs:
        z = spec.z if spec.z is not None else spec.dims[2] / 2.0
        center = np.array([spec.x, spec.y, z], dtype=np.float64)
        R = _rotation(spec.yaw_deg, spec.pitch_deg, spec.roll_deg)
        scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(_box_mesh(center, R, spec.dims))
        )
        gt.append(BlockPose(position=center, rotation=R, confidence=1.0, source="synthetic"))

    if params.ground:
        half = params.ground_size / 2.0
        cx = float(np.mean([s.x for s in specs])) if specs else 0.0
        cy = float(np.mean([s.y for s in specs])) if specs else 0.0
        ground = o3d.geometry.TriangleMesh.create_box(params.ground_size, params.ground_size, 0.02)
        ground.translate([cx - half, cy - half, -0.02])
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ground))

    sensor_pos = np.asarray(params.sensor_pos, dtype=np.float64)
    if params.look_at is not None:
        target = np.asarray(params.look_at, dtype=np.float64)
    elif specs:
        target = np.array(
            [np.mean([s.x for s in specs]), np.mean([s.y for s in specs]), params.dims[2] / 2.0]
        )
    else:
        target = sensor_pos + np.array([1.0, 0.0, -1.0])

    dirs = _ray_directions(sensor_pos, target, params)
    origins = np.broadcast_to(sensor_pos, dirs.shape)
    rays = o3d.core.Tensor(
        np.hstack([origins, dirs]).astype(np.float32), dtype=o3d.core.Dtype.Float32
    )
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    hit = np.isfinite(t_hit)

    t = t_hit[hit]
    d = dirs[hit]
    if params.range_noise_std > 0.0:
        t = t + rng.normal(0.0, params.range_noise_std, size=t.shape)
    pts = sensor_pos[None, :] + t[:, None] * d

    if params.dropout > 0.0:
        keep = rng.random(len(pts)) >= params.dropout
        pts = pts[keep]

    return pts.astype(np.float64), gt


def random_scene(
    n_blocks: int,
    params: SynthParams | None = None,
    rng: np.random.Generator | None = None,
    spacing: float = 1.2,
    max_tilt_deg: float = 3.0,
) -> tuple[np.ndarray, list[BlockPose]]:
    """Generate a scene of `n_blocks` non-overlapping upright-ish blocks on a jittered grid."""
    if rng is None:
        rng = np.random.default_rng()
    if params is None:
        params = SynthParams()

    cols = int(np.ceil(np.sqrt(n_blocks)))
    specs: list[BlockSpec] = []
    for i in range(n_blocks):
        gx, gy = i % cols, i // cols
        specs.append(
            BlockSpec(
                x=gx * spacing + float(rng.uniform(-0.15, 0.15)),
                y=gy * spacing + float(rng.uniform(-0.15, 0.15)),
                yaw_deg=float(rng.uniform(0.0, 360.0)),
                pitch_deg=float(rng.uniform(-max_tilt_deg, max_tilt_deg)),
                roll_deg=float(rng.uniform(-max_tilt_deg, max_tilt_deg)),
                dims=params.dims,
            )
        )
    return generate_scene(specs, params, rng)
