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


@dataclass
class ClutterParams:
    """Environment clutter placed on the sides, keeping the working area free.

    Blocks live in the central band (|y - scene_center_y| < side_offset); trees,
    a container, and wood logs sit beyond it on the +/- Y sides. Clutter is added
    to the raycast scene as returns/occluders only — it produces NO ground truth,
    so it exercises detector precision (false positives) rather than recall.
    """

    trees: bool = True
    n_trees_per_side: int = 2
    container: bool = True
    logs: bool = True
    n_logs: int = 4
    side_offset: float = 3.0  # |y| offset where side clutter begins (m)
    x_spread: float = 3.5  # clutter spread along the viewing depth (x, m)


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


def _cylinder_mesh(radius: float, length: float, center: np.ndarray, R: np.ndarray):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=16)
    mesh.rotate(R, center=(0.0, 0.0, 0.0))  # default axis is +Z
    mesh.translate(center)
    return mesh


def _tree_mesh(
    x: float,
    y: float,
    ground_z: float = 0.0,
    trunk_h: float = 2.2,
    trunk_r: float = 0.13,
    canopy_r: float = 0.95,
):
    """A crude tree: vertical trunk cylinder + spherical canopy on top."""
    trunk = _cylinder_mesh(trunk_r, trunk_h, np.array([x, y, ground_z + trunk_h / 2.0]), np.eye(3))
    canopy = o3d.geometry.TriangleMesh.create_sphere(radius=canopy_r, resolution=12)
    canopy.translate([x, y, ground_z + trunk_h + canopy_r * 0.5])
    trunk += canopy
    return trunk


def _container_mesh(
    center_xy: tuple[float, float],
    yaw_deg: float = 0.0,
    ground_z: float = 0.0,
    dims: tuple[float, float, float] = (6.06, 2.44, 2.59),
):
    """A shipping-container-sized box resting on the ground."""
    box = o3d.geometry.TriangleMesh.create_box(dims[0], dims[1], dims[2])
    box.translate([-dims[0] / 2.0, -dims[1] / 2.0, 0.0])
    box.rotate(_rotation(yaw_deg, 0.0, 0.0), center=(0.0, 0.0, 0.0))
    box.translate([center_xy[0], center_xy[1], ground_z])
    return box


def _log_mesh(
    x: float,
    y: float,
    yaw_deg: float = 0.0,
    length: float = 3.0,
    radius: float = 0.2,
    ground_z: float = 0.0,
):
    """A wood log: cylinder laid horizontal (default +Z axis rotated to the ground plane)."""
    R = _rotation(yaw_deg, 90.0, 0.0)  # Ry(90) lays the axis horizontal, then yaw about Z
    return _cylinder_mesh(radius, length, np.array([x, y, ground_z + radius]), R)


def _build_clutter(
    clutter: ClutterParams, center_xy: tuple[float, float], rng: np.random.Generator
):
    """Legacy meshes for side clutter around (cx, cy). Working area (central band) stays free."""
    cx, cy = center_xy
    meshes = []
    if clutter.trees:
        for side in (-1.0, 1.0):
            for _ in range(clutter.n_trees_per_side):
                tx = cx + float(rng.uniform(-1.0, clutter.x_spread))
                ty = cy + side * (clutter.side_offset + float(rng.uniform(-0.3, 0.6)))
                meshes.append(_tree_mesh(tx, ty))
    if clutter.container:
        meshes.append(
            _container_mesh(
                (cx + clutter.x_spread * 0.5, cy + clutter.side_offset + 1.7), yaw_deg=8.0
            )
        )
    if clutter.logs:
        for k in range(clutter.n_logs):
            lx = cx + float(rng.uniform(-0.5, clutter.x_spread))
            ly = cy - (clutter.side_offset + 0.3 + 0.45 * k)
            meshes.append(_log_mesh(lx, ly, yaw_deg=float(rng.uniform(-12.0, 12.0))))
    return meshes


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
    clutter: ClutterParams | None = None,
) -> tuple[np.ndarray, list[BlockPose]]:
    """Raycast a scene of blocks -> (pts_world (N,3) float64, list[BlockPose]).

    GT poses use the block geometric center and R = Rz(yaw)Ry(pitch)Rx(roll).
    `clutter`, if given, adds side environment (trees/container/logs) as
    unlabeled returns — it changes the cloud but never the ground truth.
    """
    if params is None:
        params = SynthParams()
    if rng is None:
        rng = np.random.default_rng()

    cx = float(np.mean([s.x for s in specs])) if specs else 0.0
    cy = float(np.mean([s.y for s in specs])) if specs else 0.0

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
        ground = o3d.geometry.TriangleMesh.create_box(params.ground_size, params.ground_size, 0.02)
        ground.translate([cx - half, cy - half, -0.02])
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ground))

    if clutter is not None:
        for mesh in _build_clutter(clutter, (cx, cy), rng):
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

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
    clutter: ClutterParams | None = None,
) -> tuple[np.ndarray, list[BlockPose]]:
    """Generate a scene of `n_blocks` non-overlapping upright-ish blocks on a jittered grid.

    Pass `clutter` to add side environment (trees/container/logs) around the blocks.
    """
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
    return generate_scene(specs, params, rng, clutter=clutter)
