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

from dataclasses import asdict, dataclass, field
from typing import Iterator

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
    range_noise_slope: float = 0.0  # additional std (m) per meter of range
    dropout: float = 0.0  # fraction of hit points randomly removed
    stripe_dropout: float = 0.0  # fraction of azimuth columns removed contiguously
    outlier_fraction: float = 0.0  # random returns relative to retained hit count
    outlier_radius: float = 0.5  # maximum displacement of an outlier (m)
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
    near_field_boxes: int = 0
    near_field_offset: float = 0.9
    near_field_dims: tuple[float, float, float] = (0.72, 0.48, 0.75)


@dataclass
class SceneSample:
    """Rich synthetic result that still unpacks as ``(points, ground_truth)``.

    Labels refer to the first surface hit before noise is applied.  ``-1`` is
    ground or clutter; block instances are zero-based indices into
    ``ground_truth``.  The manifest is JSON-serializable and records generation
    inputs so failures can be reproduced.
    """

    points: np.ndarray
    ground_truth: list[BlockPose]
    instance_ids: np.ndarray
    face_ids: np.ndarray
    ranges: np.ndarray
    ray_directions: np.ndarray
    sensor_origin: np.ndarray
    manifest: dict = field(default_factory=dict)

    def __iter__(self) -> Iterator:
        yield self.points
        yield self.ground_truth

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return (self.points, self.ground_truth)[index]


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
    for k in range(clutter.near_field_boxes):
        side = -1.0 if k % 2 == 0 else 1.0
        center = np.array(
            [cx + 0.35 * k, cy + side * clutter.near_field_offset, clutter.near_field_dims[2] / 2]
        )
        meshes.append(_box_mesh(center, _rotation(17.0 * k, 0.0, 0.0), clutter.near_field_dims))
    return meshes


def _resting_center_z(R: np.ndarray, dims: tuple[float, float, float]) -> float:
    """Center Z that places the lowest rotated cuboid corner on Z=0."""
    half = np.asarray(dims, dtype=np.float64) / 2.0
    corners = np.array(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ]
    )
    return -float(np.min((R @ corners.T).T[:, 2]))


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


def _validate_params(params: SynthParams) -> np.ndarray:
    """Validate generator inputs before constructing Open3D geometry."""
    for name in ("az_fov_deg", "el_fov_deg", "ang_res_deg", "ground_size"):
        value = float(getattr(params, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")
    for name in ("az_fov_deg", "el_fov_deg"):
        value = float(getattr(params, name))
        if value >= 180.0:
            raise ValueError(f"{name} must be less than 180 degrees, got {value}")
    for name in ("range_noise_std", "range_noise_slope"):
        value = float(getattr(params, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite, got {value}")
    for name in ("dropout", "stripe_dropout", "outlier_fraction"):
        value = float(getattr(params, name))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    if not np.isfinite(params.outlier_radius) or params.outlier_radius < 0.05:
        raise ValueError(
            "outlier_radius must be finite and at least 0.05 m, "
            f"got {params.outlier_radius}"
        )
    dims = np.asarray(params.dims, dtype=np.float64)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0.0):
        raise ValueError(f"dims must contain three positive finite values, got {params.dims}")
    sensor_pos = np.asarray(params.sensor_pos, dtype=np.float64)
    if sensor_pos.shape != (3,) or not np.all(np.isfinite(sensor_pos)):
        raise ValueError("sensor_pos must contain three finite coordinates")
    if params.look_at is not None:
        look_at = np.asarray(params.look_at, dtype=np.float64)
        if look_at.shape != (3,) or not np.all(np.isfinite(look_at)):
            raise ValueError("look_at must contain three finite coordinates")
        if np.linalg.norm(look_at - sensor_pos) <= 1e-12:
            raise ValueError("look_at must not coincide with sensor_pos")
    return sensor_pos


def _validate_spec(spec: BlockSpec, index: int) -> None:
    """Reject invalid block geometry before passing it to Open3D."""
    scalar_names = ("x", "y", "yaw_deg", "pitch_deg", "roll_deg")
    for name in scalar_names:
        value = float(getattr(spec, name))
        if not np.isfinite(value):
            raise ValueError(f"blocks[{index}].{name} must be finite, got {value}")
    if spec.z is not None and not np.isfinite(float(spec.z)):
        raise ValueError(f"blocks[{index}].z must be finite when provided, got {spec.z}")
    dims = np.asarray(spec.dims, dtype=np.float64)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0.0):
        raise ValueError(
            f"blocks[{index}].dims must contain three positive finite values, got {spec.dims}"
        )


def generate_scene(
    specs: list[BlockSpec],
    params: SynthParams | None = None,
    rng: np.random.Generator | None = None,
    clutter: ClutterParams | None = None,
) -> SceneSample:
    """Raycast a scene of blocks -> (pts_world (N,3) float64, list[BlockPose]).

    GT poses use the block geometric center and R = Rz(yaw)Ry(pitch)Rx(roll).
    `clutter`, if given, adds side environment (trees/container/logs) as
    unlabeled returns — it changes the cloud but never the ground truth.
    """
    if params is None:
        params = SynthParams()
    if rng is None:
        rng = np.random.default_rng()
    sensor_pos = _validate_params(params)
    for index, spec in enumerate(specs):
        _validate_spec(spec, index)

    cx = float(np.mean([s.x for s in specs])) if specs else 0.0
    cy = float(np.mean([s.y for s in specs])) if specs else 0.0

    gt: list[BlockPose] = []
    scene = o3d.t.geometry.RaycastingScene()
    block_geometry_ids: list[int] = []
    for spec in specs:
        R = _rotation(spec.yaw_deg, spec.pitch_deg, spec.roll_deg)
        z = spec.z if spec.z is not None else _resting_center_z(R, spec.dims)
        center = np.array([spec.x, spec.y, z], dtype=np.float64)
        block_geometry_ids.append(scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(_box_mesh(center, R, spec.dims))
        ))
        gt.append(BlockPose(position=center, rotation=R, confidence=1.0, source="synthetic"))

    if params.ground:
        half = params.ground_size / 2.0
        ground = o3d.geometry.TriangleMesh.create_box(params.ground_size, params.ground_size, 0.02)
        ground.translate([cx - half, cy - half, -0.02])
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ground))

    if clutter is not None:
        for mesh in _build_clutter(clutter, (cx, cy), rng):
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    if params.look_at is not None:
        target = np.asarray(params.look_at, dtype=np.float64)
    elif specs:
        target = np.array(
            [np.mean([s.x for s in specs]), np.mean([s.y for s in specs]), params.dims[2] / 2.0]
        )
    else:
        target = sensor_pos + np.array([1.0, 0.0, -1.0])
    if not np.all(np.isfinite(target)):
        raise ValueError("ray target must contain three finite coordinates")
    if np.linalg.norm(target - sensor_pos) <= 1e-12:
        raise ValueError("ray target must not coincide with sensor_pos")

    dirs = _ray_directions(sensor_pos, target, params)
    origins = np.broadcast_to(sensor_pos, dirs.shape)
    rays = o3d.core.Tensor(
        np.hstack([origins, dirs]).astype(np.float32), dtype=o3d.core.Dtype.Float32
    )
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    hit = np.isfinite(t_hit)

    if params.stripe_dropout > 0.0:
        n_az = max(int(params.az_fov_deg / params.ang_res_deg), 1)
        width = min(max(int(round(n_az * params.stripe_dropout)), 1), n_az)
        start = int(rng.integers(0, n_az - width + 1))
        columns = np.arange(len(dirs)) % n_az
        hit &= (columns < start) | (columns >= start + width)

    t = t_hit[hit]
    d = dirs[hit]
    geometry_ids = ans["geometry_ids"].numpy()[hit].astype(np.int64)
    primitive_ids = ans["primitive_ids"].numpy()[hit].astype(np.int64)
    instance_ids = np.full(len(t), -1, dtype=np.int64)
    for instance_id, geometry_id in enumerate(block_geometry_ids):
        instance_ids[geometry_ids == geometry_id] = instance_id
    face_ids = np.where(instance_ids >= 0, primitive_ids // 2, -1)
    if params.range_noise_std > 0.0 or params.range_noise_slope > 0.0:
        std = params.range_noise_std + params.range_noise_slope * np.maximum(t, 0.0)
        t = t + rng.normal(0.0, std, size=t.shape)
    pts = sensor_pos[None, :] + t[:, None] * d

    keep = np.ones(len(pts), dtype=bool)
    if params.dropout > 0.0:
        keep &= rng.random(len(pts)) >= params.dropout
    if not np.all(keep):
        pts = pts[keep]
        t = t[keep]
        d = d[keep]
        instance_ids = instance_ids[keep]
        face_ids = face_ids[keep]

    if params.outlier_fraction > 0.0 and len(pts):
        count = int(round(len(pts) * params.outlier_fraction))
    else:
        count = 0
    if count > 0:
        source = rng.integers(0, len(pts), size=count)
        offsets = rng.normal(size=(count, 3))
        offsets /= np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-12)
        offsets *= rng.uniform(0.05, params.outlier_radius, size=(count, 1))
        pts = np.vstack([pts, pts[source] + offsets])
        t = np.concatenate([t, np.linalg.norm(pts[-count:] - sensor_pos, axis=1)])
        d = np.vstack([d, (pts[-count:] - sensor_pos) / t[-count:, None]])
        instance_ids = np.concatenate([instance_ids, np.full(count, -1, dtype=np.int64)])
        face_ids = np.concatenate([face_ids, np.full(count, -1, dtype=np.int64)])

    manifest = {
        "generator": "blockpose.synth.v2",
        "params": asdict(params),
        "blocks": [asdict(spec) for spec in specs],
        "clutter": asdict(clutter) if clutter is not None else None,
    }
    return SceneSample(
        points=pts.astype(np.float64),
        ground_truth=gt,
        instance_ids=instance_ids,
        face_ids=face_ids,
        ranges=t.astype(np.float64),
        ray_directions=d.astype(np.float64),
        sensor_origin=sensor_pos,
        manifest=manifest,
    )


def random_scene(
    n_blocks: int,
    params: SynthParams | None = None,
    rng: np.random.Generator | None = None,
    spacing: float = 1.2,
    max_tilt_deg: float = 3.0,
    clutter: ClutterParams | None = None,
) -> SceneSample:
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
