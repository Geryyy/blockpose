# Blockpose

Blockpose is a standalone workspace for reliable 6-DoF pose estimation of
concrete blocks from point clouds. Development is **Python-first**: the
current phase is a Python/Open3D prototype track (detection, registration,
evaluation against real snapshot data) used to de-risk the algorithmic
approach. Once the approach is validated, the core moves to C++20 with
pybind11 bindings and a thin, optional ROS 2 integration layer, per the
implementation plan — the C++/CMake/ROS 2 scaffolding has been removed for
now and returns at that point, not before. The current repository contains
the implementation roadmap, development environment, tests, and a curated
export of models and configuration from the source ROS workspace.

## Start here

- [`docs/current-detector-pipeline.md`](docs/current-detector-pipeline.md) is
  the canonical description of current, as-built detector behavior, including
  its default and opt-in paths, limits, and reproducible diagnostics.
- [`docs/implementation-plan.md`](docs/implementation-plan.md) is the future
  architecture and migration plan, including the later C++/ROS 2 phases; it is
  not a description of the current implementation.
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) explains the Docker and local
  development workflow.
- [`config/`](config/) contains calibration, perception, initialization, block
  seed, transform, and world-model configuration.
- [`assets/`](assets/) contains the block model, registration templates,
  environment models, and robot/sensor meshes.

## Repository layout

```text
assets/           models, meshes, templates, worlds, and textures
config/           runtime and reference configuration
docker/           core development image
docs/             maintained project documentation
launch_context/   reference ROS launch files from the source workspace
notes/            historical setup notes
python/           Python package scaffold
tests/            automated checks
tools/            recording and export helpers
urdf/             reference robot and sensor descriptions
```

`launch_context/`, `urdf/`, much of `config/`, and the existing assets are
reference/provenance material rather than a finished standalone runtime. Some
files still name packages from the original ROS workspace and are intentionally
kept unchanged so their origin and integration assumptions remain inspectable.
Treat those exported reference files as unmaintained unless their use is
documented by the current pipeline or environment documentation.

## Clone and large assets

Git LFS is used for the 291 MB Seibersdorf OBJ, its four high-resolution texture
PNGs, and the office-container Blender source. These files are required by the
full simulation environment, while the smaller runtime meshes remain ordinary
Git objects.

Install Git LFS before cloning, or initialize it and hydrate an existing clone:

```bash
git lfs install
git lfs pull
```

To verify that the expected assets are present rather than pointer files:

```bash
git lfs ls-files
```

Generated datasets, bags, and diagnostic dumps do not belong in this repository.
Keep them in an external dataset directory; canonical `.npz` samples and lockfiles
are deliberately not ignored when a small fixture or reproducibility lock needs
to be committed intentionally.

## Development

The supported environment uses Docker plus a uv-managed virtual environment;
see `docs/ENVIRONMENT.md` for details. Docker Compose requires a host dataset
path in `BLOCKPOSE_DATASET`, which it mounts at `/data` inside the container.

```bash
export BLOCKPOSE_DATASET=/absolute/path/to/block_perception_dataset
make image
make shell
# inside the container
make python-sync
make check
```

The devcontainer uses the same core image and automatically mounts the repo-local
`data/` directory at `/data` with `BLOCKPOSE_DATASET=/data`.

`make check` currently runs the Python test suite and lint only. C++ build,
ctest, and clang-format targets will return once the core migrates back to
C++/ROS 2.

## Reference bag workflow

No bag was present when the source material was exported. From the original ROS
workspace, record the point cloud, transforms, joint state, and camera metadata:

```bash
tools/record_reference_bag.sh pzs100_filter_ref
```

Then export the recording from this repository:

```bash
source install/setup.bash
python3 tools/export_reference_bag.py \
  measurements/reference/pzs100_filter_ref \
  --output reference_bag_export \
  --point-topic /seyond/points \
  --image-topic /blackfly_rotated/image_rect \
  --camera-info-topic /blackfly_rotated/camera_info
```

For gripper removal, record `/tf`, `/tf_static`, `/joint_states` (or
`/joint_states_rviz`), and `/seyond/points` together. The exporter can produce
point clouds, images, TF streams, camera information, and a manifest. Its output
directory is ignored because captured data should be curated into the external
dataset described by the implementation plan.

## Coordinate-frame note

The exported block stack uses ROS frame `world` for block seeds and the world
model. Gazebo has a separate world frame. The PZS100 spawner maps ROS `world`
into Gazebo `world` with:

```yaml
xyz: [0.0, -6.0, 0.0]
rpy_deg: [0.0, 0.0, 180.0]
```

Do not insert this spawn mapping into the ROS TF tree. The reference
`T_world_seyond` in `config/transforms_pzs100_initial_pose_1.yaml` is derived
from the URDF and `config/initialization/initialization_horizontal.yaml`.
