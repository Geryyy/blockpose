# Blockpose

Blockpose is a standalone workspace for reliable 6-DoF pose estimation of
concrete blocks from point clouds. The intended implementation is a C++20 core
with Python bindings and a thin, optional ROS 2 integration layer. The current
repository is an early scaffold: it contains the implementation roadmap,
development environment, tests for that environment, and a curated export of
models and configuration from the source ROS workspace.

## Start here

- [`block_pose_estimation_plan_v4.md`](block_pose_estimation_plan_v4.md) is the
  canonical implementation plan and the source of truth for agentic work. Its
  versioned filename is retained for now to preserve history and existing links;
  do not create a second copy under a generic name.
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) explains the Docker and local
  development workflow.
- [`source_manifest.yaml`](source_manifest.yaml) records where exported
  reference material came from and what has not yet been exported.
- [`config/`](config/) contains calibration, perception, initialization, block
  seed, transform, and world-model configuration.
- [`assets/`](assets/) contains the block model, registration templates,
  environment models, and robot/sensor meshes.

## Repository layout

```text
apps/             small executables and environment checks
assets/           models, meshes, templates, worlds, and textures
config/           runtime and reference configuration
docker/           core and ROS 2 development images
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
Consult `source_manifest.yaml` before treating an exported file as maintained
Blockpose code.

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
see `docs/ENVIRONMENT.md` for details.

```bash
export BLOCKPOSE_DATASET=/absolute/path/to/block_perception_dataset
make image
make shell
# inside the container
make python-sync
make check
```

The devcontainer uses the same core image. The planned core path is ROS-free;
the ROS 2 image exists for bag export and later integration only.

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
