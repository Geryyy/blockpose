# Blockpose

Blockpose is a standalone workspace for reliable 6-DoF pose estimation of
concrete blocks from point clouds. The current implementation is a
Python/Open3D detector used as the behavioral reference. The next development
step is to add a C++20 core and thin ROS 2 evaluation node so complete rosbags
can drive the port, beginning with `SCENE_DISCOVERY`; this scaffolding is
planned but is **not implemented in this repository yet**. It is an evaluation
path, not a production replacement. The current repository contains the
implementation roadmap, development environment, tests, and a curated export
of models and configuration from the source ROS workspace.

## Start here

- [`docs/current-detector-pipeline.md`](docs/current-detector-pipeline.md) is
  the canonical description of current, as-built detector behavior, including
  its default and opt-in paths, limits, and reproducible diagnostics.
- [`docs/next-steps.md`](docs/next-steps.md) is the active C++20/ROS 2 takeover
  sequence and defines its replay, reviewed-snapshot, and promotion gates.
- [`docs/implementation-plan.md`](docs/implementation-plan.md) is the future
  target architecture; it is not a description of the current implementation
  or the active ordering.
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
ctest, and clang-format targets do not exist yet; adding their minimal
scaffolding is the first task in [`docs/next-steps.md`](docs/next-steps.md).

## Label a replay scene-discovery capture

```bash
# Start replay with capture enabled; trigger Scene Discovery once in RViz.
ros2 launch concrete_block_perception remote_replay_world_model.launch.py \
  use_sim_time:=True rviz:=True tool:=pzs100_description \
  world_model_scene_discovery_params:=/workspaces/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_perception/config/grip_at_top_world_model_scene_discovery_capture.yaml

# Open the captured cloud/RGB/TF and seeded annotations; no uv required.
python3 tools/annotate_snapshot.py \
  --root /workspaces/ros2_baustelle_ws/scene_discovery_capture \
  /workspaces/ros2_baustelle_ws/scene_discovery_capture/<capture-name>
```

The capture directory contains `cloud.pcd`, `rgb.png`, `tf.yaml`, detector
priors, and `annotations.yaml`. The annotator defaults its ROI to the full
captured cloud and draws it cyan in both views; projected annotation boxes use
their review colors. Approve/reject priors, add missed blocks, then save. Use
the cloud/TF as authority when RGB is marked stale.

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
