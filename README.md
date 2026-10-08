# Blockpose

Blockpose is the development workspace for concrete-block perception data,
annotation, offline evaluation, and learned RGB priors. The production ROS 2
detector lives in the sibling `concrete_block_detector` package; this repository
trains and evaluates optional RGB evidence modules against captured rosbags.

## Start here

- [`docs/STATUS.md`](docs/STATUS.md) — **read this first.** Current measured
  state, what ships, what is deliberately excluded, and the pick-up point for
  ROS 2 integration.
- [`docs/detection-pipeline.md`](docs/detection-pipeline.md) — how the geometry
  detector actually works: what happens to a point cloud at every stage, and
  what `DetectionResult` contains.
- [`docs/dino-rgb-prior-ros-integration.md`](docs/dino-rgb-prior-ros-integration.md)
  is the authoritative architecture and integration contract for the optional
  DINO RGB proposer and verifier. **Its `ScoreRgbCandidates` contract is known
  to encode a falsified usage of `dino_verifier` — see Phase 4 of the
  readiness plan below before implementing against it.**
- [`docs/production-readiness-plan.md`](docs/production-readiness-plan.md) is
  the current plan for getting those RGB priors production-ready, with the
  measured state, the open defects, and explicit exit gates and kill criteria.
- [`docs/annotation-tool.md`](docs/annotation-tool.md) explains how to review
  scene-discovery captures and create ground truth.
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
reference/provenance material, not a standalone ROS runtime. The runtime
integration boundary is documented in the DINO RGB prior architecture.

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

`make check` runs the repository checks. The production C++ detector has its
own build and replay tests in the source ROS workspace.

## Portable C++ core

`cpp/` is the production algorithm: it contains the ROS-free filtering,
including the PZS100 K9/K11 → K10/K12 outward rail-box self-filter, geometry,
priors, cuboid refinement, selection, and candidate tracing used by
`concrete_block_detector`. Adapters resolve their own TF names and pass the
world-frame rail poses into this shared implementation. Build and install it
before configuring the ROS wrapper:

The standalone PZS100 profile is
[`config/pzs100_gripper_filter.yaml`](config/pzs100_gripper_filter.yaml). It
is intentionally ROS-free and can be used by snapshot/annotation adapters on
the GPU machine.

The snapshot annotator uses this profile automatically when a capture contains
the recorded `world -> elastic/K9`, `K10_left_rail`, `K11`, and
`K12_right_rail` transforms.  The **Remove PZS100 gripper points** preprocessing
toggle starts enabled only when that TF, profile, and the native `_core` API are
available; it removes the same points from the display, auto-fit, and manual
snap operations.  A legacy `gripper_exclusions.yaml` remains supported as a
fallback when the native PZS100 filter cannot be configured, but is no longer
required for PZS100 captures.

```bash
cmake -S cpp -B .build/blockpose-core -DBLOCKPOSE_CORE_BUILD_TESTING=ON
cmake --build .build/blockpose-core --parallel
ctest --test-dir .build/blockpose-core --output-on-failure
cmake --install .build/blockpose-core --prefix <blockpose-core-prefix>

# The ROS package uses normal CMake package discovery; no relative includes.
CMAKE_PREFIX_PATH=<blockpose-core-prefix>:$CMAKE_PREFIX_PATH \
  colcon build --packages-select concrete_block_detector
```

The optional `blockpose._core` nanobind module exposes the same geometry core
to standalone Python tools. It is intentionally separate from the pure Python
annotation/training package while packaging is being consolidated (see
[`docs/cpp-nanobind-consolidation.md`](docs/cpp-nanobind-consolidation.md)
for the full migration plan and current gaps):

```bash
python3 -m pip install nanobind
cmake -S cpp -B build/blockpose-python -DBLOCKPOSE_CORE_BUILD_PYTHON=ON \
  -DBLOCKPOSE_CORE_BUILD_TESTING=OFF -DBLOCKPOSE_VISION_BUILD=OFF \
  -Dnanobind_DIR="$(python3 -m nanobind --cmake_dir)"
cmake --build build/blockpose-python --parallel
cp build/blockpose-python/_core.*.so python/blockpose/
```

The `.so` must be copied into `python/blockpose/` directly, not installed via
`cmake --install` into `site-packages/blockpose/` — this repo's `blockpose`
package is an editable install that resolves via
`_editable_impl_blockpose.pth` straight to `python/blockpose/`, so anything
installed into `site-packages/blockpose/` is dead and never imported.

The compiled extension links against the system OpenCV runtime
(`libopencv-dev`, not just pip's `opencv-python`) — see
`docker/Dockerfile.core` and `models/production/README.md` if
`import blockpose._core` fails with a missing `libopencv_*.so`.

`blockpose._core.discover(points)` now also returns `refined_candidate_trace`
and the full `PipelineCounts`, beyond the original geometry-only baseline;
`discover()` still only runs with C++ defaults (no custom
`DetectionParameters` from Python yet). See
[`docs/cpp-nanobind-consolidation.md`](docs/cpp-nanobind-consolidation.md)
for exactly what is and isn't bound today.

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
directory is ignored; curate reviewed captures in the external dataset mount
used for offline evaluation and training.

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
from the URDF and `config/initialization/initialization_horizontal.yaml` — and
holds unchanged for `initialization_outside.yaml` (`initial_pose:=2`), which the
Gazebo launches default to since issue 235. See that file's
`reference.applies_to` for why, and which poses it does not cover.
