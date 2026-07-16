# Development environment

Two Docker images, one uv-managed venv, no conda, no host dependencies beyond
Docker.

```text
docker/Dockerfile.core   Ubuntu 24.04, gcc-13, CMake/Ninja/ccache,
                         Eigen 3.4, Ceres 2.2, Open3D C++ SDK 0.19.0 (/opt/open3d),
                         yaml-cpp, nlohmann-json, CLI11, pybind11, GTest, uv.
                         Phases P–5 and all of CI live here. NO ROS.

docker/Dockerfile.ros2   FROM blockpose-core + ROS 2 Jazzy, colcon,
                         rosbag2 + MCAP plugin. Phase 6 wrapper, bag export,
                         live integration only.
```

## Quick start

Docker Compose requires a host dataset directory to be provided via
`BLOCKPOSE_DATASET`; Compose mounts that path at `/data` inside the container.

```bash
export BLOCKPOSE_DATASET=$HOME/data/block_perception_dataset   # host path
make image          # build core image
make shell          # enter container
# inside:
make python-sync    # create .venv from pyproject/lockfile
make check          # C++ build + ctest (env_check) + pytest + lint
```

Or open the repo in VS Code / Claude Code and use the devcontainer
(`.devcontainer/devcontainer.json`) — same image, `postCreateCommand` runs
`uv sync` and the CMake configure. Unlike Docker Compose, the devcontainer
mounts the repo-local `data/` directory at `/data` and sets
`BLOCKPOSE_DATASET=/data` automatically.

## Conventions and rationale

- **Repo mounted at `/workspace`, dataset at `/data`.** Bags and canonical
  samples never live in the image or the repo; images stay stateless.
- **Version pinning has exactly two synchronization points:**
  `OPEN3D_VERSION` in `Dockerfile.core` must equal the `open3d` wheel pin in
  `pyproject.toml` (asserted by `tests/python/test_environment.py`). Bump both
  together.
- **The venv lives in-project (`/workspace/.venv`)** so it persists across
  container restarts via the repo mount. `uv sync` is idempotent and fast.
- **ROS is never sourced in the core image**, and even in the ROS image it is
  sourced only in interactive shells — plain-CMake builds inside the ROS image
  prove the core has no hidden ROS dependency.
- **ccache** persists in `/workspace/.cache/ccache` (repo mount), so image
  rebuilds don't lose compilation caches.
- **GUI visualization is available via host X11 forwarding.** Docker Compose
  and the devcontainer pass `DISPLAY`, set `QT_X11_NO_MITSHM=1`, and mount the
  host X socket. Rerun can still run on the host (native app or `rerun --serve`)
  with the container connecting to it via host networking; `.rrd` files written
  to `/workspace` or `/data` can also be opened on the host directly.
- **`make check` == CI.** The agent's definition of done for any change is
  `make check` passing inside the core container.

## Known pitfalls

- If the Open3D SDK download 404s, the asset naming changed — check
  https://github.com/isl-org/Open3D/releases and fix the URL in
  `Dockerfile.core` (keep the version pin discipline).
- `/tf_static` is latched at bag start: when trimming MCAPs, preserve the
  first messages or re-inject static transforms (see plan §data).
- GUI-based Open3D visualization inside the container uses the host X server.
  If windows fail to open, confirm `DISPLAY` is set before starting the
  container. Some desktops may still require allowing local Docker clients to
  connect to X11.
- On the crane PC, deploy the ros2 image as-is; it is the deployment
  artifact, not a separate packaging step.
