# Development environment

Two Docker images, one uv-managed venv, no conda, no host dependencies beyond
Docker.

```text
docker/Dockerfile.core   Ubuntu 24.04, gcc-13, CMake/Ninja, libeigen3-dev,
                         libgtest-dev, libopencv-dev, uv + an image-baked
                         /opt/venv (open3d, opencv-python, torch, ...).
                         nanobind is NOT baked into the image: it is a `dev`
                         extra that `uv sync --all-extras` adds to /opt/venv.
                         Phases P–5 and all of CI live here. NO ROS.

docker/Dockerfile.ros2   FROM blockpose-core + ROS 2 Jazzy, colcon,
                         rosbag2 + MCAP plugin. Phase 6 wrapper, bag export,
                         live integration only.
```

## C++ build prerequisites

`cpp/` (the ROS-facing `blockpose_core` package, see
[`README.md`](../README.md#portable-c-core)) needs, at the system level:

- **libeigen3-dev** — `cpp/CMakeLists.txt` does
  `find_package(Eigen3 3.3 REQUIRED NO_MODULE)` unconditionally; every
  configure of the core fails without it, Python or vision or not.
- **libgtest-dev** — only needed to build/run the core's own unit tests
  (`-DBLOCKPOSE_CORE_BUILD_TESTING=ON`, the CMake default, and the
  `ctest` invocation under "Portable C++ core" in the README).
  `tools/check_cpp_consumer.sh` (the `make ship-check` gate) deliberately
  configures with testing OFF and does not need this package — see that
  script's own comment for why.
- **libopencv-dev** — needed for two independent reasons: it is what the
  optional `blockpose::vision` CMake target
  (`BLOCKPOSE_VISION_BUILD`/`find_package(OpenCV COMPONENTS core imgproc
  calib3d)`) links against, and it is what the compiled
  `blockpose._core` nanobind extension needs present as a *system* shared
  library at import time — `opencv-python`'s bundled, privately-namespaced
  runtime does not satisfy that (see `models/production/README.md`).
  Without it, `blockpose::vision` is silently skipped at configure time
  rather than failing the build, which makes the gap easy to miss.
- **nanobind** — a `dev`-extra pip package (`pyproject.toml`), not a system
  package. `find_package(nanobind CONFIG REQUIRED)` needs its CMake config
  pointed at explicitly; it is not on `CMAKE_PREFIX_PATH` by default. Build
  the extension with `-Dnanobind_DIR="$(python3 -m nanobind --cmake_dir)"`
  as shown under "Portable C++ core" in the README.

Ceres, a system Open3D C++ SDK, yaml-cpp, nlohmann-json and CLI11 are not
build dependencies of anything currently in `cpp/` or `docker/Dockerfile.core`
— an earlier version of this doc listed them, but `cpp/` links only against
Eigen (always) and OpenCV (optionally); Open3D is consumed exclusively as the
`open3d` Python wheel pinned in `pyproject.toml`. If a future change actually
needs one of those, add it to the apt list next to a comment recording why,
the same way `libeigen3-dev`/`libgtest-dev`/`libopencv-dev` are documented
above.

## Quick start

Docker Compose requires a host dataset directory to be provided via
`BLOCKPOSE_DATASET`; Compose mounts that path at `/data` inside the container.

```bash
export BLOCKPOSE_DATASET=$HOME/data/block_perception_dataset   # host path
make image          # build core image
make shell          # enter container
# inside:
make python-sync    # create .venv from pyproject/lockfile
make check          # pytest + lint (Python surface only)
make ship-check     # build + install cpp/ and run the ROS-facing consumer
                    # gate (tools/check_cpp_consumer.sh) -- not part of
                    # `check`; run it before shipping anything the ROS
                    # node links against, see Makefile for why it's split
```

`make check` does not touch `cpp/` at all — it is `pytest` + `ruff` only (see
`Makefile`). The C++ core has its own gate, `make ship-check`, which
configures, builds and installs `blockpose_core` with no Python and no
GTest, then compiles and runs `cpp/tests/consumer` against the *installed*
package the way the ROS node's `find_package(blockpose_core)` does:

```bash
tools/check_cpp_consumer.sh   # equivalent to `make ship-check`
# or, to build+install by hand and keep the tree around for inspection:
cmake -S cpp -B build/blockpose-core \
  -DBLOCKPOSE_CORE_BUILD_TESTING=OFF -DBLOCKPOSE_CORE_BUILD_PYTHON=OFF \
  -DCMAKE_INSTALL_PREFIX=/tmp/blockpose-core-install
cmake --build build/blockpose-core --parallel
cmake --install build/blockpose-core
cmake -S cpp/tests/consumer -B build/blockpose-consumer \
  -DCMAKE_PREFIX_PATH=/tmp/blockpose-core-install
cmake --build build/blockpose-consumer --parallel
build/blockpose-consumer/blockpose_core_consumer
```

To build and run the core's own unit tests instead (needs `libgtest-dev`,
see above):

```bash
cmake -S cpp -B build/blockpose-core-tests   # BUILD_TESTING=ON is the default
cmake --build build/blockpose-core-tests --parallel
ctest --test-dir build/blockpose-core-tests --output-on-failure
```

Or open the repo in VS Code / Claude Code and use the devcontainer
(`.devcontainer/devcontainer.json`) — same image; `postCreateCommand` runs
`uv sync --all-extras --inexact` (it does not configure `cpp/` — run
`make ship-check` or the CMake commands above for that). Unlike Docker
Compose, the devcontainer mounts the repo-local `data/` directory at `/data`
and sets `BLOCKPOSE_DATASET=/data` automatically.

## Conventions and rationale

- **Repo mounted at `/workspace`, dataset at `/data`.** Bags and canonical
  samples never live in the image or the repo; images stay stateless.
- **Open3D is a Python wheel, not a system SDK.** `pyproject.toml` pins
  `open3d==0.19.0`; `tests/python/test_environment.py` asserts against that
  pin. `docker/Dockerfile.core`'s own `uv pip install` of `open3d` (baked
  into `/opt/venv` at image build time) is currently unpinned — if that
  drifts from the `pyproject.toml` pin, `uv sync` inside the container
  reconciles it back to the pinned version, but the image build itself will
  not catch the mismatch.
- **The venv lives at `/opt/venv`, not in the repo.** `Dockerfile.core` sets
  `UV_PROJECT_ENVIRONMENT=/opt/venv` (and `VIRTUAL_ENV`/`PATH` to match), so
  `uv sync` and `make python-sync` write there, not to `/workspace/.venv`.
  Measured, not assumed: inside the core container `uv sync --all-extras`
  put `nanobind` and `open3d==0.19.0` into
  `/opt/venv/lib/python3.12/site-packages`. A `/workspace/.venv` left over
  from a host-side `uv sync` is ignored inside the container — an earlier
  version of this doc claimed the venv was in-project so it would persist
  across restarts; it does not, the image-baked one does. `uv sync` is
  idempotent and fast, so re-creating `/opt/venv` state is cheap anyway.
- **ROS is never sourced in the core image**, and even in the ROS image it is
  sourced only in interactive shells — plain-CMake builds inside the ROS image
  prove the core has no hidden ROS dependency.
- **GUI visualization is available via host X11 forwarding.** Docker Compose
  and the devcontainer pass `DISPLAY`, set `QT_X11_NO_MITSHM=1`, and mount the
  host X socket. Rerun can still run on the host (native app or `rerun --serve`)
  with the container connecting to it via host networking; `.rrd` files written
  to `/workspace` or `/data` can also be opened on the host directly.
- **Definition of done is two gates, not one.** `make check` (pytest + ruff)
  covers the Python surface; `make ship-check` covers the ROS-facing C++ one.
  A Python-only change can pass `check` while breaking what the ROS node
  links against — that is exactly how the missing Eigen went unnoticed. Both
  are run by hand inside the core container: there is no CI runner in this
  repo yet (no `.github/workflows/`), so "CI" here means the same two
  commands, not a hosted job.

## Known pitfalls

- If `cpp/` fails to configure with "Could not find a package configuration
  file provided by Eigen3" or "Could NOT find GTest", `libeigen3-dev` /
  `libgtest-dev` are missing from the image — see "C++ build prerequisites"
  above; this is the failure mode `make ship-check` exists to catch before
  it reaches the ROS-facing surface.
- `/tf_static` is latched at bag start: when trimming MCAPs, preserve the
  first messages or re-inject static transforms (see plan §data).
- GUI-based Open3D visualization inside the container uses the host X server.
  If windows fail to open, confirm `DISPLAY` is set before starting the
  container. Some desktops may still require allowing local Docker clients to
  connect to X11.
- On the crane PC, deploy the ros2 image as-is; it is the deployment
  artifact, not a separate packaging step.
