#!/usr/bin/env bash
# Verify the ROS-facing C++ surface: configure, build and INSTALL the core with
# no Python and no GTest, then compile and run a downstream consumer against the
# installed package. Run this before shipping anything the ROS node links.
#
# GTest is not installed in every dev container, so testing is off here on
# purpose: the ROS node does not need it, and requiring it would make this
# check unrunnable exactly where it matters.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${1:-$(mktemp -d)}"
# A caller-supplied work dir used to be allowed not to exist yet: every
# consumer of $work was a `cmake -B`, which creates it. The configure log
# below is a plain redirect, which does not -- and a missing dir would then
# be misreported as "configure failed" when configure never ran.
mkdir -p "$work"

# Preflight: fail with a diagnostic naming the fix instead of letting a
# missing prerequisite surface as a bare CMake configure error. This check
# exists because it already happened once: Eigen3 was absent from this
# container and docker/Dockerfile.core does not install it, and the only
# symptom was `Could not find a package configuration file provided by
# "Eigen3"` with cmake's own output swallowed by the >/dev/null below.
if ! command -v cc >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1 \
  && ! command -v gcc >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
  echo "check_cpp_consumer.sh: no C/C++ compiler found on PATH." >&2
  echo "  Install one, e.g. 'apt-get install build-essential'." >&2
  echo "  See docs/ENVIRONMENT.md for the expected dev container." >&2
  exit 1
fi

configure_log="$work/configure.log"
if ! cmake -S "$root/cpp" -B "$work/core" \
  -DBLOCKPOSE_CORE_BUILD_TESTING=OFF -DBLOCKPOSE_CORE_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$work/install" \
  >"$configure_log" 2>&1; then
  if grep -qi 'eigen3' "$configure_log"; then
    echo "check_cpp_consumer.sh: CMake could not find Eigen3." >&2
    echo "  Install it, e.g. 'apt-get install libeigen3-dev' (provides the" >&2
    echo "  CMake package Eigen3::Eigen via NO_MODULE config, required by" >&2
    echo "  cpp/CMakeLists.txt). See docs/ENVIRONMENT.md for the expected" >&2
    echo "  dev container, which is supposed to have it preinstalled." >&2
  else
    echo "check_cpp_consumer.sh: core CMake configure failed for a reason" >&2
    echo "  other than a missing prerequisite this script knows to diagnose." >&2
    echo "  Full cmake output:" >&2
    cat "$configure_log" >&2
  fi
  exit 1
fi
cmake --build "$work/core" -j"$(nproc)" >/dev/null
cmake --install "$work/core" >/dev/null
cmake -S "$root/cpp/tests/consumer" -B "$work/consumer" \
  -DCMAKE_PREFIX_PATH="$work/install" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$work/consumer" -j"$(nproc)" >/dev/null
"$work/consumer/blockpose_core_consumer"
