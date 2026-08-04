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
cmake -S "$root/cpp" -B "$work/core" \
  -DBLOCKPOSE_CORE_BUILD_TESTING=OFF -DBLOCKPOSE_CORE_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$work/install" >/dev/null
cmake --build "$work/core" -j"$(nproc)" >/dev/null
cmake --install "$work/core" >/dev/null
cmake -S "$root/cpp/tests/consumer" -B "$work/consumer" \
  -DCMAKE_PREFIX_PATH="$work/install" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$work/consumer" -j"$(nproc)" >/dev/null
"$work/consumer/blockpose_core_consumer"
