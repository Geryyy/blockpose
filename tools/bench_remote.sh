#!/usr/bin/env bash
# Sync this checkout to cds-ebc, build blockpose._core there, run tools/bench.py, fetch the JSON.
# usage: tools/bench_remote.sh [bench.py args]   e.g. --seeds 25 --jobs 16
set -euo pipefail
HOST=${BENCH_HOST:-geraldebmer@100.101.15.36}
BASE=${BENCH_BASE:-cbs_sim/bench}  # relative to the remote home
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=$(cd "$REPO/../.." && pwd)  # workspace src/, the layout bench.py resolves its inputs from
SSH=(ssh -o BatchMode=yes "$HOST")

git -C "$REPO" describe --always --dirty > "$REPO/REV"
"${SSH[@]}" "mkdir -p $BASE/src/concrete_block_stack/concrete_block_detector/config \
  $BASE/src/sensor_simulation/seyond_simulation/scan_mode"
rsync -a --delete --exclude '__pycache__' --exclude '*.so' \
  --include '/python/***' --include '/cpp/***' --include '/tools/***' --include '/REV' \
  --exclude '*' "$REPO/" "$HOST:$BASE/src/concrete_block_stack/blockpose/"
rm -f "$REPO/REV"
rsync -a "$SRC/concrete_block_stack/concrete_block_detector/config/detector.yaml" \
  "$HOST:$BASE/src/concrete_block_stack/concrete_block_detector/config/"
rsync -a "$SRC/sensor_simulation/seyond_simulation/scan_mode/robin_w.csv" \
  "$HOST:$BASE/src/sensor_simulation/seyond_simulation/scan_mode/"

# venv: python3 -m venv $BASE/venv && pip install numpy open3d==0.19.0 pyyaml scipy nanobind
"${SSH[@]}" "set -e; cd $BASE; B=src/concrete_block_stack/blockpose
  cmake -S \$B/cpp -B build -DCMAKE_BUILD_TYPE=Release -DBLOCKPOSE_CORE_BUILD_PYTHON=ON \
    -DBLOCKPOSE_VISION_BUILD=OFF -DPython_EXECUTABLE=\$PWD/venv/bin/python \
    -Dnanobind_DIR=\$(venv/bin/python -m nanobind --cmake_dir) > build.log
  cmake --build build -j8 >> build.log && cp build/_core*.so \$B/python/blockpose/
  venv/bin/python \$B/tools/bench.py --out \$PWD/runs $*"
mkdir -p "$REPO/runs/bench"
rsync -a "$HOST:$BASE/runs/" "$REPO/runs/bench/"
