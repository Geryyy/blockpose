#!/usr/bin/env bash
set -euo pipefail

bag_name="${1:-pzs100_reference}"
out_dir="${2:-measurements/reference}"

source /workspaces/ros2_baustelle_ws/install/setup.bash

mkdir -p "${out_dir}"

ros2 bag record -s sqlite3 -o "${out_dir}/${bag_name}" \
  /tf \
  /tf_static \
  /joint_states \
  /joint_states_rviz \
  /seyond/points \
  /seyond/points/cloudini \
  /blackfly_rotated/image_rect \
  /blackfly_rotated/image_rect/compressed \
  /blackfly_rotated/camera_info \
  /camera/image \
  /camera/image/compressed \
  /camera/camera_info \
  /clock

