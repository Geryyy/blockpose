#!/usr/bin/env bash
# Record a reference bag for offline evaluation (see tools/export_reference_bag.py,
# which turns the recorded bag into RGB images, point clouds and TF).
#
# This script was originally written against exactly one ROS workspace and one
# sensor stack, with the setup path and topic list baked in as literals. That
# meant every operator on a different workspace or sensor rig had to edit the
# script itself to record a bag at all. Both are now overridable, with the
# original values kept as defaults so existing invocations are unchanged.
#
# Usage:
#   tools/record_reference_bag.sh [bag_name] [out_dir]
#   tools/record_reference_bag.sh --help
#
# Environment variables:
#   ROS_SETUP_BASH   Path to the ROS 2 workspace setup.bash to source.
#                     Default: /workspaces/ros2_baustelle_ws/install/setup.bash
#   BAG_TOPICS        Whitespace-separated list of topics to record, overriding
#                     the default list below (set this to change *which*
#                     topics are captured, not just how many).
#
# set -euo pipefail: fail loudly rather than record a partial or empty bag.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/record_reference_bag.sh [bag_name] [out_dir]

Records a ROS 2 bag (sqlite3 storage) to "<out_dir>/<bag_name>".

Arguments:
  bag_name   Name of the bag to create (default: pzs100_reference)
  out_dir    Directory to write the bag into (default: measurements/reference)

Environment variables:
  ROS_SETUP_BASH   Path to the ROS 2 workspace setup.bash to source before
                    recording. Default:
                    /workspaces/ros2_baustelle_ws/install/setup.bash
  BAG_TOPICS       Whitespace-separated list of topics to record, replacing
                    the default list (see the script header for what the
                    defaults are and why each one is there).

Examples:
  tools/record_reference_bag.sh
  tools/record_reference_bag.sh my_bag measurements/scratch
  ROS_SETUP_BASH=/opt/ros/humble/setup.bash tools/record_reference_bag.sh
  BAG_TOPICS="/tf /clock /my/points" tools/record_reference_bag.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

bag_name="${1:-pzs100_reference}"
out_dir="${2:-measurements/reference}"

ros_setup_bash="${ROS_SETUP_BASH:-/workspaces/ros2_baustelle_ws/install/setup.bash}"

# Default topic list, unchanged from when it was a literal in the record
# command below. What each group is for, and how much of it is actually
# verifiable from this repository rather than assumed:
#   - /tf, /tf_static: read by tools/export_reference_bag.py (--tf-topic /tf,
#     --tf-static-topic /tf_static, written out as tf/*.jsonl). Without them
#     the export has no way to place cloud and camera frames.
#   - /joint_states, /joint_states_rviz: NOT read by the exporter. They are
#     recorded because README's "Reference bag workflow" section calls for
#     `/tf`, `/tf_static`, `/joint_states` (or `/joint_states_rviz`) and
#     `/seyond/points` together for gripper removal. Both names are kept
#     because the rig publishes both -- see launch_context/pzs100_bringup.
#     launch.py, which drives the arm off `joint_states_rviz` (the
#     per-rail-corrected EPSCOPE q9), not plain `joint_states`.
#   - /seyond/points: the LiDAR cloud, and the exporter's --point-topic
#     default. /seyond/points/cloudini is its compressed companion topic; the
#     exporter reads the raw one, the compressed one is recorded so the bag
#     stays replayable against the compressed consumer.
#   - /blackfly_rotated/image_rect, /blackfly_rotated/camera_info: the
#     exporter's --image-topic and --camera-info-topic defaults, plus the
#     /compressed image variant.
#   - /camera/image(/compressed), /camera/camera_info: these appear nowhere
#     else in this repository -- no tool defaults to them and no launch file
#     publishes them. They were in the original hardcoded list, so they are
#     kept verbatim to leave the default invocation byte-identical; treat
#     them as second-rig topic names, not as a documented fallback.
#   - /clock: sim/replay time source. The launch files under launch_context/
#     run with use_sim_time: True, so a bag replayed for them needs /clock or
#     time never advances.
# Override with BAG_TOPICS if your workspace uses different topic names.
default_topics=(
  /tf
  /tf_static
  /joint_states
  /joint_states_rviz
  /seyond/points
  /seyond/points/cloudini
  /blackfly_rotated/image_rect
  /blackfly_rotated/image_rect/compressed
  /blackfly_rotated/camera_info
  /camera/image
  /camera/image/compressed
  /camera/camera_info
  /clock
)

if [[ -n "${BAG_TOPICS:-}" ]]; then
  # Word-split on whitespace, but with pathname expansion off. Leaving globbing
  # on is not theoretical: BAG_TOPICS="tf_* /clock", run from a directory
  # containing files tf_a and tf_b, expands to `tf_a tf_b /clock` and records
  # the wrong topics with no error at all. Topic names cannot legally contain
  # glob characters, so nothing is lost by disabling it, and a silently wrong
  # recording is the one failure this script cannot warn about after the fact.
  set -f
  # shellcheck disable=SC2206 # intentional word-splitting on a user-supplied list
  topics=(${BAG_TOPICS})
  set +f
else
  topics=("${default_topics[@]}")
fi

# Preflight: fail with a diagnostic naming the fix instead of letting a
# missing setup file or missing ROS 2 install surface as an opaque
# "No such file or directory" from `source`, or a `ros2: command not found`
# after the bag directory has already been created.
if [[ ! -f "${ros_setup_bash}" ]]; then
  echo "record_reference_bag.sh: ROS setup file not found: ${ros_setup_bash}" >&2
  echo "  Set ROS_SETUP_BASH to the setup.bash of your ROS 2 workspace, e.g." >&2
  echo "    ROS_SETUP_BASH=/opt/ros/humble/setup.bash tools/record_reference_bag.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ros_setup_bash}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "record_reference_bag.sh: 'ros2' not found on PATH after sourcing" >&2
  echo "  ${ros_setup_bash}" >&2
  echo "  Check that this is a valid ROS 2 workspace/install setup file." >&2
  exit 1
fi

mkdir -p "${out_dir}"

ros2 bag record -s sqlite3 -o "${out_dir}/${bag_name}" "${topics[@]}"
