#!/usr/bin/env bash
# Move reviewed cleanup candidates into an external quarantine.  This program
# intentionally has no permanent-removal operation.
set -euo pipefail
IFS=$'\n\t'
LC_ALL=C

readonly BACKUP_COMMIT='48f10e307b855195a64342b8a140ad064c8042b0'
readonly EXPECTED_CANONICAL=23
readonly EXPECTED_BACKUPS=54
readonly EXPECTED_TRACKED_BACKUPS=5
readonly EXPECTED_UNTRACKED_BACKUPS=49
readonly SCRIPT_NAME="${0##*/}"

if (( BASH_VERSINFO[0] < 4 )); then
  printf '%s requires Bash 4 or newer.\n' "$SCRIPT_NAME" >&2
  exit 2
fi

usage() {
  cat <<'USAGE'
Usage: tools/quarantine_cleanup.sh [options]

Preview reviewed cleanup candidates by default.  Nothing is moved unless
--apply is provided and the typed confirmation is accepted.

Options:
  --group NAME                 Select one group (may be repeated):
                               backups, local-generated, immediate-docs,
                               editor-state, ros-provenance
  --all-safe                   Select all groups except ros-provenance.
  --include-ros-provenance     Permit selection of the ROS provenance group.
  --quarantine-parent DIR      Existing external same-filesystem directory.
                               Default: ignored .cleanup-quarantine/ in repo.
  --apply                      Move candidates after typed confirmation.
  --help                       Show this help.

With no --group option, all non-ROS groups are previewed.  The ROS provenance
group is deliberately opt-in and is never selected without
--include-ros-provenance. The quarantine is kept on the same filesystem, so
moves remain recoverable renames rather than copy-and-delete operations.
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

for command_name in git realpath stat du mv mkdir; do
  require_command "$command_name"
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die 'run this script from inside the target Git working tree'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
[[ "$REPO_ROOT" != / && -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ]] \
  || die 'Git root validation failed'

APPLY=0
INCLUDE_ROS=0
EXPLICIT_GROUPS=0
QUARANTINE_PARENT="$REPO_ROOT/.cleanup-quarantine"
QUARANTINE_PARENT_EXPLICIT=0
declare -a REQUESTED_GROUPS=()

while (( $# > 0 )); do
  case "$1" in
    --help)
      usage
      exit 0
      ;;
    --apply)
      APPLY=1
      ;;
    --all-safe)
      EXPLICIT_GROUPS=1
      REQUESTED_GROUPS+=(backups local-generated immediate-docs editor-state)
      ;;
    --include-ros-provenance)
      INCLUDE_ROS=1
      ;;
    --group)
      (( $# >= 2 )) || die '--group needs a group name'
      EXPLICIT_GROUPS=1
      REQUESTED_GROUPS+=("$2")
      shift
      ;;
    --group=*)
      EXPLICIT_GROUPS=1
      REQUESTED_GROUPS+=("${1#--group=}")
      ;;
    --quarantine-parent)
      (( $# >= 2 )) || die '--quarantine-parent needs a directory'
      QUARANTINE_PARENT="$2"
      QUARANTINE_PARENT_EXPLICIT=1
      shift
      ;;
    --quarantine-parent=*)
      QUARANTINE_PARENT="${1#--quarantine-parent=}"
      QUARANTINE_PARENT_EXPLICIT=1
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

if (( ! EXPLICIT_GROUPS )); then
  REQUESTED_GROUPS=(backups local-generated immediate-docs editor-state)
fi

declare -A GROUP_SELECTED=()
for group_name in "${REQUESTED_GROUPS[@]}"; do
  case "$group_name" in
    backups|local-generated|immediate-docs|editor-state)
      GROUP_SELECTED["$group_name"]=1
      ;;
    ros-provenance)
      (( INCLUDE_ROS )) || die 'ros-provenance requires --include-ros-provenance'
      GROUP_SELECTED["$group_name"]=1
      ;;
    *)
      die "unknown group: $group_name"
      ;;
  esac
done

declare -a ROS_PROVENANCE_PATHS=(
  assets/block_model/gazebo_concrete_block/model.config
  assets/block_model/gazebo_concrete_block/model.sdf
  assets/environment/seibersdorf_environment/materials/textures/texture_1001.png
  assets/environment/seibersdorf_environment/materials/textures/texture_1002.png
  assets/environment/seibersdorf_environment/materials/textures/texture_1003.png
  assets/environment/seibersdorf_environment/materials/textures/texture_1004.png
  assets/environment/seibersdorf_environment/meshes/seibersdorf_environment.mtl
  assets/environment/seibersdorf_environment/meshes/seibersdorf_environment.obj
  assets/environment/seibersdorf_environment/model.config
  assets/environment/seibersdorf_environment/model.sdf
  assets/environment/seibersdorf_office_container/meshes/model.dae
  assets/environment/seibersdorf_office_container/model.config
  assets/environment/seibersdorf_office_container/model.sdf
  assets/environment/seibersdorf_office_container/seibersdorf_office_container.blend
  assets/environment/worlds/empty.world
  assets/environment/worlds/epsilon_crane.world
  assets/sensors/meshes/blackfly.stl
  assets/sensors/meshes/hesai_frame.stl
  assets/sensors/meshes/livox_frame.stl
  assets/sensors/meshes/sensor_konsole.stl
  assets/sensors/meshes/seyond.stl
  config/block_seeds/world_model_seed_b0.yaml
  config/block_seeds/world_model_seed_legacy_2block.yaml
  config/block_seeds/world_model_seed_none.yaml
  config/block_seeds/world_model_seed_pick_place.yaml
  config/block_seeds/world_model_seed_truck_loading.yaml
  config/gazebo_frame_mapping.yaml
  config/initialization/initialization_abovelog.yaml
  config/initialization/initialization_horizontal.yaml
  config/initialization/initialization_outside.yaml
  config/initialization/initialization_transport.yaml
  config/initialization/initialization_truckbed.yaml
  config/initialization/initialization_zero.yaml
  config/initialization/initialization_zero_feasible.yaml
  config/perception/block_detection_tracking.yaml
  config/perception/block_registration.yaml
  config/world_model/world_model.yaml
  launch_context/gazebo_base.launch.py
  launch_context/gazebo_basic_pick_and_place_pzs100.launch.py
  launch_context/gazebo_model_bt.launch.py
  launch_context/gazebo_wall_assembly_pzs100.launch.py
  launch_context/pzs100_bringup.launch.py
  urdf/crane.urdf.xacro
  urdf/perception_kit.urdf.xacro
  urdf/perception_kit_standalone.urdf.xacro
  urdf/pzs100_initial_pose_1.urdf
  urdf/timber_loader_AIT.urdf.xacro
)
(( ${#ROS_PROVENANCE_PATHS[@]} == 47 )) || die 'internal ROS allowlist must contain 47 paths'

declare -a CANDIDATES=()
declare -A CANDIDATE_GROUP=()
declare -A CANDIDATE_SEEN=()

add_candidate() {
  local group="$1"
  local rel="$2"
  [[ "$rel" != /* && "$rel" != *'..'* && -n "$rel" ]] \
    || die "unsafe allowlist path: $rel"
  if [[ -n "${CANDIDATE_SEEN[$rel]+x}" ]]; then
    die "candidate selected by more than one group: $rel"
  fi
  CANDIDATE_SEEN["$rel"]=1
  CANDIDATE_GROUP["$rel"]="$group"
  CANDIDATES+=("$rel")
}

is_tracked() {
  git ls-files --error-unmatch -- "$1" >/dev/null 2>&1
}

require_untracked_ignored() {
  local rel="$1"
  ! git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1 \
    || die "local-generated candidate is tracked: $rel"
  git check-ignore -q -- "$rel" \
    || die "local-generated candidate is not ignored: $rel"
}

require_untracked() {
  local rel="$1"
  ! git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1 \
    || die "untracked-only candidate is tracked: $rel"
}

check_backup_guard() {
  local -a canonical=()
  local -a tracked_backups=()
  local -a untracked_backups=()
  local path

  git cat-file -e "${BACKUP_COMMIT}^{commit}" 2>/dev/null \
    || die "annotation recovery commit is unavailable: $BACKUP_COMMIT"
  mapfile -t canonical < <(git ls-files 'data/registration_snapshots/**/annotations.yaml')
  (( ${#canonical[@]} == EXPECTED_CANONICAL )) \
    || die "expected $EXPECTED_CANONICAL canonical annotations, found ${#canonical[@]}"
  for path in "${canonical[@]}"; do
    git diff --quiet -- "$path" || die "canonical annotation has unstaged changes: $path"
    git diff --cached --quiet -- "$path" || die "canonical annotation has staged changes: $path"
  done

  mapfile -t tracked_backups < <(git ls-files 'data/registration_snapshots/**/annotations.yaml.backup-*')
  mapfile -t untracked_backups < <(git ls-files --others --exclude-standard \
    'data/registration_snapshots/**/annotations.yaml.backup-*')
  (( ${#tracked_backups[@]} == EXPECTED_TRACKED_BACKUPS )) \
    || die "expected $EXPECTED_TRACKED_BACKUPS tracked backups, found ${#tracked_backups[@]}"
  (( ${#untracked_backups[@]} == EXPECTED_UNTRACKED_BACKUPS )) \
    || die "expected $EXPECTED_UNTRACKED_BACKUPS untracked backups, found ${#untracked_backups[@]}"
  (( ${#tracked_backups[@]} + ${#untracked_backups[@]} == EXPECTED_BACKUPS )) \
    || die "expected $EXPECTED_BACKUPS total backups"

  for path in "${tracked_backups[@]}" "${untracked_backups[@]}"; do
    add_candidate backups "$path"
  done
}

add_local_generated() {
  local rel
  local -a paths=(.venv build .cache .pytest_cache .ruff_cache snapshot.rrd registration_snapshot.rrd)
  for rel in "${paths[@]}"; do
    if [[ -e "$REPO_ROOT/$rel" || -L "$REPO_ROOT/$rel" ]]; then
      require_untracked_ignored "$rel"
      add_candidate local-generated "$rel"
    fi
  done
}

add_editor_state() {
  local rel
  for rel in .idea .vscode; do
    if [[ -e "$REPO_ROOT/$rel" || -L "$REPO_ROOT/$rel" ]]; then
      require_untracked "$rel"
      add_candidate editor-state "$rel"
    fi
  done
}

if [[ -n "${GROUP_SELECTED[backups]+x}" ]]; then
  check_backup_guard
fi
if [[ -n "${GROUP_SELECTED[local-generated]+x}" ]]; then
  add_local_generated
fi
if [[ -n "${GROUP_SELECTED[immediate-docs]+x}" ]]; then
  is_tracked notes/Setup.md || die 'immediate-docs allowlist item is not tracked: notes/Setup.md'
  add_candidate immediate-docs notes/Setup.md
fi
if [[ -n "${GROUP_SELECTED[editor-state]+x}" ]]; then
  add_editor_state
fi
if [[ -n "${GROUP_SELECTED[ros-provenance]+x}" ]]; then
  for ros_path in "${ROS_PROVENANCE_PATHS[@]}"; do
    is_tracked "$ros_path" || die "ROS allowlist item is not tracked: $ros_path"
    add_candidate ros-provenance "$ros_path"
  done
fi

validate_candidate() {
  local rel="$1"
  local source="$REPO_ROOT/$rel"
  local resolved
  [[ "$rel" != /* && "$rel" != *'..'* && "$rel" != *$'\n'* ]] \
    || die "unsafe candidate path: $rel"
  [[ -e "$source" || -L "$source" ]] || die "candidate disappeared: $rel"
  [[ ! -L "$source" ]] || die "refusing symlink candidate: $rel"
  resolved="$(realpath -e -- "$source")" || die "cannot resolve candidate: $rel"
  [[ "$resolved" == "$REPO_ROOT/"* && "$resolved" != "$REPO_ROOT" ]] \
    || die "candidate resolves outside repository: $rel"
  if is_tracked "$rel" && [[ -n "$(git status --porcelain=v1 -- "$rel")" ]]; then
    die "tracked candidate has staged or unstaged changes: $rel"
  fi
  # A candidate itself may not be a symlink.  Links inside an allowed
  # directory are preserved as link entries by mv; this script never walks
  # through them or follows them to an external target.
}

git_state_for() {
  local rel="$1"
  local status
  if is_tracked "$rel"; then
    status="$(git status --porcelain=v1 -- "$rel")"
    if [[ -n "$status" ]]; then
      printf 'tracked:%s' "${status%%$'\n'*}"
    else
      printf 'tracked:clean'
    fi
  else
    printf 'untracked'
  fi
}

format_size() {
  du -sh -- "$1" | awk '{print $1}'
}

for candidate in "${CANDIDATES[@]}"; do
  validate_candidate "$candidate"
done

if [[ -e "$QUARANTINE_PARENT" || -L "$QUARANTINE_PARENT" ]]; then
  QUARANTINE_PARENT="$(realpath -e -- "$QUARANTINE_PARENT")" \
    || die "cannot resolve quarantine parent: $QUARANTINE_PARENT"
  [[ -d "$QUARANTINE_PARENT" && ! -L "$QUARANTINE_PARENT" ]] \
    || die 'quarantine parent must be a real directory'
elif (( ! QUARANTINE_PARENT_EXPLICIT )) \
  && [[ "$QUARANTINE_PARENT" == "$REPO_ROOT/.cleanup-quarantine" ]]; then
  git check-ignore -q -- .cleanup-quarantine/probe \
    || die 'default in-repository quarantine must be ignored by Git'
else
  die "quarantine parent does not exist: $QUARANTINE_PARENT"
fi
if [[ "$QUARANTINE_PARENT" == "$REPO_ROOT" || "$QUARANTINE_PARENT" == "$REPO_ROOT/"* ]]; then
  [[ "$QUARANTINE_PARENT" == "$REPO_ROOT/.cleanup-quarantine" ]] \
    || die 'only the ignored .cleanup-quarantine directory is allowed inside the repository'
fi
SAME_FILESYSTEM=0
QUARANTINE_DEVICE_PATH="$QUARANTINE_PARENT"
[[ -e "$QUARANTINE_DEVICE_PATH" ]] || QUARANTINE_DEVICE_PATH="$REPO_ROOT"
if [[ "$(stat -c %d -- "$REPO_ROOT")" == "$(stat -c %d -- "$QUARANTINE_DEVICE_PATH")" ]]; then
  SAME_FILESYSTEM=1
fi
if (( APPLY && ! SAME_FILESYSTEM )); then
  die 'quarantine parent must be on the same filesystem as the repository'
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$QUARANTINE_PARENT" == / ]]; then
  QUARANTINE_ROOT="/$(basename -- "$REPO_ROOT")-quarantine-$TIMESTAMP"
else
  QUARANTINE_ROOT="$QUARANTINE_PARENT/$(basename -- "$REPO_ROOT")-quarantine-$TIMESTAMP"
fi
PAYLOAD_ROOT="$QUARANTINE_ROOT/payload"
[[ ! -e "$QUARANTINE_ROOT" && ! -L "$QUARANTINE_ROOT" ]] \
  || die "quarantine collision: $QUARANTINE_ROOT already exists"

for candidate in "${CANDIDATES[@]}"; do
  [[ ! -e "$PAYLOAD_ROOT/$candidate" && ! -L "$PAYLOAD_ROOT/$candidate" ]] \
    || die "destination collision: $PAYLOAD_ROOT/$candidate"
done

printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Mode: %s\n' "$([[ "$APPLY" == 1 ]] && printf apply || printf preview)"
printf 'Quarantine destination: %s\n' "$QUARANTINE_ROOT"
if (( ! SAME_FILESYSTEM )); then
  printf 'WARNING: this preview parent is on a different filesystem; choose an external same-filesystem --quarantine-parent before --apply.\n'
fi
printf 'Selected candidates: %d\n' "${#CANDIDATES[@]}"
printf '%-18s %-12s %-9s %s\n' GROUP SIZE GIT_STATE SOURCE
for candidate in "${CANDIDATES[@]}"; do
  printf '%-18s %-12s %-9s %s\n' \
    "${CANDIDATE_GROUP[$candidate]}" \
    "$(format_size "$REPO_ROOT/$candidate")" \
    "$(git_state_for "$candidate")" \
    "$candidate"
  printf '  -> %s\n' "$PAYLOAD_ROOT/$candidate"
done

if [[ -n "${GROUP_SELECTED[ros-provenance]+x}" ]]; then
  printf '\nInbound ROS-provenance references outside the selected bundle:\n'
  if command -v rg >/dev/null 2>&1; then
    if ! rg -n \
      'launch_context|urdf/|assets/environment|assets/sensors|gazebo_concrete_block|gazebo_frame_mapping|config/(block_seeds|initialization|perception|world_model)' \
      "$REPO_ROOT" \
      --glob '!launch_context/**' --glob '!urdf/**' --glob '!assets/environment/**' \
      --glob '!assets/sensors/**' --glob '!assets/block_model/gazebo_concrete_block/**' \
      --glob '!config/block_seeds/**' --glob '!config/initialization/**' \
      --glob '!config/perception/**' --glob '!config/world_model/**' \
      --glob '!config/gazebo_frame_mapping.yaml' --glob '!tools/quarantine_cleanup.sh' \
      --glob '!data/**' --glob '!.git/**' --glob '!.venv/**' \
      --glob '!.openclaude/worktrees/**'; then
      printf '  (none found)\n'
    fi
  else
    printf '  rg is unavailable; inspect references before applying.\n'
  fi
fi

if (( ! APPLY )); then
  printf '\nPreview only. Re-run with --apply to move these candidates into quarantine.\n'
  exit 0
fi

printf '\nType MOVE_TO_QUARANTINE to continue: '
IFS= read -r confirmation || die 'confirmation was not provided'
[[ "$confirmation" == MOVE_TO_QUARANTINE ]] || die 'confirmation did not match; no files were moved'

# Recheck every source and destination immediately before the first write.
for candidate in "${CANDIDATES[@]}"; do
  validate_candidate "$candidate"
  [[ ! -e "$PAYLOAD_ROOT/$candidate" && ! -L "$PAYLOAD_ROOT/$candidate" ]] \
    || die "destination collision: $PAYLOAD_ROOT/$candidate"
done

if [[ ! -d "$QUARANTINE_PARENT" ]]; then
  mkdir -- "$QUARANTINE_PARENT"
fi
mkdir -- "$QUARANTINE_ROOT"
mkdir -- "$PAYLOAD_ROOT"
MANIFEST="$QUARANTINE_ROOT/manifest.tsv"
RESTORE_SCRIPT="$QUARANTINE_ROOT/restore.sh"
printf 'relative_path\tgroup\tgit_state\tbytes\tdestination\n' > "$MANIFEST"

cat > "$RESTORE_SCRIPT" <<RESTORE
#!/usr/bin/env bash
set -euo pipefail
IFS=\$'\\n\\t'
readonly REPO_ROOT=$(printf '%q' "$REPO_ROOT")
readonly PAYLOAD_ROOT=$(printf '%q' "$PAYLOAD_ROOT")
readonly MANIFEST=$(printf '%q' "$MANIFEST")
[[ -d "\$REPO_ROOT/.git" || -f "\$REPO_ROOT/.git" ]] || { printf 'ERROR: repository root is unavailable: %s\\n' "\$REPO_ROOT" >&2; exit 1; }
while IFS=\$'\\t' read -r rel group state bytes destination; do
  [[ "\$rel" == relative_path ]] && continue
  [[ -n "\$rel" && "\$rel" != /* && "\$rel" != *'..'* ]] || { printf 'ERROR: unsafe manifest path: %s\\n' "\$rel" >&2; exit 1; }
  source="\$PAYLOAD_ROOT/\$rel"
  target="\$REPO_ROOT/\$rel"
  [[ -e "\$source" && ! -L "\$source" ]] || { printf 'ERROR: quarantined source unavailable: %s\\n' "\$source" >&2; exit 1; }
  [[ ! -e "\$target" && ! -L "\$target" ]] || { printf 'ERROR: restore would overwrite: %s\\n' "\$target" >&2; exit 1; }
  mkdir -p -- "\$(dirname -- "\$target")"
  mv -- "\$source" "\$target"
  printf 'Restored %s\\n' "\$rel"
done < "\$MANIFEST"
RESTORE
chmod 700 "$RESTORE_SCRIPT"

for candidate in "${CANDIDATES[@]}"; do
  source="$REPO_ROOT/$candidate"
  destination="$PAYLOAD_ROOT/$candidate"
  mkdir -p -- "$(dirname -- "$destination")"
  bytes="$(du -sb -- "$source" | awk '{print $1}')"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$candidate" "${CANDIDATE_GROUP[$candidate]}" "$(git_state_for "$candidate")" \
    "$bytes" "$destination" >> "$MANIFEST"
  mv -- "$source" "$destination"
done

printf '\nMoved %d candidate(s) to %s\n' "${#CANDIDATES[@]}" "$QUARANTINE_ROOT"
printf 'Restore with: %q\n' "$RESTORE_SCRIPT"

declare -a TRACKED_SOURCES=()
for candidate in "${CANDIDATES[@]}"; do
  if is_tracked "$candidate"; then
    TRACKED_SOURCES+=("$candidate")
  fi
done
if (( ${#TRACKED_SOURCES[@]} > 0 )); then
  printf 'To stage only the tracked moves, run:\n  git add -u --'
  for candidate in "${TRACKED_SOURCES[@]}"; do
    printf ' %q' "$candidate"
  done
  printf '\n'
fi
