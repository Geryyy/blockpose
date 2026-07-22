# Cleanup inventory — 2026-07-22

This is a deletion *review list*, not an instruction that any item is safe to
remove without the checks below.  It covers repository content as well as local
generated files, with those two classes deliberately kept separate.

## Scope and current state

The repository is now Python/Open3D-first.  The canonical description of the
implemented detector is [the current pipeline](current-detector-pipeline.md).
The C++/CMake/ROS 2 scaffolding is explicitly deferred in the
[README](../README.md).  This inventory identifies old planning material and
the ROS/Gazebo simulation stack which no longer support that direction.

Annotation preservation was completed before this review:

- **23 canonical, tracked `annotations.yaml` files are clean**.
- The annotation resolution commit is `48f10e307b855195a64342b8a140ad064c8042b0`
  (`Resolve duplicate sequence 3 annotation`).
- There are **54** timestamped `annotations.yaml.backup-*` files: **5 tracked**
  and **49 untracked**.  They occupy about **57 KB**.  Do not treat them as
  canonical annotations; retain the 23 canonical files and the above commit as
  the recovery point.

The working tree also contains a user-modified `docs/.obsidian/workspace.json`.
Do not overwrite, stage, or delete it as part of this pass.  There are multiple
`.openclaude/worktrees/` Git worktrees; they may have active owners and are not
ordinary build artefacts.  Do not delete or prune them during repository
cleanup.

## Immediate tracked deletion candidates

These files have a direct replacement or are an old personal checklist.  Review
their history/references once, then delete them in a dedicated Git commit if
the owner agrees.

| Path | Size | Why it is a candidate | Replacement / check |
| --- | ---: | --- | --- |
| `notes/Setup.md` | small checklist | Historical, partially completed project notes, not a maintained runbook. | Keep only any still-open task in an issue or current plan. |

## Consolidate or replace before deleting

These are not immediate deletions: each contains useful context, but its title
or assumptions make it easy to mistake for current behavior.  Preserve the
needed material in the indicated destination and then remove the old source.

| Path | Size | Proposed treatment | Risk / required check |
| --- | ---: | --- | --- |
| `docs/implementation-plan.md` | 29,296 B | Replace with a short, explicitly future/archived roadmap, or extract only still-approved milestones into the README and current pipeline. | It is linked from README and documents a deferred C++/ROS architecture. Update that link in the same commit. |
| `docs/detector-plan.md` | 15,846 B | Fold measured assumptions and accepted decisions into `current-detector-pipeline.md`; archive/delete the historical iteration narrative. | It links to `implementation-plan.md`; do not leave a dead cross-link. |
| `docs/verify-placed.md` | 5,020 B | Keep while v1 is opt-in, or merge its detailed v1/course safeguards into `current-detector-pipeline.md` before deleting. | Its 3 cm v1 contract is current opt-in behavior; the facade's legacy default is a separate 5 cm contract. Preserve that distinction and update inbound links. |
| `docs/ENVIRONMENT.md` | 3,506 B | Keep only the Python/Open3D developer workflow; remove ROS deployment claims if the ROS stack is removed. | README links to it and the Docker files currently support it. |

`docs/Snapshot Annotation.md` is a unique historical approval/deletion ledger,
not duplicate tool instructions. Archive it, or transfer that provenance into a
maintained dataset record and update links, before considering deletion.

Keep `docs/current-detector-pipeline.md` and `docs/annotation-tool.md`: they are
the current canonical behavior and annotation guidance, respectively.

## ROS and simulation provenance bundle — owner decision

If ROS/Gazebo simulation is formally out of scope, archive or externalize the
following tracked provenance as one coherent change rather than leaving
dangling scenes, launch files, or robot descriptions. The originally audited
bundle contained 57 tracked paths. The narrowed candidate list below contains
47 tracked files after excluding 10 active environment, calibration, and
planned gripper inputs; recalculate it again immediately before acting. The
remaining large component is `assets/environment/` at about
**352 MB**; the broader audited ROS/simulation material occupied about **367
MB** apparent size.

```text
launch_context/
urdf/
assets/environment/
assets/sensors/
assets/block_model/gazebo_concrete_block/
config/gazebo_frame_mapping.yaml
config/block_seeds/
config/initialization/
config/perception/
config/world_model/
```

Review separately before including them in this bundle:

- `assets/block_model/ConcreteBlock.ply` may still be useful to the Python
  prototype, whereas its `gazebo_concrete_block/` subdirectory is simulation
  only.
- `.devcontainer/`, `docker/`, and `docker-compose.yml` are the active
  Python/Open3D development and test environment. They are explicit keep items,
  not part of a ROS cleanup.
- `config/calibration/` is active projection runtime/tool/test input and must be
  kept.
- `assets/pzs100/` supports the explicitly planned gripper-aware filter and is
  an owner decision; keep it during a legacy sweep.
- `assets/templates/` and `assets/wall_plans.yaml` have Python consumers; keep
  them unless the relevant code and tests are removed too.
- `config/scenes/`, `config/detector_eval.yaml`, `config/snapshot_*`, and
  `config/wall_plans.yaml` are used by the active prototype/evaluation flow and
  are explicit **keep** items.

Before removal, search `README.md`, `Makefile`, Docker/Compose files, launch
files, `python/`, `tools/`, and tests for every top-level bundle path.  If a
deployment workflow still consumes the URDF, calibration, or world model,
defer this bundle rather than deleting a subset.

### Separate ROS-dependent owner decisions

The following are ROS-dependent but should not be silently folded into the
simulation bundle:

- `tools/record_reference_bag.sh` and `tools/export_reference_bag.py` require an
  external ROS workspace/runtime. Remove them only with the corresponding
  README reference-bag workflow.
- `mcap-ros2-support` in `pyproject.toml`/`uv.lock` may still support data
  interchange; remove it only after confirming no recording/export workflow
  needs it and regenerating the lockfile.
- Keep `config/transforms_pzs100_initial_pose_1.yaml`: active Python tools,
  tests, and a notebook consume it.

## Local generated-data cleanup (not a Git deletion)

These are ignored or untracked local files.  Removing them frees local disk but
does not remove repository history and should not be committed.

| Local path or pattern | Apparent size / count | Treatment |
| --- | ---: | --- |
| `.venv/` | 6.9 GB | Recreate with the documented environment setup when needed. |
| `build/` | 23 MB | Generated build output; recreate via the build command. |
| `.cache/` | 11 MB | Local cache (including ccache); safe only if no active build relies on it. |
| `.pytest_cache/`, `.ruff_cache/` | about 100 KB | Test/lint caches; size varies after test runs. |
| `snapshot.rrd` | 1.5 MB | Local visualization recording; keep only if it is an intended artifact. |
| `registration_snapshot.rrd` | 143 B | Local visualization recording/stub; normally discard. |
| `data/registration_snapshots/**/annotations.yaml.backup-*` | 54 files, about 57 KB | Remove only after verifying canonical annotations and commit `48f10e3`; use version-control removal for the 5 tracked files and local cleanup for the other 49. |
| `.openclaude/worktrees/` | about 5.1 GB | **Do not remove in this pass.** Confirm each worktree's owner and Git status first. |

## Explicit keep list

Keep the active Python prototype and its proof/evaluation material:

- `README.md`, `pyproject.toml`, `uv.lock`, `Makefile`, `python/`, `tests/`,
  and the active tools in `tools/`.
- `.devcontainer/`, `docker/`, and `docker-compose.yml`, which implement the
  active Python/Open3D development workflow.
- `data/registration_snapshots/` canonical content: the 23
  `annotations.yaml` files plus their clouds, images, masks, metadata,
  registrations, transforms, and exclusions manifest.
- `docs/current-detector-pipeline.md`, `docs/annotation-tool.md`, and any
  retained, updated environment document.
- `assets/templates/`, `assets/wall_plans.yaml`, `config/detector_eval.yaml`,
  `config/scenes/`, `config/snapshot_plan_priors.yaml`,
  `config/snapshot_plan_topology.yaml`, `config/snapshot_scenarios.yaml`, and
  `config/wall_plans.yaml`.
- `config/calibration/`, `config/transforms_pzs100_initial_pose_1.yaml`, and
  `assets/pzs100/` pending the planned gripper-aware implementation.
- The repo configuration files (`.gitignore`, `.gitattributes`, AGENTS.md) and
  the user-managed `.openclaude/settings.json`.

## Ordered manual pass

1. Preserve annotation recovery: confirm `48f10e3`, the 23 canonical files,
   and the clean canonical-file status.  Do not mix backups with canonical
   data.
2. Resolve the immediate setup-note candidate, transferring any still-open task
   to an issue first. Preserve or relocate the snapshot review ledger before
   considering it separately.
3. Decide the fate of the three overlapping plan/verification documents and
   update README and cross-links in the same commit.
4. Make a single yes/no decision on ROS/Gazebo provenance. If no, archive and
   remove the approved bundle atomically and update README references without
   removing the active Docker/devcontainer environment.
5. Commit tracked deletions separately from local ignored cleanup.  Do not
   include `docs/.obsidian/workspace.json` unless its owner specifically asks.
6. Only after the Git change is reviewed, clean local generated output and the
   untracked backup files.  Leave `.openclaude/worktrees/` for their owners.

## Safe verification commands

Run these non-mutating inspection checks before and after each manual batch:

```bash
git status --short
git show --stat --oneline 48f10e3
git ls-files 'data/registration_snapshots/**/annotations.yaml' | wc -l
git status --short -- 'data/registration_snapshots/**/annotations.yaml'
find data/registration_snapshots -type f -name 'annotations.yaml.backup-*' | wc -l
git ls-files 'data/registration_snapshots/**/annotations.yaml.backup-*' | wc -l
git ls-files --others --exclude-standard \
  'data/registration_snapshots/**/annotations.yaml.backup-*' | wc -l
git worktree list --porcelain
rg -n --glob '*.md' \
  'Snapshot Annotation|detector-plan|implementation-plan|verify-placed' .
rg -n 'launch_context|urdf/|assets/environment|gazebo|ros2|docker-compose' . \
  --glob '!data/**' --glob '!.git/**' --glob '!.venv/**' \
  --glob '!.openclaude/worktrees/**'
```

After reviewing the Git diff, run the mutating validation step separately:

```bash
make check
```

`make check` invokes `uv sync` and writes or recreates environment and cache
state; it is not a read-only inspection command.

Expected annotation counts before backup cleanup are **23 canonical**, **54
backups**, **5 tracked backups**, and **49 untracked backups**.  Treat any
other result as a stop-and-investigate condition.
