#!/usr/bin/env python3
"""Generate config/annotated_snapshots_seibersdorf.yaml from a snapshots directory.

That file's own header claims its selection is "derived, not hand-listed" and
says "Regenerate rather than edit by hand" -- but until this script, nothing
regenerated it. An operator exporting a fresh bag (tools/export_reference_bag.py
+ tools/annotate_snapshot.py) had no way to turn the resulting annotated
snapshot tree back into the manifest the eval tools consume.

Qualification rule (matches the committed file's header exactly, verified
against data/seibersdorf_dataset before writing this): a snapshot directory
qualifies when its annotations.yaml has annotation_complete: true AND at
least one evaluation-target block, i.e. review_status == 'approved' and no
ignore_reason (blockpose.annotations.AnnotationTarget.is_evaluation_target --
the same predicate the eval tools use to build ground truth). Checked
independently against a `.backup-*` sibling file (written only by
tools/annotate_snapshot.py's save path) as a second signal: on the committed
Seibersdorf tree both signals agree on exactly the same 25 directories, which
is what the committed header claims. Qualification here is decided by the
annotation_complete+approved-block signal alone; the backup signal is
reported (via a stderr note, not a failure) when it disagrees, since a
backup written mid-review of a not-yet-approved snapshot is expected, not a
bug.

Scene grouping: the capture directory name is `<scene>__<ros-time-stamp>`;
scene is everything before the first '__'. This reproduces
config/rgb_ablation_splits_seibersdorf.yaml's 24 `groups` keys exactly on the
committed data -- asserted, not asserted-in-prose, by
tests/python/test_build_snapshot_manifest.py::test_scene_labels_match_rgb_ablation_splits.

The `dataset:` field is emitted repo-relative when `root` lies inside the
repository, because tools/audit_annotation_poses.py resolves it as
`REPO_ROOT / spec["dataset"]`; an absolute path there would bake this
checkout's location into a committed config and break every other clone.
Override with --dataset-label for a root outside the repo.

scenes_without_annotation.covered_by_scene / rgb_similarity_to_covering_scene,
deliberately NOT emitted here: the committed file records those as "measured
RGB similarity ... of a representative frame" (mean-normalised 16x-downsampled
correlation). Reproducing that requires (a) a fixed choice of "representative
frame" per scene and (b) a similarity metric, neither of which is specified
precisely enough in the committed file to regenerate byte-for-byte, and a
16x-box-average grayscale correlation of each scene's first frame against
every annotated scene's first frame, tried while writing this tool, picked
the wrong covering scene for 3 of the 5 unannotated scenes (chose
pickup_and_stack_rotated over misaligned_stack for pre_stack/stacked/stacking)
and reproduced only 2 of 5 similarity scores closely. Inventing a
covered_by_scene assignment this script cannot actually justify would be
worse than omitting the field, so scenes_without_annotation here carries only
snapshots_in_scene -- a purely mechanical count -- and the fields the
committed file adds by hand are left for a human (or a follow-up tool that
pins down "representative frame" and the exact similarity transform) to add
back.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotations import SnapshotAnnotations, load_annotations  # noqa: E402

APPROXIMATE_POSE_PROVENANCE = "rgb_mask_pca_fallback"


def dataset_label_for(root: Path) -> str:
    """Repo-relative label when `root` is inside the repo, else the raw path.

    tools/audit_annotation_poses.py resolves this field as
    `REPO_ROOT / spec["dataset"]`, and the committed manifest carries
    `data/seibersdorf_dataset`. Emitting `/workspace/data/seibersdorf_dataset`
    instead would still resolve on the machine that generated it (pathlib's
    `/` lets an absolute right-hand side win) and break on every other clone,
    which is the failure this repo is least able to notice.
    """
    resolved = root.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def scene_of(snapshot_name: str) -> str:
    """Scene label = capture directory prefix before the trailing '__<stamp>'.

    Matches config/rgb_ablation_splits_seibersdorf.yaml's grouping so this
    composes with the existing leave-one-scenario-out folds instead of
    defining a second, competing grouping (same rule the committed manifest
    documents under `grouping`).
    """
    return snapshot_name.split("__", 1)[0]


def has_backup(snapshot_dir: Path) -> bool:
    """A tool-written backup exists once annotate_snapshot.py has re-saved this file.

    save_annotations_atomic (blockpose/annotation_tool.py) writes
    `annotations.yaml.backup-<UTC stamp>` on every save *of a manifest that
    already exists* -- so it is one save behind: a snapshot initialised by
    tools/init_snapshot_annotations.py and never re-saved has none. That is
    exactly why this is a cross check and not the qualification rule.
    """
    return any(snapshot_dir.glob("annotations.yaml.backup-*"))


def qualifies(annotations: SnapshotAnnotations) -> bool:
    """annotation_complete AND at least one approved, non-ignored block.

    Mirrors AnnotationTarget.is_evaluation_target, the same predicate the
    eval tools use to build ground truth -- this manifest exists to list
    snapshots that contribute ground truth, so it must use the same rule.
    """
    return annotations.annotation_complete and any(
        target.is_evaluation_target for target in annotations.targets
    )


def build_manifest(
    root: Path, *, dataset_label: str | None = None, warn: Any = None
) -> dict[str, Any]:
    """Scan `root` and build the manifest dict in the committed file's shape.

    Qualification is decided solely by annotation_complete + approved-block
    (`qualifies`). The tool-written-backup signal is checked only as a cross
    check, reported through `warn` (defaults to printing to stderr) rather
    than failing the build: annotate_snapshot.py writes a backup on every
    save, including saves of incomplete or all-pending manifests, so a
    backup existing without the snapshot qualifying is expected during
    active review, not necessarily a bug. On the committed Seibersdorf tree
    the two signals happen to agree on exactly the same 25 directories
    (verified separately; see this module's docstring and
    tests/python/test_build_snapshot_manifest.py) -- that agreement is what
    let the committed file's header claim the classification is
    unambiguous, but this generator does not assume it holds on every tree.
    """
    if warn is None:
        warn = lambda message: print(message, file=sys.stderr)  # noqa: E731
    if not root.is_dir():
        raise NotADirectoryError(f"snapshot root does not exist: {root}")

    snapshot_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    scene_all_snapshots: dict[str, list[str]] = defaultdict(list)
    scene_annotated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualifying_by_signal_a: set[str] = set()
    qualifying_by_signal_b: set[str] = set()

    total_approved_blocks = 0
    total_approximate_poses = 0

    for snapshot_dir in snapshot_dirs:
        name = snapshot_dir.name
        scene_all_snapshots[scene_of(name)].append(name)

        manifest_path = snapshot_dir / "annotations.yaml"
        if not manifest_path.is_file():
            continue

        annotations = load_annotations(manifest_path)
        if has_backup(snapshot_dir):
            qualifying_by_signal_b.add(name)
        if not qualifies(annotations):
            continue
        qualifying_by_signal_a.add(name)

        approved = [t for t in annotations.targets if t.is_evaluation_target]
        approximate = [t for t in approved if t.provenance == APPROXIMATE_POSE_PROVENANCE]
        visibility_counts = Counter(t.visibility for t in approved)

        total_approved_blocks += len(approved)
        total_approximate_poses += len(approximate)

        scene_annotated[scene_of(name)].append(
            {
                "name": name,
                "blocks": len(approved),
                "approximate_poses": len(approximate),
                "visibility": dict(sorted(visibility_counts.items())),
            }
        )

    disagreement = qualifying_by_signal_a.symmetric_difference(qualifying_by_signal_b)
    if disagreement:
        warn(
            "note: annotation_complete+approved-block signal and the "
            "tool-written-backup signal disagree on "
            f"{len(disagreement)} director{'y' if len(disagreement) == 1 else 'ies'}: "
            f"{sorted(disagreement)}. Qualification here follows "
            "annotation_complete+approved-block only; a disagreement usually "
            "means a snapshot mid-review (saved, not yet approved), but is "
            "worth a look if unexpected."
        )

    scenes: dict[str, Any] = {}
    scenes_without_annotation: dict[str, Any] = {}
    for scene in sorted(scene_all_snapshots):
        snapshots_in_scene = len(scene_all_snapshots[scene])
        annotated = scene_annotated.get(scene)
        if annotated:
            scenes[scene] = {
                "snapshots_in_scene": snapshots_in_scene,
                "annotated": sorted(annotated, key=lambda entry: entry["name"]),
            }
        else:
            # covered_by_scene / rgb_similarity_to_covering_scene deliberately
            # omitted -- see module docstring.
            scenes_without_annotation[scene] = {"snapshots_in_scene": snapshots_in_scene}

    totals = {
        "snapshots_in_dataset": len(snapshot_dirs),
        "annotated_snapshots": len(qualifying_by_signal_a),
        "approved_blocks": total_approved_blocks,
        "approximate_poses": total_approximate_poses,
        "scenes_total": len(scene_all_snapshots),
        "scenes_annotated": len(scenes),
        "scenes_without_annotation": len(scenes_without_annotation),
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset_label or dataset_label_for(root),
        "selection_criteria": (
            "one annotation per DISTINCT scene: annotation_complete: true AND at "
            "least one approved block. Consecutive captures within a scene are "
            "near-duplicate frames and are deliberately left unannotated."
        ),
        "grouping": (
            "scene = capture directory prefix before '__'; matches "
            "config/rgb_ablation_splits_seibersdorf.yaml groups"
        ),
        "totals": totals,
        "scenes": scenes,
    }
    if scenes_without_annotation:
        manifest["scenes_without_annotation"] = scenes_without_annotation
    return manifest


def render(manifest: dict[str, Any]) -> str:
    header = (
        "# Generated by tools/build_snapshot_manifest.py -- do not hand-edit.\n"
        "# scenes_without_annotation here carries only snapshots_in_scene: this\n"
        "# tool does not compute covered_by_scene / rgb_similarity_to_covering_scene\n"
        "# (see the tool's module docstring for why).\n"
    )
    return header + yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "data" / "seibersdorf_dataset",
        help="snapshot directory to scan (default: data/seibersdorf_dataset)",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default=None,
        help="value for the 'dataset:' field (default: root, repo-relative when inside the repo)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write here instead of stdout (never overwrites)"
    )
    args = parser.parse_args()

    manifest = build_manifest(args.root, dataset_label=args.dataset_label)
    rendered = render(manifest)

    if args.output is None:
        sys.stdout.write(rendered)
        return 0
    if args.output.exists():
        parser.error(f"refusing to overwrite existing file: {args.output}")
    args.output.write_text(rendered)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
