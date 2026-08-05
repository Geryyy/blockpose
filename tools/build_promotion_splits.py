#!/usr/bin/env python3
"""Generate a *genuine* promotion-gate split manifest, driven by real sessions.

Context: `config/rgb_ablation_splits_seibersdorf.yaml` groups snapshots by the
scenario label encoded in each capture directory name (e.g. `wall`, `wet`,
`shadow`). That label is a real experimental-condition boundary, but its own
header admits it is not a verified independent recording session --
`grouping_evidence: scenario_label_directory_prefix_not_validated_run_ids` --
and it is marked `selection_eligible: false` / `final_enabled: false`
accordingly. That means every number this repo has measured so far
(`docs/production-readiness-plan.md`, `experiments/dino_proposer_pooled_ablation.md`)
is development-only: nothing has ever actually been held back and scored
once. With 62 approved blocks over 25 snapshots, a +-2 block swing is inside
the noise a promotion decision needs to survive.

This script builds the same manifest *shape* (`schema_version`,
`selection_eligible`, `final_enabled`, `grouping_evidence`, `groups` with
`split`/`fold`/`snapshots`) but the grouping comes from an EXPLICIT
snapshot -> recording-session mapping the operator supplies (see
`config/seibersdorf_sessions.yaml` for the existing example), not from
parsing directory names. A caller-chosen subset of those sessions is reserved
as `split: final` -- the promotion gate -- and the rest become one
`development` fold per session (leave-one-session-out), same as the existing
manifest.

Because the grouping now comes from an operator-asserted session id rather
than a directory-name heuristic, `grouping_evidence` records
`operator_supplied_session_mapping` -- an honest step up from "not validated
run ids", but NOT the same claim as `validated_recording_run_ids`
(`config/rgb_ablation_splits.yaml`'s stricter loader,
`tools/ablate_rgb_proposals.load_split_manifest`, requires exactly that
string before it will honor `final_enabled` -- this tool's output does not
attempt to satisfy that loader, only `experiments/fold_safety.py`, which is
the consumer this track requires).

This script refuses to emit a split that cannot gate anything:
  * an empty reserved-session set (nothing held out at all).
  * a reserved session id absent from the session mapping (typo).
  * a session named as both `development` and `final` -- the exact bug class
    `experiments/fold_safety.py` exists to keep out: its module docstring
    records that a `split_group` spanning both once leaked `final` data into
    training because only the group, not each sample, was checked.
  * a reserved (final) set whose snapshots carry zero approved blocks -- a
    gate nothing can pass or fail is not a gate.
  * no development sessions left once the reserve is removed -- a manifest
    with nothing to develop against is equally unusable, just in the other
    direction.

Usage:
    uv run python tools/build_promotion_splits.py \\
        --sessions config/seibersdorf_sessions.yaml \\
        --reserve 2026_07_07_wall 2026_07_15_shadow \\
        --out config/promotion_splits_seibersdorf.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotations import load_annotations  # noqa: E402

SCHEMA_VERSION = 1
GROUPING_EVIDENCE = "operator_supplied_session_mapping"


def load_session_map(path: Path) -> dict[str, str]:
    """Load a snapshot -> session_id mapping, e.g. ``config/seibersdorf_sessions.yaml``."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"{path}: session mapping must be a non-empty snapshot -> session_id mapping"
        )
    mapping: dict[str, str] = {}
    for snapshot, session in data.items():
        if not isinstance(snapshot, str) or not isinstance(session, str) or not session.strip():
            raise ValueError(
                f"{path}: every entry must be a string snapshot name -> non-empty string "
                f"session id, got {snapshot!r}: {session!r}"
            )
        mapping[snapshot] = session
    return mapping


def count_approved_blocks(dataset_root: Path, snapshots: Iterable[str]) -> dict[str, int]:
    """Approved (ground-truth-eligible) block count per snapshot.

    Reads each snapshot's own ``annotations.yaml`` and applies exactly
    ``AnnotationTarget.is_evaluation_target`` (``review_status == 'approved'``
    and no ``ignore_reason``) -- the same rule every scored report in this
    repo uses for ground truth, so a session's count here means the same
    thing it means in ``tools/eval_dino_proposer_pooled.py`` or
    ``config/annotated_snapshots_seibersdorf.yaml``.
    """
    counts: dict[str, int] = {}
    for snapshot in snapshots:
        annotation_path = dataset_root / snapshot / "annotations.yaml"
        if not annotation_path.is_file():
            raise ValueError(
                f"missing annotations for snapshot {snapshot!r}: {annotation_path} does not exist"
            )
        annotations = load_annotations(annotation_path)
        counts[snapshot] = sum(1 for target in annotations.targets if target.is_evaluation_target)
    return counts


def build_promotion_splits(
    session_map: Mapping[str, str],
    reserve_sessions: Iterable[str],
    approved_blocks: Mapping[str, int],
    *,
    development_sessions: Iterable[str] | None = None,
) -> dict:
    """Partition a session mapping into a promotion-gate split manifest.

    ``reserve_sessions`` names the sessions withheld as the promotion gate
    (``split: final``). By default every other session named in
    ``session_map`` becomes its own ``development`` fold
    (leave-one-session-out, same shape as
    ``config/rgb_ablation_splits_seibersdorf.yaml``); pass
    ``development_sessions`` to restrict development to an explicit subset
    instead (e.g. to drop sessions that belong to neither split).

    Raises ``ValueError`` -- refusing to emit anything -- for every way this
    split could fail to actually gate a promotion decision; see the module
    docstring for the full list.
    """
    reserve = {str(session) for session in reserve_sessions}
    if not reserve:
        raise ValueError(
            "reserve_sessions is empty: a promotion gate needs at least one held-out session"
        )
    all_sessions = {session for session in session_map.values()}
    unknown_reserve = reserve - all_sessions
    if unknown_reserve:
        raise ValueError(
            f"reserved session ids not present in the session mapping: {sorted(unknown_reserve)} "
            f"(known sessions: {sorted(all_sessions)})"
        )

    if development_sessions is None:
        development = all_sessions - reserve
    else:
        development = {str(session) for session in development_sessions}
        unknown_development = development - all_sessions
        if unknown_development:
            raise ValueError(
                "development session ids not present in the session mapping: "
                f"{sorted(unknown_development)}"
            )
        leaked = development & reserve
        if leaked:
            raise ValueError(
                "session(s) named as both development and final (promotion-gate): "
                f"{sorted(leaked)}. A split_group cannot resolve to both -- this is the exact "
                "leak experiments/fold_safety.py's per-sample checks exist to keep out. Remove "
                "the overlapping session(s) from one list."
            )
    if not development:
        raise ValueError(
            "no development sessions remain once the reserved set is removed: "
            f"{sorted(reserve)}; a manifest with nothing to develop against is unusable"
        )

    snapshots_by_session: dict[str, list[str]] = {}
    for snapshot, session in session_map.items():
        if session in reserve or session in development:
            snapshots_by_session.setdefault(session, []).append(snapshot)

    missing_counts = sorted(
        snapshot
        for snapshots in snapshots_by_session.values()
        for snapshot in snapshots
        if snapshot not in approved_blocks
    )
    if missing_counts:
        raise ValueError(f"no approved-block count supplied for snapshots: {missing_counts}")

    final_blocks = sum(
        approved_blocks[snapshot]
        for session in reserve
        for snapshot in snapshots_by_session.get(session, [])
    )
    if final_blocks == 0:
        raise ValueError(
            f"reserved (final) session set {sorted(reserve)} contains zero approved blocks; "
            "a promotion gate that can never be failed or passed is not usable -- reserve "
            "session(s) with at least one approved block"
        )

    groups: dict[str, dict] = {}
    for fold, session in enumerate(sorted(development)):
        groups[session] = {
            "split": "development",
            "fold": fold,
            "snapshots": sorted(snapshots_by_session[session]),
        }
    for session in sorted(reserve):
        groups[session] = {
            "split": "final",
            "snapshots": sorted(snapshots_by_session[session]),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "selection_eligible": True,
        "final_enabled": True,
        "grouping_evidence": GROUPING_EVIDENCE,
        "groups": groups,
    }


def render_manifest(manifest: dict, *, source: Path, dataset_root: Path, reserve: list[str]) -> str:
    """Render ``manifest`` as YAML with a provenance header, matching sibling manifests' style."""
    header = (
        "# Promotion-gate split manifest generated by tools/build_promotion_splits.py from an\n"
        "# operator-supplied session mapping -- NOT inferred from directory-name prefixes. See\n"
        "# config/rgb_ablation_splits_seibersdorf.yaml's grouping_evidence for why that file\n"
        "# cannot honestly gate a promotion decision. Regenerate rather than edit by hand:\n"
        f"#   uv run python tools/build_promotion_splits.py --sessions {source} "
        f"--reserve {' '.join(reserve)} --out <this file>\n"
        "#\n"
        f"# source session mapping: {source}\n"
        f"# dataset root used for approved-block counts: {dataset_root}\n"
        f"# reserved (final / promotion-gate) sessions: {sorted(set(reserve))}\n"
    )
    return header + yaml.safe_dump(manifest, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        required=True,
        help="snapshot -> session_id mapping, e.g. config/seibersdorf_sessions.yaml",
    )
    parser.add_argument(
        "--reserve",
        nargs="+",
        required=True,
        metavar="SESSION_ID",
        help="session id(s) to hold out as the promotion gate (split: final)",
    )
    parser.add_argument(
        "--development-sessions",
        nargs="+",
        default=None,
        metavar="SESSION_ID",
        help="explicit development session allowlist (default: every session not reserved)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "seibersdorf_dataset",
        help="dataset root to read annotations.yaml from, for approved-block counts",
    )
    parser.add_argument("--out", type=Path, required=True, help="output split manifest path")
    args = parser.parse_args()

    session_map = load_session_map(args.sessions)
    approved_blocks = count_approved_blocks(args.dataset_root, session_map.keys())
    manifest = build_promotion_splits(
        session_map,
        args.reserve,
        approved_blocks,
        development_sessions=args.development_sessions,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render_manifest(
            manifest, source=args.sessions, dataset_root=args.dataset_root, reserve=args.reserve
        )
    )

    final_snapshots = [
        snapshot
        for group in manifest["groups"].values()
        if group["split"] == "final"
        for snapshot in group["snapshots"]
    ]
    final_blocks = sum(approved_blocks[snapshot] for snapshot in final_snapshots)
    development_groups = sum(1 for g in manifest["groups"].values() if g["split"] == "development")
    print(
        f"wrote {args.out}: {development_groups} development folds, "
        f"{len(final_snapshots)} final snapshots ({final_blocks} approved blocks) reserved from "
        f"{sorted(set(args.reserve))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
