"""Load the region-prior JSONL manifest and split it by recording session/fold.

One record per *snapshot* (see ``export.py``), unlike
``experiments.rgb_candidate_prior.dataset``'s one record per *candidate*. The
split is still by ``split_group`` (session/rosbag/day) and never by a finer
unit, for the same reason: nearby captures are correlated, so mixing them
across train/held-out leaks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegionSample:
    """One exported snapshot's union foreground-region target."""

    sample_id: str
    split_group: str
    image_path: Path
    mask_path: Path
    block_count: int
    raw: dict


@dataclass(frozen=True)
class RegionDataset:
    """Exported per-snapshot samples plus their export root, grouped for splits."""

    root: Path
    samples: tuple[RegionSample, ...]

    @classmethod
    def load(cls, export_root: str | Path) -> RegionDataset:
        root = Path(export_root)
        samples: list[RegionSample] = []
        for line in (root / "samples.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(
                RegionSample(
                    sample_id=record["sample_id"],
                    split_group=record["split_group"],
                    image_path=root / record["image"],
                    mask_path=root / record["mask"],
                    block_count=int(record["block_count"]),
                    raw=record,
                )
            )
        return cls(root=root, samples=tuple(samples))

    def groups(self) -> dict[str, list[RegionSample]]:
        grouped: dict[str, list[RegionSample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.split_group].append(sample)
        return dict(grouped)


def split_by_session(
    dataset: RegionDataset, holdout_groups: set[str]
) -> tuple[list[RegionSample], list[RegionSample]]:
    """Partition samples into ``(train, holdout)`` by whole ``split_group``.

    ``holdout_groups`` must name groups that exist; every sample of a
    held-out group goes to holdout so no correlated snapshot leaks into
    train.
    """
    available = set(dataset.groups())
    unknown = holdout_groups - available
    if unknown:
        raise ValueError(f"unknown holdout groups: {sorted(unknown)} (have {sorted(available)})")
    train, holdout = [], []
    for sample in dataset.samples:
        (holdout if sample.split_group in holdout_groups else train).append(sample)
    return train, holdout


def development_fold_of_snapshot(manifest: dict) -> dict[str, int]:
    """Map snapshot name -> development fold id from a loaded split manifest.

    Only ``development``-split groups are included; ``final`` snapshots
    (``selection_eligible: false``) are never returned, so callers cannot
    accidentally train or select on them. (Deliberately re-implemented here,
    not imported from ``experiments.rgb_candidate_prior.model`` -- see this
    package's docstring on why the two pipelines stay separate -- but the
    logic and the fold-safety rule below are identical.)
    """
    mapping: dict[str, int] = {}
    for group in manifest["groups"].values():
        if group["split"] != "development":
            continue
        for snapshot in group["snapshots"]:
            mapping[snapshot] = group["fold"]
    return mapping


def fold_of_group(samples: list[RegionSample], snapshot_fold: dict[str, int]) -> int | None:
    """Development fold id for one ``split_group``, resolved via true snapshot names.

    ``split_group`` may be a session id assigned via ``--sessions``, not a raw
    snapshot name, so this always resolves through each sample's recorded
    ``source.snapshot`` (here identical to ``sample_id``, but read from the
    same field the exporter writes rather than assumed) instead of
    string-matching ``split_group`` against the manifest -- the same rule
    ``experiments.rgb_candidate_prior.model.fold_of_group`` enforces, since a
    coarser group string spanning two folds was a real leakage bug found and
    fixed for the verifier earlier. Returns ``None`` if no member snapshot is
    in a development fold.
    """
    folds = {
        snapshot_fold[snapshot]
        for sample in samples
        if (snapshot := sample.raw.get("source", {}).get("snapshot")) in snapshot_fold
    }
    if not folds:
        return None
    if len(folds) > 1:
        raise ValueError(
            f"split_group spans multiple development folds ({sorted(folds)}); "
            "held-out splitting requires exactly one fold per split_group"
        )
    return next(iter(folds))


def select_development_groups(
    dataset: RegionDataset, snapshot_fold: dict[str, int]
) -> tuple[RegionDataset, dict[str, int]]:
    """Restrict a dataset to samples whose ``split_group`` resolves to a development fold.

    Returns the restricted dataset plus the resolved ``{split_group: fold}``
    map, so callers can build per-fold held-out splits with
    ``split_by_session``. Each sample is also checked individually against its
    own recorded ``source.snapshot``, not just its group's resolved fold, so a
    ``final`` snapshot sharing a session bucket with a ``development`` one is
    never silently kept.
    """
    group_fold: dict[str, int] = {}
    for group, samples in dataset.groups().items():
        fold = fold_of_group(samples, snapshot_fold)
        if fold is not None:
            group_fold[group] = fold
    kept = tuple(
        sample
        for sample in dataset.samples
        if sample.split_group in group_fold
        and sample.raw.get("source", {}).get("snapshot") in snapshot_fold
    )
    return RegionDataset(root=dataset.root, samples=kept), group_fold
