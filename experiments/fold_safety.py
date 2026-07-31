"""Session/fold-safety helpers shared by every offline RGB-prior package.

One non-negotiable rule applies everywhere a manifest-driven split
(``config/rgb_ablation_splits*.yaml``) is used: split by whole recording
session, never by a finer unit, and resolve a sample's fold via its own
recorded ``source.snapshot`` -- never by string-matching ``split_group``,
which can be a coarser session id spanning real development/final
boundaries. A ``split_group`` mixing development and final snapshots once
leaked ``final`` data into training because only the group, not each sample,
was checked; every function here enforces the per-sample check so that bug
class can't recur. This module exists so the rule is enforced in exactly one
place instead of once per package.

Callers only need a sample with ``.split_group``/``.raw`` and a dataset with
``.root``/``.samples``/``.groups()`` -- see ``experiments.rgb_candidate_prior.dataset.Sample``/
``CandidateDataset`` and ``experiments.rgb_region_prior.dataset.RegionSample``/``RegionDataset``.
"""

from __future__ import annotations

from typing import Any


def split_by_session(dataset: Any, holdout_groups: set[str]) -> tuple[list[Any], list[Any]]:
    """Partition samples into ``(train, holdout)`` by whole ``split_group``.

    ``holdout_groups`` must name groups that exist; every sample of a
    held-out group goes to holdout so no correlated crop/snapshot leaks into
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
    accidentally train or select on them.
    """
    mapping: dict[str, int] = {}
    for group in manifest["groups"].values():
        if group["split"] != "development":
            continue
        for snapshot in group["snapshots"]:
            mapping[snapshot] = group["fold"]
    return mapping


def fold_of_group(samples: list[Any], snapshot_fold: dict[str, int]) -> int | None:
    """Development fold id for one ``split_group``, resolved via true snapshot names.

    ``split_group`` may be a session id, not a raw snapshot name, so this
    always resolves through each sample's recorded ``source.snapshot``
    rather than string-matching ``split_group`` against the manifest.
    Returns ``None`` if no member snapshot is in a development fold (i.e.
    the group is entirely ``final``/unlisted data).
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
    dataset: Any, snapshot_fold: dict[str, int]
) -> tuple[Any, dict[str, int]]:
    """Restrict a dataset to samples whose ``split_group`` resolves to a development fold.

    Returns the restricted dataset plus the resolved ``{split_group: fold}``
    map, so callers can build per-fold held-out splits with
    ``split_by_session``.

    A ``split_group`` (session) can mix development and ``final`` snapshots
    if they happen to land in the same session bucket; keeping every sample
    of a group once *any* member resolves to a fold would leak ``final``
    (``selection_eligible: false``) candidates into training. Each sample is
    therefore also checked individually against its own recorded
    ``source.snapshot``, not just its group's resolved fold.
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
    return type(dataset)(root=dataset.root, samples=kept), group_fold
