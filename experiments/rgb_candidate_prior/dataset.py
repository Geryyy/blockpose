"""Load exported candidate records and split them by recording session.

The split is by ``split_group`` (session / rosbag / day) and **never** by frame,
crop, or candidate: nearby captures, crops, and candidates from one scene are
correlated, so mixing them across train/test leaks (design doc: "Split by
recording session/rosbag/day, never by frame, crop, or candidate").
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    """One exported candidate record with resolved absolute image paths."""

    sample_id: str
    split_group: str
    label: str
    provenance: str
    rgb_path: Path
    conditioning_path: Path
    raw: dict

    @property
    def block_id(self) -> str:
        """Everything but the trailing candidate token, e.g. ``bagA/snap/block_0``."""
        return self.sample_id.rsplit("/", 1)[0]


@dataclass(frozen=True)
class CandidateDataset:
    """Exported samples plus their export root, grouped for session-level splits."""

    root: Path
    samples: tuple[Sample, ...]

    @classmethod
    def load(cls, export_root: str | Path) -> CandidateDataset:
        root = Path(export_root)
        samples: list[Sample] = []
        for line in (root / "samples.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(
                Sample(
                    sample_id=record["sample_id"],
                    split_group=record["split_group"],
                    label=record["label"],
                    provenance=record["provenance"],
                    rgb_path=root / record["rgb"],
                    conditioning_path=root / record["conditioning"],
                    raw=record,
                )
            )
        return cls(root=root, samples=tuple(samples))

    def groups(self) -> dict[str, list[Sample]]:
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.split_group].append(sample)
        return dict(grouped)

    def candidate_sets(self) -> dict[str, list[Sample]]:
        """Samples grouped by labelled block (the V1 ranking unit)."""
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.block_id].append(sample)
        return dict(grouped)

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for sample in self.samples:
            counts[sample.label] += 1
        return dict(counts)


def split_by_session(
    dataset: CandidateDataset, holdout_groups: set[str]
) -> tuple[list[Sample], list[Sample]]:
    """Partition samples into ``(train, holdout)`` by whole ``split_group``.

    ``holdout_groups`` must name groups that exist; every sample of a held-out
    group goes to holdout so no correlated crop leaks into train.
    """
    available = set(dataset.groups())
    unknown = holdout_groups - available
    if unknown:
        raise ValueError(f"unknown holdout groups: {sorted(unknown)} (have {sorted(available)})")
    train, holdout = [], []
    for sample in dataset.samples:
        (holdout if sample.split_group in holdout_groups else train).append(sample)
    return train, holdout
