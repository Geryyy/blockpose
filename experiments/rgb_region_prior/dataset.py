"""Load the region-prior JSONL manifest: one record per *snapshot* (see
``export.py``), unlike ``experiments.rgb_candidate_prior.dataset``'s one
record per *candidate*.

Session/fold-safety splitting itself (``split_by_session``,
``development_fold_of_snapshot``, ``fold_of_group``,
``select_development_groups``) is shared with that package via
``experiments.fold_safety`` and re-exported here for import-site convenience.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from experiments.fold_safety import development_fold_of_snapshot as development_fold_of_snapshot
from experiments.fold_safety import fold_of_group as fold_of_group
from experiments.fold_safety import select_development_groups as select_development_groups
from experiments.fold_safety import split_by_session as split_by_session  # noqa: F401


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
