"""PyTorch dataset for the aligned DailyDVS-200 representations."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from daily_dvs200_loader import load_events
from event_representations import build_representations


DEFAULT_DATASET_ROOT = Path(
    "/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200"
)
REQUIRED_MANIFEST_FIELDS = {"path", "class_index", "person_id"}


def class_balanced_limit(
    records: list[dict[str, str]], max_samples: int
) -> list[dict[str, str]]:
    """Select a deterministic, approximately equal number from every class."""
    if max_samples >= len(records):
        return records
    grouped: dict[int, list[dict[str, str]]] = {}
    for record in records:
        grouped.setdefault(int(record["class_index"]), []).append(record)
    classes = sorted(grouped)
    quotas = {class_index: 0 for class_index in classes}
    if max_samples < len(classes):
        positions = np.linspace(0, len(classes) - 1, max_samples, dtype=np.int64)
        for position in positions:
            quotas[classes[int(position)]] = 1
    else:
        base, remainder = divmod(max_samples, len(classes))
        for rank, class_index in enumerate(classes):
            quotas[class_index] = min(base + (rank < remainder), len(grouped[class_index]))
        unassigned = max_samples - sum(quotas.values())
        while unassigned:
            changed = False
            for class_index in classes:
                if quotas[class_index] < len(grouped[class_index]):
                    quotas[class_index] += 1
                    unassigned -= 1
                    changed = True
                    if unassigned == 0:
                        break
            if not changed:
                break

    selected = []
    for class_index in classes:
        candidates = grouped[class_index]
        quota = quotas[class_index]
        if quota:
            positions = np.linspace(0, len(candidates) - 1, quota, dtype=np.int64)
            selected.extend(candidates[int(position)] for position in positions)
    return selected


class DailyDVS200Dataset(Dataset):
    """Decode one AEDAT4 recording and return three aligned representations."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        training: bool = True,
        seed: int | None = 0,
        num_bins: int = 8,
        points_per_bin: int = 256,
        height: int = 240,
        width: int = 320,
        person_ids: Iterable[int] | None = None,
        split: str | None = None,
        max_samples: int | None = None,
        deterministic_points: bool | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.training = training
        self.seed = seed
        self.num_bins = num_bins
        self.points_per_bin = points_per_bin
        self.height = height
        self.width = width
        self.epoch = 0
        self.deterministic_points = (
            not training if deterministic_points is None else deterministic_points
        )
        if cache_in_memory and not self.deterministic_points:
            raise ValueError("memory caching requires deterministic point sampling")
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, dict[str, object]] = {}

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = REQUIRED_MANIFEST_FIELDS - fields
            if missing:
                raise ValueError(f"manifest is missing fields: {sorted(missing)}")
            records = list(reader)

        if split is not None:
            if "split" not in fields:
                raise ValueError("a split was requested but the manifest has no split column")
            records = [record for record in records if record["split"] == split]
        if person_ids is not None:
            selected_people = {int(person_id) for person_id in person_ids}
            records = [
                record for record in records
                if int(record["person_id"]) in selected_people
            ]
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be at least 1")
            records = class_balanced_limit(records, max_samples)
        if not records:
            raise ValueError("no samples remain after manifest filtering")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic training samples between epochs."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _sample_seed(self, index: int) -> int | None:
        if self.seed is None:
            return None
        return self.seed + self.epoch * len(self.records) + index

    def __getitem__(self, index: int) -> dict[str, object]:
        if index in self._cache:
            return self._cache[index]
        record = self.records[index]
        relative_path = Path(record["path"])
        events = load_events(self.dataset_root / relative_path)
        representations = build_representations(
            events,
            height=self.height,
            width=self.width,
            num_bins=self.num_bins,
            points_per_bin=self.points_per_bin,
            deterministic=self.deterministic_points,
            seed=self._sample_seed(index),
        )

        frame = np.log1p(representations["frame"]).astype(np.float32)
        voxel = np.log1p(representations["voxel"]).astype(np.float32)
        sample = {
            "frame": torch.from_numpy(frame),
            "voxel": torch.from_numpy(voxel),
            "points": torch.from_numpy(representations["points"]),
            "point_mask": torch.from_numpy(representations["point_mask"]),
            "point_bin": torch.from_numpy(representations["point_bin"]),
            "duration_us": torch.tensor(representations["duration_us"], dtype=torch.int64),
            "event_count": torch.tensor(representations["event_count"], dtype=torch.int64),
            "label": torch.tensor(int(record["class_index"]), dtype=torch.int64),
            "person_id": torch.tensor(int(record["person_id"]), dtype=torch.int64),
            "path": relative_path.as_posix(),
        }
        if self.cache_in_memory:
            self._cache[index] = sample
        return sample


def create_dataloader(
    dataset: DailyDVS200Dataset,
    *,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Construct a DataLoader whose default collation fits all returned fields."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires at least one worker")
    if shuffle is None:
        shuffle = dataset.training
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
