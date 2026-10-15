"""Official split, event-only input. Never generate gCSTR images or caches."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from create_gcstr_splits import DEFAULT_ROOT, _output_row
from gcstr_augmentation import augment_events
from gcstr_fusion_dataset import sample_event_cloud
from gcstr_representations import GCSTRConfig


def stable_seed(identity):
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")


def official_rows(root):
    result = {}
    for split in ("train", "test"):
        with (Path(root) / "splits" / f"{split}.csv").open(newline="") as handle:
            source = list(csv.DictReader(handle))
        if not source or any(row["split"].lower() != split for row in source):
            raise ValueError(f"Invalid official {split} manifest")
        rows = [_output_row(row, split, split) for row in source]
        if {int(r["label_id"]) for r in rows} != set(range(200)):
            raise ValueError(f"Expected all 200 classes in official {split}")
        for key in ("sample_id", "event_relpath", "npz_relpath"):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate {key} in {split}")
        result[split] = rows
    for key in ("sample_id", "event_relpath", "npz_relpath"):
        if {r[key] for r in result["train"]} & {r[key] for r in result["test"]}:
            raise ValueError(f"Train/Test overlap in {key}")
    return result


class SECNetOnlyDataset(Dataset):
    def __init__(self, rows, *, dataset_root=DEFAULT_ROOT, training=False,
                 seed=0, point_count=8192, augmentation="spatial", augmentation_count=5):
        self.rows, self.dataset_root = rows, Path(dataset_root)
        self.training, self.seed, self.point_count = training, seed, point_count
        self.augmentation, self.augmentation_count = augmentation, augmentation_count
        self.config, self.epoch = GCSTRConfig(), 0
        if not rows or point_count < 1 or augmentation_count < 1:
            raise ValueError("Invalid dataset configuration")
        if training and any(r["source_split"] != "train" for r in rows):
            raise ValueError("Training must not consume Test samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Epoch/sample seeds make restart reproducible independently of workers.
        identity = f"{self.seed}:{row['sample_id']}"
        if self.training:
            identity += f":epoch:{self.epoch}"
        rng = np.random.default_rng(stable_seed(identity))
        with np.load(self.dataset_root / row["npz_relpath"], allow_pickle=False) as data:
            t, x, y, p = (data[k] for k in ("t", "x", "y", "p"))
        if self.training and self.augmentation != "none":
            variant = int(rng.integers(self.augmentation_count))
            variant_seed = stable_seed(f"{self.seed}:{row['sample_id']}:{variant}")
            t, x, y, p = augment_events(t, x, y, p, mode=self.augmentation,
                rng=np.random.default_rng(variant_seed), config=self.config)
        return {"events": sample_event_cloud(t, x, y, p, self.config, self.point_count, rng),
                "label": torch.tensor(int(row["label_id"])), "sample_id": row["sample_id"]}
