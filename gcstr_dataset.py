"""PyTorch dataset for cached gCSTR views of DailyDVS-200."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from create_gcstr_splits import DEFAULT_ROOT
from gcstr_augmentation import AUGMENTATIONS, augment_events
from gcstr_representations import GCSTRConfig, VIEW_NAMES, build_gcstr_views


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
CACHE_VERSION = "official_code_v1"


class DailyDVS200GCSTRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        views: Iterable[str] = ("cstr3",),
        dataset_root: str | Path = DEFAULT_ROOT,
        config: GCSTRConfig = GCSTRConfig(),
        cache_dir: str | Path | None = None,
        imagenet_normalize: bool = True,
        augmentation: str = "none",
        augmentation_count: int = 5,
        augmentation_seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.views = tuple(views)
        self.config = config
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.imagenet_normalize = imagenet_normalize
        if augmentation not in AUGMENTATIONS:
            raise ValueError(f"unknown augmentation {augmentation!r}")
        if augmentation != "none" and split != "train":
            raise ValueError("augmentation is only allowed for the train split")
        if augmentation_count < 1:
            raise ValueError("augmentation_count must be positive")
        self.augmentation = augmentation
        self.augmentation_count = augmentation_count
        self.augmentation_seed = augmentation_seed
        unknown = set(self.views) - set(VIEW_NAMES)
        if unknown:
            raise ValueError(f"unknown views: {sorted(unknown)}")
        if not self.views:
            raise ValueError("at least one view is required")

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"sample_id", "split", "npz_relpath", "label_id"}
        fields = set(rows[0]) if rows else set()
        missing = required - fields
        if missing:
            raise ValueError(f"manifest is missing fields: {sorted(missing)}")
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"manifest contains no samples for split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def _cache_path(self, row: dict[str, str], view: str, variant: int | None = None) -> Path | None:
        if self.cache_dir is None:
            return None
        preset = f"{CACHE_VERSION}_s{self.config.image_size}_t{self.config.time_bins}"
        if variant is not None:
            preset += f"_aug_{self.augmentation}_n{self.augmentation_count}_seed{self.augmentation_seed}"
            return self.cache_dir / preset / view / self.split / f"{row['sample_id']}_v{variant}.npy"
        return self.cache_dir / preset / view / self.split / f"{row['sample_id']}.npy"

    @staticmethod
    def _atomic_save(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
                temporary_name = handle.name
                np.save(handle, array, allow_pickle=False)
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        variant = None
        if self.augmentation != "none":
            variant = int(np.random.randint(self.augmentation_count))
        images: dict[str, np.ndarray] = {}
        missing_views: list[str] = []
        for view in self.views:
            cache_path = self._cache_path(row, view, variant)
            if cache_path is not None and cache_path.is_file():
                images[view] = np.load(cache_path, allow_pickle=False)
            else:
                missing_views.append(view)

        if missing_views:
            npz_path = self.dataset_root / row["npz_relpath"]
            with np.load(npz_path, allow_pickle=False) as data:
                t, x, y, p = data["t"], data["x"], data["y"], data["p"]
                if variant is not None:
                    identity = f"{self.augmentation_seed}:{row['sample_id']}:{variant}"
                    variant_seed = int.from_bytes(
                        hashlib.sha256(identity.encode()).digest()[:8], "little"
                    )
                    t, x, y, p = augment_events(
                        t, x, y, p, mode=self.augmentation,
                        rng=np.random.default_rng(variant_seed), config=self.config,
                    )
                generated = build_gcstr_views(
                    t, x, y, p,
                    views=tuple(missing_views), config=self.config,
                )
            for view, image in generated.items():
                images[view] = image
                cache_path = self._cache_path(row, view, variant)
                if cache_path is not None:
                    self._atomic_save(cache_path, image)

        sample: dict[str, object] = {
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id": row["sample_id"],
        }
        for view in self.views:
            tensor = torch.from_numpy(images[view].astype(np.float32, copy=False))
            if self.imagenet_normalize:
                tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            sample[view] = tensor
        return sample
