"""Joint cache for synchronized event-cloud and gCSTR inputs."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict

import numpy as np
import torch

from gcstr_dataset import (
    DailyDVS200GCSTRDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from gcstr_fusion_dataset import sample_event_cloud
from gcstr_augmentation import augment_events
from gcstr_representations import build_gcstr_views


def stable_seed(text):
    return int.from_bytes(
        hashlib.sha256(text.encode()).digest()[:8], "little"
    )


class CachedFusionDataset(DailyDVS200GCSTRDataset):
    def __init__(
        self, manifest_path, split, *,
        point_count=8192, seed=0, **kwargs
    ):
        if kwargs.get("augmentation", "none") not in ("none", "spatial"):
            raise ValueError("Only none/spatial augmentation is supported")
        if point_count < 1:
            raise ValueError("point_count must be positive")

        super().__init__(
            manifest_path, split,
            views=("cstr3", "xt_p", "ty_p"), **kwargs
        )
        self.point_count = point_count
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        # Training loader must use persistent_workers=False so workers
        # receive the updated epoch when a new iterator is created.
        self.epoch = int(epoch)

    def joint_cache_path(self, row, source, variant):
        if self.cache_dir is None:
            return None

        stat = source.stat()
        metadata = {
            "version": "joint_events_views_v1",
            "source": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "config": asdict(self.config),
            "views": self.views,
            "sample_id": row["sample_id"],
            "augmentation": self.augmentation,
            "augmentation_seed": self.augmentation_seed,
            "variant": variant,
        }
        key = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        return (
            self.cache_dir / "joint_events_views_v1"
            / self.split / f"{key}.npz"
        )

    @staticmethod
    def save_joint_cache(path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".npz", delete=False
            ) as handle:
                temporary = handle.name
                np.savez(handle, **arrays)
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def __getitem__(self, index):
        row = self.rows[index]
        epoch = self.epoch if self.split == "train" else 0
        rng = np.random.default_rng(stable_seed(
            f"{self.seed}:{epoch}:{row['sample_id']}"
        ))

        variant = None
        if self.augmentation != "none":
            variant = int(rng.integers(self.augmentation_count))

        source = self.dataset_root / row["npz_relpath"]
        cache = self.joint_cache_path(row, source, variant)

        if cache is not None and cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                t, x, y, p = (
                    data[key] for key in ("t", "x", "y", "p")
                )
                images = {view: data[view] for view in self.views}
        else:
            with np.load(source, allow_pickle=False) as data:
                t, x, y, p = (
                    data[key] for key in ("t", "x", "y", "p")
                )

            if variant is not None:
                augmentation_rng = np.random.default_rng(stable_seed(
                    f"{self.augmentation_seed}:"
                    f"{row['sample_id']}:{variant}"
                ))
                t, x, y, p = augment_events(
                    t, x, y, p, mode=self.augmentation,
                    rng=augmentation_rng, config=self.config,
                )

            images = build_gcstr_views(
                t, x, y, p, views=self.views, config=self.config
            )
            if cache is not None:
                self.save_joint_cache(
                    cache, dict(t=t, x=x, y=y, p=p, **images)
                )

        sample = {
            "sample_id": row["sample_id"],
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "events": sample_event_cloud(
                t, x, y, p, self.config, self.point_count, rng
            ),
        }
        for view, image in images.items():
            tensor = torch.from_numpy(image.astype(np.float32))
            if self.imagenet_normalize:
                tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            sample[view] = tensor
        return sample