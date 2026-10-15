"""Shared event-level preprocessing for point and image branches.

First version builds all views jointly, without reusing old independently
selected augmentation caches. No zero padding; empty streams fail explicitly.
"""
import hashlib
import numpy as np
import torch

from gcstr_dataset import DailyDVS200GCSTRDataset, IMAGENET_MEAN, IMAGENET_STD
from gcstr_augmentation import augment_events
from gcstr_representations import build_gcstr_views, _validate_events


def sample_event_cloud(t, x, y, p, config, count, rng):
    t, x, y, p = _validate_events(t, x, y, p, config)
    if count < 1:
        raise ValueError("point count must be positive")
    indices = np.sort(rng.choice(len(t), count, replace=len(t) < count))
    # Define time range BEFORE point sampling, shared with the full projection.
    time = t.astype(np.float64) - float(t[0])
    time /= max(float(time[-1]), 1e-6)
    cloud = np.stack((x[indices] / config.sensor_width,
                      y[indices] / config.sensor_height, time[indices],
                      p[indices].astype(np.float32) * 2 - 1))
    return torch.from_numpy(cloud.astype(np.float32))


class DailyDVS200FusionDataset(DailyDVS200GCSTRDataset):
    def __init__(self, manifest_path, split, *, point_count=8192, seed=0, **kwargs):
        if kwargs.get("cache_dir") is not None:
            raise ValueError("joint fusion cache not implemented; omit cache_dir")
        if kwargs.get("augmentation", "none") not in ("none", "spatial"):
            raise ValueError("first fusion version supports only none/spatial augmentation")
        super().__init__(manifest_path, split, views=("cstr3", "xt_p", "ty_p"), **kwargs)
        self.point_count, self.seed = point_count, seed
        if point_count < 1:
            raise ValueError("point_count must be positive")

    def __getitem__(self, index):
        row = self.rows[index]
        identity = f"{self.seed}:{row['sample_id']}"
        fixed_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
        # np.random is worker-seedable by PyTorch DataLoader / worker_init_fn.
        rng = np.random.default_rng(int(np.random.randint(2**32)) if self.split == "train" else fixed_seed)
        with np.load(self.dataset_root / row["npz_relpath"], allow_pickle=False) as data:
            t, x, y, p = (data[k] for k in ("t", "x", "y", "p"))
        if self.augmentation != "none":
            variant = int(rng.integers(self.augmentation_count))
            identity = f"{self.augmentation_seed}:{row['sample_id']}:{variant}"
            variant_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
            t, x, y, p = augment_events(t, x, y, p, mode=self.augmentation,
                                       rng=np.random.default_rng(variant_seed), config=self.config)
        images = build_gcstr_views(t, x, y, p, views=self.views, config=self.config)
        result = {"sample_id": row["sample_id"], "label": torch.tensor(int(row["label_id"])),
                  "events": sample_event_cloud(t, x, y, p, self.config, self.point_count, rng)}
        for view, image in images.items():
            tensor = torch.from_numpy(image.astype(np.float32))
            result[view] = (tensor - IMAGENET_MEAN) / IMAGENET_STD if self.imagenet_normalize else tensor
        return result
