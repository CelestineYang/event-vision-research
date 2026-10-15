"""Image-only structural control; custom model, not official EV-CGNet.

Removes SECNet and Attention1. Three independent learned query banks directly
read XY/XT/TY tokens. Not a parameter-matched control.
"""
import hashlib
import numpy as np
import torch
from torch import nn

from gcstr_dataset import DailyDVS200GCSTRDataset, IMAGENET_MEAN, IMAGENET_STD
from gcstr_augmentation import augment_events
from gcstr_representations import build_gcstr_views
from gcstr_models import GCSTRBranchClassifier
from gcstr_secnet_fusion import CrossAttention, GCSTRSECNetFusion, VIEWS


class ImageQueryBlock(nn.Module):
    def __init__(self, dim=128, prototypes=16, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(prototypes, dim) * .02)
        self.view_attention = nn.ModuleDict({v: CrossAttention(dim, heads) for v in VIEWS})
        self.fusion = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, tokens):
        q = self.queries[None].expand(tokens[VIEWS[0]].shape[0], -1, -1)
        z = [self.view_attention[v](q, tokens[v]) for v in VIEWS]
        return (q + self.fusion(torch.cat(z, -1))).mean(1)


class ImageQueryClassifier(nn.Module):
    # Reuse the existing exact token extraction/checkpoint interfaces, but do
    # not construct a SECNet or register any unused trainable point parameters.
    image_tokens = GCSTRSECNetFusion.image_tokens
    load_image_checkpoints = GCSTRSECNetFusion.load_image_checkpoints

    def __init__(self, num_classes=200, dim=128, prototypes=16, heads=4):
        super().__init__()
        self.freeze_images = True
        self.image_encoders = nn.ModuleDict({v: GCSTRBranchClassifier(v, num_classes, False) for v in VIEWS})
        self.image_encoders.requires_grad_(False)
        self.view_adapters = nn.ModuleDict({v: nn.Linear(512, dim) for v in VIEWS})
        self.view_positions = nn.ModuleDict({v: nn.Sequential(nn.Linear(2, dim), nn.GELU(),
                                                            nn.Linear(dim, dim)) for v in VIEWS})
        self.interactions = nn.ModuleList(ImageQueryBlock(dim, prototypes, heads) for _ in range(3))
        self.classifier = nn.Linear(3 * dim, num_classes)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.image_encoders.eval()
        return self

    def forward(self, batch):
        tokens = {v: self.image_tokens(v, batch[v]) for v in VIEWS}
        return self.classifier(torch.cat([block(tokens) for block in self.interactions], -1))


class ImageQueryDataset(DailyDVS200GCSTRDataset):
    """Same epoch/sample variant selection as CachedFusionDataset, no point IO.

Can reuse existing gcstr_cache image variants; no full-event cache is written.
Workers must be recreated after set_epoch (persistent_workers=False).
"""
    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, views=VIEWS, **kwargs)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.rows[index]
        epoch = self.epoch if self.split == 'train' else 0
        identity = f"{self.seed}:{epoch}:{row['sample_id']}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little'))
        variant = int(rng.integers(self.augmentation_count)) if self.augmentation != 'none' else None
        images, missing = {}, []
        for v in VIEWS:
            path = self._cache_path(row, v, variant)
            if path is not None and path.is_file():
                images[v] = np.load(path, allow_pickle=False)
            else:
                missing.append(v)
        if missing:
            with np.load(self.dataset_root / row['npz_relpath'], allow_pickle=False) as data:
                t, x, y, p = (data[k] for k in ('t', 'x', 'y', 'p'))
            if variant is not None:
                identity = f"{self.augmentation_seed}:{row['sample_id']}:{variant}"
                augseed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')
                t, x, y, p = augment_events(t, x, y, p, mode=self.augmentation,
                                           rng=np.random.default_rng(augseed), config=self.config)
            generated = build_gcstr_views(t, x, y, p, views=tuple(missing), config=self.config)
            for v, array in generated.items():
                images[v] = array
                path = self._cache_path(row, v, variant)
                if path is not None:
                    self._atomic_save(path, array)
        sample = {'label': torch.tensor(int(row['label_id']), dtype=torch.long), 'sample_id': row['sample_id']}
        for v in VIEWS:
            image = torch.from_numpy(images[v].astype(np.float32))
            sample[v] = (image - IMAGENET_MEAN) / IMAGENET_STD if self.imagenet_normalize else image
        return sample
