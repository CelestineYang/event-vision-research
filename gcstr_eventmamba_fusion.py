"""EventMamba replacement preserving the existing image/prototype fusion path."""
import torch
from torch import nn

from eventmamba_encoder import EventMambaEncoder
from gcstr_secnet_fusion import GCSTRSECNetFusion, VIEWS


class GCSTREventMambaFusion(GCSTRSECNetFusion):
    def __init__(self, num_classes=200, dim=128, prototypes=16, heads=4,
                 point_encoder=None, freeze_images=True, point_aux=False):
        encoder = EventMambaEncoder() if point_encoder is None else point_encoder
        super().__init__(num_classes, dim, prototypes, heads, encoder, freeze_images)
        for stage in self.interactions:
            stage.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.point_head = (nn.Sequential(nn.LayerNorm(encoder.stage_channels[-1]),
                                        nn.Linear(encoder.stage_channels[-1], num_classes))
                           if point_aux else None)

    def encode_views(self, views):
        return {v: self.image_tokens(v, views[v]) for v in VIEWS}

    def fuse_features(self, features, coords, views):
        vectors = [stage(f, c, views) for stage, f, c in zip(self.interactions, features, coords)]
        return self.classifier(torch.cat(vectors, -1))

    def forward_outputs(self, events, views):
        tokens = self.encode_views(views)
        features, coords = self.point_encoder.forward_features(events)
        return {'logits': self.fuse_features(features, coords, tokens),
                'point_logits': None if self.point_head is None else self.point_head(features[-1].mean(1))}

    def forward_batch_outputs(self, batch):
        return self.forward_outputs(batch['events'], {v: batch[v] for v in VIEWS})

    def forward(self, events, views):
        return self.forward_outputs(events, views)['logits']
