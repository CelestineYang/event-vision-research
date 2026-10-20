"""Independent window-level A fusion entry; no training side effects."""
from pathlib import Path
import torch

from eventmamba_v2_pretrained import PretrainedEventMambaV2
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_secnet_fusion import VIEWS


class PretrainedWindowFusion(GCSTREventMambaFusion):
    def __init__(self, encoder, *, freeze_images=True, freeze_points=False):
        super().__init__(num_classes=101, point_encoder=encoder,
                         freeze_images=freeze_images, point_aux=False)
        self.freeze_points = freeze_points
        encoder.requires_grad_(not freeze_points)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self,'freeze_points',False): self.point_encoder.eval()
        return self

    def forward_outputs(self, events, views):
        tokens = self.encode_views(views)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_points):
            features, coords = self.point_encoder.forward_features(events)
            point_logits = self.point_encoder.classify_features(features)
        return {'logits': self.fuse_features(features,coords,tokens),
                'point_logits': point_logits}


def load_pretrained_fusion(experiment, point_checkpoint, *,
                          freeze_images=True, freeze_points=False):
    encoder = PretrainedEventMambaV2.from_checkpoint(point_checkpoint)
    if encoder.classifier[-1].out_features != 101:
        raise ValueError('Expected UCF101 point checkpoint')
    model = PretrainedWindowFusion(encoder, freeze_images=freeze_images,
                                  freeze_points=freeze_points)
    paths = {v:Path(experiment)/'checkpoints'/v/'best_loss.pt' for v in VIEWS}
    for view,path in paths.items():
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint.get('view') != view: raise ValueError('Image checkpoint view mismatch')
        model.image_encoders[view].load_state_dict(checkpoint['model_state'],strict=True)
    model.pretrained_sources = {**encoder.load_report,'images':{v:str(p) for v,p in paths.items()}}
    return model
