"""Fresh UCF GM fusion; only independent pretrained branch weights are loaded."""
from pathlib import Path
import torch
from eventmamba_v2_pretrained import PretrainedEventMambaV2
from ucf101_pretrained_fusion import PretrainedWindowFusion
from gcstr_feature_memory_ablation import AblationFusion
from gcstr_feature_memory_fusion import sha256
from gcstr_secnet_fusion import VIEWS

ARCHITECTURE = 'ucf101_independent_branches_gm_v1'


def load_model(config):
    if config['num_classes'] != 101 or config['mode'] != 'GM':
        raise ValueError('Expected UCF101 GM')
    if 'source_checkpoint' in config or 'source_sha256' in config:
        raise ValueError('Fusion A checkpoint fields are forbidden')
    path = Path(config['point_checkpoint'])
    if sha256(path) != config['point_sha256']:
        raise ValueError('Independent Point checkpoint changed')
    point = PretrainedEventMambaV2.from_checkpoint(path)
    if not all(torch.isfinite(v).all() for v in point.state_dict().values()):
        raise ValueError('Nonfinite Point checkpoint')
    base = PretrainedWindowFusion(point, freeze_images=True, freeze_points=False)
    manifest = sha256(Path(config['experiment'])/'window_manifest.csv')
    sources = {}
    for view in VIEWS:
        path = Path(config['image_checkpoints'][view])
        if sha256(path) != config['image_sha256'][view]:
            raise ValueError('Independent image checkpoint changed: '+view)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['view'] != view or saved['config']['num_classes'] != 101:
            raise ValueError('Wrong image branch identity')
        if saved['config']['manifest_sha256'] != manifest:
            raise ValueError('Image pretraining/data manifest mismatch; audit revised split before reuse')
        base.image_encoders[view].load_state_dict(saved['model_state'], strict=True)
        if not all(torch.isfinite(v).all() for v in base.image_encoders[view].state_dict().values()):
            raise ValueError('Nonfinite image checkpoint')
        sources[view] = dict(path=str(path), epoch=saved['epoch'], manifest_sha256=manifest)
    model = AblationFusion(base, 'GM', config['seed'], 'base')
    model.source_config = dict(initialization='independent pretrained branches; random base fusion; zero PF/memory residual',
                               point=point.load_report, images=sources)
    return model
