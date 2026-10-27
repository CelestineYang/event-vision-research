"""THU GM initialization from independent pretrained branches, not UCF Fusion A."""
from pathlib import Path
import torch
from eventmamba_v1_pretrained import PretrainedEventMambaV1
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_feature_memory_fusion import FeatureMemoryFusion,sha256
from gcstr_secnet_fusion import VIEWS
from gcstr_feature_memory_ablation import AblationFusion

class THUWindowFusion(GCSTREventMambaFusion):
    def __init__(self,encoder):
        super().__init__(num_classes=50,point_encoder=encoder,freeze_images=True,point_aux=False)
    def forward_outputs(self,events,views):
        tokens=self.encode_views(views);features,coords=self.point_encoder.forward_features(events)
        return {'logits':self.fuse_features(features,coords,tokens),'point_logits':self.point_encoder.classify_features(features)}

def load_model(config):
    if config['num_classes']!=50:raise ValueError('THU requires 50 classes')
    path=Path(config['point_checkpoint'])
    if sha256(path)!=config['point_sha256']:raise ValueError('Point checkpoint changed')
    base=THUWindowFusion(PretrainedEventMambaV1.from_checkpoint(path))
    report={}
    for view in VIEWS:
        path=Path(config['image_checkpoints'][view])
        if sha256(path)!=config['image_sha256'][view]:raise ValueError('Image checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['view']!=view or saved['config']['classes']!=50:raise ValueError('Wrong THU image checkpoint')
        for name,value in saved['config']['identity'].items():
            if sha256(Path(config['store'])/name)!=value:raise ValueError('Image source data mismatch: '+name)
        base.image_encoders[view].load_state_dict(saved['model_state'],strict=True)
        report[view]={'path':str(path),'epoch':saved['epoch']}
    model=AblationFusion(base,config['mode'],config['seed'],config['ablation_id'])
    model.source_config={'initialization':'independent pretrained branches; random fusion; zero residual gate/memory',
        'point':base.point_encoder.load_report,'images':report}
    return model
