"""Independent DVSAction encoders, fresh fusion and zero residual GM."""
from pathlib import Path
import torch
from eventmamba_v2_pretrained import PretrainedEventMambaV2
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_feature_memory_fusion import sha256
from gcstr_feature_memory_ablation import AblationFusion
from gcstr_secnet_fusion import VIEWS
ARCHITECTURE='dvsaction_independent_branches_gm_v1'
class ActionPoint(PretrainedEventMambaV2):
 def __init__(self):
  super().__init__(num_classes=10)
  self.group_number=[512,256,128]
  for module,n in zip(self.local_grouper_list,self.group_number):module.groups=n
class ActionFusion(GCSTREventMambaFusion):
 def __init__(self,point):super().__init__(num_classes=10,point_encoder=point,freeze_images=True,point_aux=False)
 def forward_outputs(self,events,views):
  tokens=self.encode_views(views);features,coords=self.point_encoder.forward_features(events)
  return {'logits':self.fuse_features(features,coords,tokens),'point_logits':self.point_encoder.classify_features(features)}
def load_model(c):
 if c['num_classes']!=10 or c['mode']!='GM' or 'source_checkpoint' in c or 'source_sha256' in c:raise ValueError('Independent DVSAction GM only')
 if sha256(Path(c['point_checkpoint']))!=c['point_sha256']:raise ValueError('Point identity changed')
 d=torch.load(c['point_checkpoint'],map_location='cpu',weights_only=True)
 point=ActionPoint();point.load_state_dict({k.removeprefix('module.'):v for k,v in d['net'].items()},strict=True)
 point.load_report={'checkpoint':c['point_checkpoint'],'sha256':c['point_sha256'],'epoch':d['epoch'],'accuracy':d['acc'],'groups':[512,256,128],'group_source':'author log name dvsaction_2048_512_batch16 and published configuration; groups not encoded in state_dict','stage_channels':list(point.stage_channels)}
 base=ActionFusion(point);sources={}
 for v in VIEWS:
  path=Path(c['image_checkpoints'][v])
  if sha256(path)!=c['image_sha256'][v]:raise ValueError('Image identity changed')
  s=torch.load(path,map_location='cpu',weights_only=False)
  if s['view']!=v or s['config']['classes']!=10:raise ValueError('Image classes/view mismatch')
  for name,h in s['config']['identity'].items():
   root=Path(c['frames']) if name.startswith('frames_') else Path(c['store']);name=name.removeprefix('frames_')
   if sha256(root/name)!=h:raise ValueError('Image/data mismatch '+name)
  base.image_encoders[v].load_state_dict(s['model_state'],strict=True)
  sources[v]={'checkpoint':str(path),'epoch':s['epoch'],'sha256':c['image_sha256'][v],'selection':'video_nll'}
 if not all(torch.isfinite(v).all() for v in base.state_dict().values()):raise ValueError('Nonfinite weights')
 model=AblationFusion(base,'GM',c['seed'],'base');model.source_config={'initialization':'independent pretrained branches; random fusion; zero residuals','point':point.load_report,'images':sources}
 return model
