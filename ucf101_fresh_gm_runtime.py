"""UCF101 fresh-branch GM protocol and immutable source identity."""
import json
from pathlib import Path
import numpy as np
from gcstr_feature_memory_ablation import SCHEMA, validate_policy
from ucf101_fresh_gm_fusion import ARCHITECTURE
from gcstr_feature_memory_fusion import sha256
from gcstr_video_sequence_dataset import VideoSequenceDataset as OriginalDataset
from feature_ablation_runtime import SOURCE_FILES as COMMON_FILES

def configuration(path):
    c = json.loads(Path(path).read_text())
    validate_policy(c)
    if c["mode"] != "GM" or c["ablation_id"] != "base" or "source_checkpoint" in c or "source_sha256" in c:
        raise ValueError("Fresh GM only; Fusion A inputs forbidden")
    if type(c.get("data_version_confirmed")) is not bool:
        raise ValueError("Explicit dataset-version confirmation required")
    required = dict(architecture=ARCHITECTURE, schema=SCHEMA, num_classes=101,
                    precision='FP32', optimizer='AdamW', scheduler=None, early_stopping=False,
                    selection_protocol='original_test_as_val', independent_test=False,
                    selection_metric='video_top1', selection_tiebreak='video_nll', dataset='UCF101-DVS', point_count=2048, point_sensor_width=320,
                    point_sensor_height=240, freeze_image_backbones=True,
                    freeze_point_bn_running_stats=True)
    for k, v in required.items():
        if k not in c or type(c[k]) is not type(v) or c[k] != v:
            raise ValueError(f'Unsupported protocol: {k}')
    for k in ('epochs', 'video_batch_size_per_gpu', 'tbptt_steps'):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(f'Invalid budget: {k}')
    for k in ('workers', 'seed'):
        if type(c[k]) is not int or c[k] < 0:
            raise ValueError(f'Invalid {k}')
    for k in ('lr_point', 'lr_existing_fusion', 'lr_new_modules', 'gradient_clip'):
        if not np.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f'Invalid {k}')
    for k in ('weight_decay', 'point_aux_weight'):
        if not np.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f'Invalid {k}')
    if c.get('ddp_world_size', 1) != 1:
        raise ValueError('This ablation protocol is single GPU')
    return c



class VideoSequenceDataset(OriginalDataset):
    def __init__(self, config, split, epoch=0, smoke=False):
        local = dict(config, video_batch_size_per_gpu=min(2,config['video_batch_size_per_gpu'])) if smoke else config
        super().__init__(local, split, epoch, smoke)


def identity(config, data, model):
    train = VideoSequenceDataset(config, 'train')
    test = VideoSequenceDataset(config, 'test')
    if set(train.groups) & set(test.groups):
        raise ValueError('Train/Test recording overlap')
    root = Path(__file__).parent
    files = set(COMMON_FILES) | {'ucf101_fresh_gm_fusion.py','ucf101_fresh_gm_runtime.py',
        'train_ucf101_fresh_gm.py','eventmamba_v2_pretrained.py','eventmamba_v2_model.py',
        'eventmamba_v2_layers.py','ucf101_pretrained_fusion.py','ucf101_event_store.py',
        'ucf101_window_data.py','gcstr_representations.py','gcstr_eventmamba_fusion.py',
        'gcstr_secnet_fusion.py','gcstr_models.py'}
    return dict(architecture=ARCHITECTURE, config=config,
        code={f:sha256(root/f) for f in sorted(files)},
        data={f:sha256(Path(config['experiment'])/f) for f in
              ('dataset_audit.json','window_manifest.csv','video_splits.csv','class_to_idx.json')},
        event_store={f:sha256(Path(config['store'])/f) for f in ('summary.json','videos.json')},
        coverage=dict(train_videos=len(train.groups), train_windows=train.windows,
                      test_videos=len(test.groups), test_windows=test.windows),
        source=model.source_config)


def select_best(best, metrics, epoch):
    """Top-1 primary, lower NLL breaks exact ties; preserve earliest full tie."""
    top1, nll = metrics['video_top1'], metrics['video_nll']
    if not np.isfinite([top1, nll]).all():
        raise ValueError('Nonfinite selection score')
    primary = (top1, -nll) > (best['top1'], -best['nll'])
    secondary = nll < best['nll_best']
    result = dict(best)
    if primary:
        result.update(top1=top1, nll=nll, epoch=epoch)
    if secondary:
        result.update(nll_best=nll, nll_epoch=epoch)
    return result, primary, secondary
