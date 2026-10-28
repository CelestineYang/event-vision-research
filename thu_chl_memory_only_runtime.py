"""THU protocol and immutable source identity for independent branch initialization."""
import json
from pathlib import Path
import numpy as np
from gcstr_memory_only_fusion import ARCHITECTURE, MemoryOnlyFusion
SCHEMA = 1
from gcstr_feature_memory_fusion import sha256
from thu_chl_sequence_dataset import THUVideoSequenceDataset
from feature_ablation_runtime import SOURCE_FILES as COMMON_FILES

def configuration(path):
    c = json.loads(Path(path).read_text())
    expected = dict(mode="M", base_mode="M", ablation_id="memory_only",
                    frame_query_source="none", gate_policy="none", history_to_gate=False,
                    history_direct_residual=True, memory_update="learned_gate")
    for k,v in expected.items():
        if type(c.get(k)) is not type(v) or c[k] != v:
            raise ValueError("Invalid M policy: " + k)
    required = dict(architecture=ARCHITECTURE, schema=SCHEMA, num_classes=50,
                    precision='FP32', optimizer='AdamW', scheduler=None, early_stopping=False,
                    selection_protocol='original_test_as_val', independent_test=False,
                    selection_metric='video_top1', selection_tiebreak='video_nll', dataset='THU-EACT-50-CHL', point_count=2048, point_sensor_width=346,
                    point_sensor_height=260, freeze_image_backbones=True,
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



def identity(config, data, model):
    train = THUVideoSequenceDataset(config, 'train')
    test = THUVideoSequenceDataset(config, 'test')
    if set(train.groups) & set(test.groups):
        raise ValueError('Train/Test overlap')
    if (train.windows, len(train.groups), test.windows, len(test.groups)) != (7129,1786,2392,544):
        raise ValueError('Unexpected full THU coverage')
    files = set(COMMON_FILES) | {'gcstr_memory_only_fusion.py', 'thu_chl_memory_only_runtime.py', 'train_thu_chl_memory_only.py','thu_chl_ablation_runtime.py', 'thu_chl_ablation_fusion.py',
        'train_thu_chl_ablation.py', 'thu_chl_sequence_dataset.py', 'thu_chl_render.py',
        'eventmamba_v1_pretrained.py', 'eventmamba_v1_model.py', 'eventmamba_v1_layers.py',
        'gcstr_eventmamba_fusion.py', 'gcstr_secnet_fusion.py', 'gcstr_models.py',
        'gcstr_representations.py'}
    root = Path(__file__).parent
    return dict(architecture=ARCHITECTURE, config=config,
        code={f:sha256(root/f) for f in sorted(files)},
        data={f:sha256(Path(config['store'])/f) for f in
              ('config.json','samples.json','train_windows.jsonl','test_windows.jsonl')},
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


def load_model(config):
    from thu_chl_ablation_fusion import load_model as load_branches
    # Reuse strict branch SHA/data identity checks; B constructs no extras.
    baseline = load_branches(dict(config, mode='B', base_mode='B', ablation_id='base'))
    model = MemoryOnlyFusion(baseline.base, config['seed'])
    model.source_config = baseline.source_config
    return model


def parameter_report(model, config):
    from feature_ablation_runtime import parameter_report as report
    result = report(model, config)
    result['note'] = 'M retains history normalization and memory writer gate; no PF Gate or independent frame reader.'
    return result


def run_pass(model, data, device, config, optimizer=None):
    from train_gcstr_feature_memory import run_pass as original_pass
    metrics, predictions = original_pass(model, data, device, config, optimizer)
    metrics.update(ablation_id='memory_only', memory_applicable=True, gate_quantiles=None)
    for i in range(len(model.base.interactions)):
        for key in ('gate', 'pf_residual_ratio', 'gate_history_norm'):
            metrics['diagnostics'][f'stage{i}/{key}'] = None
    return metrics, predictions, None
