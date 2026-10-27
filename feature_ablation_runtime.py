"""Common protocol, identity, diagnostics and paired evaluation for ablations."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from gcstr_feature_memory_ablation import ARCHITECTURE, SCHEMA, validate_policy
from gcstr_feature_memory_fusion import sha256
from train_gcstr_feature_memory import (SOURCE_FILES as LEGACY_FILES, identity as legacy_identity,
                                      run_pass as legacy_pass)

SOURCE_FILES = (*LEGACY_FILES, 'gcstr_feature_memory_ablation.py', 'feature_ablation_runtime.py',
                'train_gcstr_feature_memory_ablation.py', 'evaluate_gcstr_feature_memory_ablation.py')
PATH_FIELDS = ('experiment', 'store', 'source_checkpoint')


def configuration(path):
    c = json.loads(Path(path).read_text())
    validate_policy(c)
    required = dict(architecture=ARCHITECTURE, schema=SCHEMA, num_classes=101,
                    precision='FP32', optimizer='AdamW', scheduler=None, early_stopping=False,
                    selection_protocol='original_test_as_val', independent_test=False,
                    selection_metric='video_nll', point_count=2048, point_sensor_width=320,
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


def identity(config, data, model):
    result = legacy_identity(config, data, model)
    result.update(architecture=ARCHITECTURE, schema=SCHEMA, ablation_id=config['ablation_id'],
                  code={f: sha256(Path(__file__).parent/f) for f in SOURCE_FILES})
    return result


def environment():
    import sys
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'mamba-ssm', 'causal-conv1d'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, packages=versions, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')


class Observer:
    def __init__(self, trace=False):
        self.gates, self.rows, self.logits = {}, [], []
        self.trace = trace

    def __call__(self, batch, output):
        for key, value in output['diagnostics'].items():
            if key.endswith('/gate'):
                self.gates.setdefault(key, []).append(value.detach().float().cpu().numpy())
        if not self.trace:
            return
        valid = batch['valid'].nonzero(as_tuple=True)[0].cpu().tolist()
        logits = output['logits'].detach().float().cpu()
        points = batch['events'].detach().cpu().numpy()
        for i in valid:
            self.rows.append(dict(video_id=batch['video_id'][i], window_id=str(batch['window_id'][i]),
                label=int(batch['label'][i]), gap_seconds=float(batch['gap_seconds'][i]),
                points_sha256=hashlib.sha256(points[i].tobytes()).hexdigest()))
            self.logits.append(logits[i].numpy().copy())

    def payload(self):
        return dict(rows=self.rows, logits=np.stack(self.logits))


def run_pass(model, data, device, config, optimizer=None, *, capture=False):
    observer = Observer(capture)
    if model.observer is not None:
        raise RuntimeError('Nested observer')
    model.observer = observer
    try:
        results, predictions = legacy_pass(model, data, device, config, optimizer)
    finally:
        model.observer = None
    results['gate_quantiles'] = {
        k: dict(zip(('p0', 'p25', 'p50', 'p75', 'p100'),
                    np.quantile(np.concatenate(v), [0, .25, .5, .75, 1]).tolist()))
        for k, v in observer.gates.items()}
    results['intervention'] = model.intervention
    results['ablation_id'] = model.ablation_id
    results['memory_applicable'] = model.mode == 'GM'
    # Null distinguishes absent modules from measurements of a disabled path.
    if model.mode != 'GM':
        results['diagnostics']['memory/write_gate'] = None
        results['diagnostics']['memory/state_norm'] = None
        for i in range(len(model.base.interactions)):
            for key in ('memory_residual_ratio', 'read_norm', 'gate_history_norm', 'no_history_read_norm'):
                results['diagnostics'][f'stage{i}/{key}'] = None
    if model.mode == 'B':
        results['gate_quantiles'] = None
    return results, predictions, observer.payload() if capture else None


def paired(reference, candidate):
    if reference['rows'] != candidate['rows']:
        raise ValueError('Interventions changed window identity/order/labels/gaps/point sampling')
    a, b = [torch.from_numpy(x['logits']).double().softmax(-1).numpy()
            for x in (reference, candidate)]
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid paired probabilities')
    groups = {}
    for i, row in enumerate(reference['rows']):
        groups.setdefault(row['video_id'], []).append(i)
    rows = []
    for video, idx in sorted(groups.items()):
        label = reference['rows'][idx[0]]['label']
        if any(reference['rows'][i]['label'] != label for i in idx):
            raise ValueError('Inconsistent video labels')
        p, q = a[idx].mean(0), b[idx].mean(0)
        rows.append(dict(video_id=video, label=label, windows=len(idx),
            reference_prediction=int(p.argmax()), candidate_prediction=int(q.argmax()),
            delta_nll=float(-np.log(max(q[label], np.finfo(float).tiny))+
                            np.log(max(p[label], np.finfo(float).tiny))),
            probability_l1=float(np.abs(q-p).sum()), probability_max_abs=float(np.abs(q-p).max())))
    rescued = [r['video_id'] for r in rows if r['candidate_prediction'] == r['label'] != r['reference_prediction']]
    damaged = [r['video_id'] for r in rows if r['reference_prediction'] == r['label'] != r['candidate_prediction']]
    return dict(videos=len(rows), rescued=rescued, damaged=damaged,
                net_correct=len(rescued)-len(damaged), video_pairs=rows,
                window_logits_max_abs=float(np.abs(candidate['logits']-reference['logits']).max()),
                window_probability_max_abs=float(np.abs(b-a).max()))


def verify_evaluation_checkpoint(saved, config, data, model):
    if saved.get('smoke') is not False:
        raise ValueError('Formal evaluation requires a non-smoke checkpoint')
    expected = identity(config, data, model)
    actual = saved['identity']
    if actual.get('architecture') == ARCHITECTURE:
        # For read-only evaluation, permit only audited path relocation.
        normalized = dict(actual, config=dict(actual['config']))
        for key in PATH_FIELDS:
            normalized['config'][key] = config[key]
        if normalized != expected:
            raise ValueError('Ablation checkpoint identity mismatch')
    else:
        if config['ablation_id'] != 'base' or actual.get('architecture') != 'feature_gate_memory_v1':
            raise ValueError('Legacy weights are only valid for original B/G/GM evaluation')
        if 'ddp_code' in actual or actual['config'].get('ddp_world_size', 1) != 1:
            raise ValueError('Legacy DDP provenance requires separate protocol audit; cannot silently reuse')
        original_config = dict(config)
        for key in ('schema', 'base_mode', 'ablation_id', *validate_policy(config).keys()):
            original_config.pop(key)
        original_config['architecture'] = 'feature_gate_memory_v1'
        expected_old = legacy_identity(original_config, data, model)
        normalized = dict(actual, config=dict(actual['config']))
        for key in PATH_FIELDS:
            normalized['config'][key] = config[key]
        if normalized != expected_old:
            raise ValueError('Legacy checkpoint protocol/source/data mismatch')
    model.load_state_dict(saved['model_state'], strict=True)


def verify_historical_gm(saved, checkpoint, record, ddp_source, config, data, model):
    """Explicit exploratory reuse; do not relabel old training as the new protocol."""
    record, checkpoint = Path(record).resolve(), Path(checkpoint).resolve()
    if json.loads((record/'COMPLETE.json').read_text())['status'] != 'verified':
        raise ValueError('Historical record is not complete')
    relative = str(checkpoint.relative_to(record))
    expected_hashes = dict((line.split(None, 1)[1].strip(), line.split(None, 1)[0])
                          for line in (record/'SHA256SUMS').read_text().splitlines() if line.strip())
    if relative not in expected_hashes or sha256(checkpoint) != expected_hashes[relative]:
        raise ValueError('Historical checkpoint checksum mismatch')
    actual = saved['identity']
    if saved.get('smoke') is not False or actual.get('architecture') != 'feature_gate_memory_v1':
        raise ValueError('Expected a formal original GM')
    if (config['base_mode'], config['ablation_id']) != ('GM', 'base') or actual['config']['mode'] != 'GM':
        raise ValueError('Historical reuse is GM evaluation only')
    if sha256(ddp_source) != actual.get('ddp_code'):
        raise ValueError('Historical DDP implementation hash mismatch')
    expected = identity(config, data, model)
    for key in ('manifest_sha256', 'video_splits_sha256', 'store_summary_sha256', 'original_source_code'):
        if actual[key] != expected[key]:
            raise ValueError(f'Historical data/source mismatch: {key}')
    for name, digest in actual['code'].items():
        if Path(name).name != name or sha256(Path(__file__).parent/name) != digest:
            raise ValueError(f'Historical model/runtime code mismatch: {name}')
    # Explicitly allow the known staged batch/DDP history, never other silent
    # learning-rate, normalization, sampling, seed or selection-rule differences.
    differences = {}
    for key, value in actual['config'].items():
        if key in PATH_FIELDS or key == 'architecture':
            continue
        if value != config.get(key):
            if key not in ('video_batch_size_per_gpu', 'ddp_world_size'):
                raise ValueError(f'Historical protocol differs unexpectedly: {key}')
            differences[key] = dict(historical=value, evaluation=config.get(key, 1))
    model.load_state_dict(saved['model_state'], strict=True)
    return dict(comparison_scope='exploratory_unmatched_training_protocol',
                training_identity=actual, evaluation_config=config, differences=differences,
                checkpoint_epoch=saved['epoch'], checkpoint_sha256=expected_hashes[relative],
                note='Single-GPU evaluation/FPS layout may differ from historical DDP metrics. '
                     'Within this intervention suite the inputs/RNG are identical.')


def parameter_report(model, config):
    groups, names = model.parameter_groups(config)
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                groups={g['name']: sum(p.numel() for p in g['params']) for g in groups},
                optimizer_parameter_names=names,
                frozen_new_modules=[n for n, p in model.named_parameters()
                                    if not n.startswith('base.') and not p.requires_grad],
                note='Tensor counts include inactive input columns in the original Gate MLP; '
                     'G/base retains historical registration for exact optimizer compatibility.')
