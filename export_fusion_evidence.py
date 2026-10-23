"""Explicit, opt-in export of five aligned heads from a TRAINED Fusion A.

No model/CUDA import or execution occurs on import. See FUSION_ROUTING.md.
"""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from routing_data import EvidenceWriter, HEADS, ROLES, PROTOCOLS, digest, fingerprint, row_key, protocol_config


class FrozenEvidence(nn.Module):
    """Wrap existing forward unchanged; layer4 hooks reuse original image maps.

    The original forward computes Point features exactly once. ResNet logits
    use the pre-adapter layer4 output through original avgpool + fc.
    Hooks exist only for one call; there is no persistent per-video state.
    """
    def __init__(self, fusion):
        super().__init__()
        self.fusion = fusion.requires_grad_(False).eval()
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, batch):
        self.fusion.eval()
        captured, handles = {}, []
        for view in HEADS[2:]:
            net = self.fusion.image_encoders[view].backbone
            def hook(module, inputs, output, name=view, backbone=net):
                if name in captured:
                    raise RuntimeError('Image layer4 unexpectedly executed more than once')
                captured[name] = backbone.fc(torch.flatten(backbone.avgpool(output), 1))
            handles.append(net.layer4.register_forward_hook(hook))
        try:
            output = self.fusion.forward_batch_outputs(batch)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(HEADS[2:]):
            raise RuntimeError('Missing original image classification paths')
        result = {'fusion': output['logits'], 'point': output['point_logits'], **captured}
        shape = result['fusion'].shape
        if any(v.shape != shape or not torch.isfinite(v).all() for v in result.values()):
            raise ValueError('Nonfinite/misaligned five-head outputs')
        return {f'{key}_logits': result[key] for key in HEADS}


def strict_fusion_state(model, state):
    """Recognize only DDP module.* and BatchForward fusion.* wrappers."""
    transformed = {}
    for key, value in state.items():
        name = key.removeprefix('module.').removeprefix('fusion.')
        if name in transformed:
            raise ValueError('Checkpoint prefix collision')
        transformed[name] = value
    # strict=True checks EVERY parameter/buffer, including original image fc.
    model.load_state_dict(transformed, strict=True)


def load_trained_fusion(checkpoint, device='cpu'):
    # Local trusted project checkpoint includes optimizer and Python/NumPy RNG.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    if cfg.get('architecture') != 'pretrained_filename_V2_prototype_A':
        raise ValueError('Not a trained filename V2 Fusion A checkpoint')
    if int(saved.get('epoch', 0)) < 1 or not saved.get('model_state'):
        raise ValueError('Full trained Fusion checkpoint required, not initializer')
    root = Path(__file__).resolve().parent
    for name, expected in cfg['code'].items():
        if Path(name).name != name or digest(root/name) != expected:
            raise ValueError(f'Original checkpoint code identity changed: {name}')
    if (cfg.get('classes'), cfg.get('point_count'), cfg.get('point_normalization'), cfg.get('point_groups')) != (
            101, 2048, 'official', [1024, 512, 256]):
        raise ValueError('Unsupported input/model protocol; do not silently replace checkpoint settings')
    from ucf101_window_data import checked_audit
    experiment = Path(cfg['experiment'])
    audit = checked_audit(experiment)
    if audit['manifest_sha256'] != cfg['manifest_sha256'] or audit['video_splits_sha256'] != cfg['video_splits_sha256']:
        raise ValueError('Checkpoint data identity mismatch')
    # Full state is authoritative; original source artifacts are not reloaded.
    # Their hashes are retained as provenance, not substituted for trained state.
    from eventmamba_v2_pretrained import PretrainedEventMambaV2
    from ucf101_pretrained_fusion import PretrainedWindowFusion
    model = PretrainedWindowFusion(PretrainedEventMambaV2(num_classes=101),
                                   freeze_images=True, freeze_points=True)
    strict_fusion_state(model, saved['model_state'])
    return FrozenEvidence(model).to(device), saved


def role_mapping(path, dataset_id, selection_protocol='strict'):
    """No implicit role assignment and no split relabeling."""
    obj = json.loads(Path(path).read_text())
    if obj.get('schema') != 1 or obj.get('dataset_id') != dataset_id:
        raise ValueError('Role mapping dataset/schema mismatch')
    protocol_config(selection_protocol)
    if obj.get('selection_protocol', 'strict') != selection_protocol:
        raise ValueError('Role file/export protocol mismatch')
    result = {}
    for row in obj['videos']:
        key = (row['source_split'], row['video_id'])
        if row['role'] not in ROLES or row['source_split'] not in ('train', 'test') or key in result:
            raise ValueError('Invalid/duplicate video role')
        result[key] = row['role']
    roles_by_video = {}
    for (source, video), role in result.items():
        if roles_by_video.setdefault(video, role) != role:
            raise ValueError('One video crosses roles')
        if source == 'test' and (role == 'gate_train' or role == 'gate_val' and selection_protocol != 'original_test_as_val'):
            raise ValueError('Original Test fitting prohibited; selection requires explicit protocol')
    return result


def make_export_dataset(experiment, split, store, roles, dataset_id, *, seed=0, sampling_epoch=0,
                        selection_protocol='strict'):
    from fusion_evidence_dataset import ExportDataset
    return ExportDataset(experiment, split, store, roles, dataset_id, seed=seed,
                         sampling_epoch=sampling_epoch, selection_protocol=selection_protocol)


def export_collate(items):
    from torch.utils.data import default_collate
    return default_collate([{k: v for k, v in i.items() if k != 'metadata'} for i in items]), [i['metadata'] for i in items]


def integration_regression(model, batch):
    """Opt-in real-weight check, same inputs/RNG; no export or optimization."""
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    def reset():
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    before = {k: v.detach().cpu().clone() for k, v in model.fusion.state_dict().items()}
    with torch.inference_mode():
        reset()
        original = model.fusion.forward_batch_outputs(batch)
        reset()
        repeated = model.fusion.forward_batch_outputs(batch)
        reset()
        actual = model(batch)
        # CUDA original-vs-original also varies by a few float32 ULPs.
        # Fixed bounds apply equally to the baseline and wrapper; never adapt
        # tolerances to the observed wrapper error.
        report = {}
        for name, value in [('repeat', repeated['logits']), ('wrapper', actual['fusion_logits'])]:
            torch.testing.assert_close(value, original['logits'], rtol=1e-6, atol=1e-5)
            torch.testing.assert_close(value.softmax(-1), original['logits'].softmax(-1), rtol=1e-5, atol=1e-6)
            if not torch.equal(value.argmax(-1), original['logits'].argmax(-1)):
                raise AssertionError('Regression top1 changed: '+name)
            report[name] = float((value-original['logits']).abs().max())
        torch.testing.assert_close(repeated['point_logits'], original['point_logits'], rtol=0, atol=0)
        torch.testing.assert_close(actual['point_logits'], original['point_logits'], rtol=0, atol=0)
        print(json.dumps({'regression_fusion_max_abs': report, 'rtol': 1e-6, 'atol': 1e-5}), flush=True)
        for view in HEADS[2:]:
            expected = model.fusion.image_encoders[view](batch[view])
            torch.testing.assert_close(actual[f'{view}_logits'], expected, rtol=1e-6, atol=1e-6)
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(m.training for m in model.modules())
    for name, value in model.fusion.state_dict().items():
        torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)
    reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--roles', required=True, type=Path)
    parser.add_argument('--source-split', required=True, choices=('train', 'test'))
    parser.add_argument('--selection-protocol', choices=PROTOCOLS, default='strict')
    parser.add_argument('--dataset-id', default='UCF101-DVS')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sampling-epoch', type=int, default=0)
    parser.add_argument('--max-videos', type=int, help='Explicit small complete-video subset')
    parser.add_argument('--integration-regression', action='store_true', help='One real batch only; writes no cache')
    parser.add_argument('--regression-batches', type=int, default=1)
    args = parser.parse_args()
    if args.regression_batches<1 or args.batch_size<1 or args.workers<0 or args.max_videos is not None and args.max_videos<1:
        parser.error('Invalid batch/workers/subset')
    if not args.integration_regression and args.output is None:
        parser.error('--output required for export')
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    if int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('First exporter is single process; no implicit DDP/FPS changes')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed+100000)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed+100000)
    roles = role_mapping(args.roles, args.dataset_id, args.selection_protocol)
    model, saved = load_trained_fusion(args.checkpoint, args.device)
    cfg = saved['config']
    role_document = json.loads(args.roles.read_text())
    for name, expected in role_document.get('source_sha256', {}).items():
        if name in ('window_manifest.csv', 'video_splits.csv', 'class_to_idx.json', 'dataset_audit.json'):
            if digest(Path(cfg['experiment'])/name) != expected:
                raise ValueError('Role mapping does not match restored checkpoint data')
    data = make_export_dataset(cfg['experiment'], args.source_split, cfg['store'], roles,
                               args.dataset_id, seed=args.seed, sampling_epoch=args.sampling_epoch,
                               selection_protocol=args.selection_protocol)
    if args.max_videos:
        selected = sorted({r['video_id'] for r in data.rows})[:args.max_videos]
        data.rows = [r for r in data.rows if r['video_id'] in selected]
    classes = json.loads((Path(cfg['experiment'])/'class_to_idx.json').read_text())
    identity = dict(dataset_id=args.dataset_id, class_to_idx=classes, time_unit='microseconds',
        **protocol_config(args.selection_protocol),
        checkpoint_sha256=digest(args.checkpoint), checkpoint_epoch=saved['epoch'],
        checkpoint_config_sha256=fingerprint(cfg), data_manifest_sha256=cfg['manifest_sha256'],
        video_splits_sha256=cfg['video_splits_sha256'], roles_sha256=digest(args.roles),
        normalization=dict(points='official t,x,y; x/320,y/240; sampled t min/max',
                           images='original full-window gCSTR + ImageNet mean/std'),
        sampling=dict(point_count=2048, seed=args.seed, epoch=args.sampling_epoch,
                      numpy='per-window SHA256 points seed; sorted sampled indices',
                      fps='original random FPS starts; process seed+100000; batch/order dependent', augmentation='none'),
        export=dict(batch_size=args.batch_size, workers=args.workers, world_size=1,
                    device=args.device, torch=str(torch.__version__), numpy=str(np.__version__),
                    scope='complete selected videos, manifest order'),
        expert_provenance=dict(fit_source_splits=['train'], selection_source_splits=['test'],
            independent_test=False, provenance_basis='existing Fusion A training protocol/config',
            upstream_point_fit_identity='unknown: no verifiable per-video manifest in official initializer',
            point='co-trained Point head from SAME restored fusion state',
            initialization_point_sha256=cfg['point_checkpoint_sha256'],
            initialization_images_sha256=cfg['image_checkpoint_sha256']),
        code={**cfg['code'], **{name:digest(Path(__file__).parent/name) for name in
             ('export_fusion_evidence.py', 'fusion_evidence_dataset.py', 'routing_data.py')}})
    batches = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, collate_fn=export_collate,
                                         **({'multiprocessing_context':'spawn'} if args.workers else {}))
    writer = None
    for step, (batch, metadata) in enumerate(batches, 1):
        batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if args.integration_regression:
            integration_regression(model, batch)
            print('PASS real-weight original Fusion/Point/image equivalence; frozen parameters/BN')
            if step >= args.regression_batches:
                return
            continue
        if writer is None:
            keys = [(args.dataset_id, args.source_split, r['video_id'], str(r['window_id'])) for r in data.rows]
            writer = EvidenceWriter(args.output, identity, keys)
        outputs = model(batch)
        logits = torch.stack([outputs[f'{h}_logits'] for h in HEADS], 1).cpu().numpy()
        writer.append(metadata, logits, batch['statistics'].cpu().numpy())
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(step=step, windows=len(writer.rows), total=len(data))), flush=True)
    if args.integration_regression:
        raise ValueError('Insufficient batches for requested regression coverage')
    writer.finish()
    print(json.dumps(dict(status='complete', output=str(args.output), windows=len(data))))


if __name__ == '__main__':
    main()
