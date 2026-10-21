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

from routing_data import EvidenceWriter, HEADS, ROLES, digest, fingerprint, row_key


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


def role_mapping(path, dataset_id):
    """No implicit role assignment and no split relabeling."""
    obj = json.loads(Path(path).read_text())
    if obj.get('schema') != 1 or obj.get('dataset_id') != dataset_id:
        raise ValueError('Role mapping dataset/schema mismatch')
    result = {}
    for row in obj['videos']:
        key = (row['source_split'], row['video_id'])
        if row['role'] not in ROLES or key in result:
            raise ValueError('Invalid/duplicate video role')
        result[key] = row['role']
    roles_by_video = {}
    for (source, video), role in result.items():
        if roles_by_video.setdefault(video, role) != role:
            raise ValueError('One video crosses roles')
        if source.lower() == 'test' and role in ('gate_train', 'gate_val'):
            raise ValueError('Original Test cannot be relabeled for Gate fitting/selection')
    return result


def event_statistics(events, row, point_count):
    t, x, y, p = events
    n = len(t)
    width, height = int(row['sensor_width']), int(row['sensor_height'])
    duration = (int(row['end_us'])-int(row['start_us']))/1e6
    if n == 0 or duration<=0:
        raise ValueError('Invalid event statistics')
    coverage = len(np.unique(np.asarray(y, dtype=np.int64)*width+np.asarray(x, dtype=np.int64)))/(width*height)
    return np.array([np.log1p(n/(duration*width*height)), np.log1p(n), coverage, point_count/n], dtype=np.float32)


def make_export_dataset(experiment, split, store, roles, dataset_id, *, seed=0, sampling_epoch=0):
    from ucf101_fusion_window_data import FusionWindowDataset

    class ExportDataset(FusionWindowDataset):
        def __init__(self):
            super().__init__(experiment, split, store, point_count=2048,
                             point_normalization='official', seed=seed)
            # Original source split stays untouched. Override augmentation and
            # sampling epoch explicitly rather than spoofing split='test'.
            self.augmentation = 'none'
            self.set_epoch(sampling_epoch)
            available = {r['video_id'] for r in self.rows}
            requested = {video for source, video in roles if source == split}
            if requested-available:
                raise ValueError('Role mapping names videos absent from the audited source split')
            self.rows = [r for r in self.rows if (split, r['video_id']) in roles]
            if not self.rows:
                raise ValueError('Role mapping selects no videos from this source split')

        def __getitem__(self, index):
            batch = super().__getitem__(index)
            row = self.rows[index]
            batch['statistics'] = torch.from_numpy(event_statistics(self.events.get(row), row, self.point_count))
            batch['metadata'] = dict(dataset_id=dataset_id, source_split=split,
                                    role=roles[(split, row['video_id'])], video_id=row['video_id'],
                                    window_id=str(row['window_id']), start_us=int(row['start_us']),
                                    end_us=int(row['end_us']), label_id=int(row['label_id']),
                                    # Current fine-tuning fit is known; upstream
                                    # official Point checkpoint has no hashed
                                    # per-video training manifest in its payload.
                                    expert_fit_exposure='seen' if split == 'train' else 'unknown',
                                    expert_selection_exposure='seen' if split == 'test' else 'unseen')
            return batch

    return ExportDataset()


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
        actual = model(batch)
        torch.testing.assert_close(actual['fusion_logits'], original['logits'], rtol=0, atol=0)
        torch.testing.assert_close(actual['point_logits'], original['point_logits'], rtol=0, atol=0)
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
    parser.add_argument('--dataset-id', default='UCF101-DVS')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sampling-epoch', type=int, default=0)
    parser.add_argument('--max-videos', type=int, help='Explicit small complete-video subset')
    parser.add_argument('--integration-regression', action='store_true', help='One real batch only; writes no cache')
    args = parser.parse_args()
    if args.batch_size<1 or args.workers<0 or args.max_videos is not None and args.max_videos<1:
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
    roles = role_mapping(args.roles, args.dataset_id)
    model, saved = load_trained_fusion(args.checkpoint, args.device)
    cfg = saved['config']
    data = make_export_dataset(cfg['experiment'], args.source_split, cfg['store'], roles,
                               args.dataset_id, seed=args.seed, sampling_epoch=args.sampling_epoch)
    if args.max_videos:
        selected = sorted({r['video_id'] for r in data.rows})[:args.max_videos]
        data.rows = [r for r in data.rows if r['video_id'] in selected]
    classes = json.loads((Path(cfg['experiment'])/'class_to_idx.json').read_text())
    identity = dict(dataset_id=args.dataset_id, class_to_idx=classes, time_unit='microseconds',
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
            verified_selection_unseen_source_splits=['train'],
            independent_test=False, provenance_basis='existing Fusion A training protocol/config',
            upstream_point_fit_identity='unknown: no verifiable per-video manifest in official initializer',
            point='co-trained Point head from SAME restored fusion state',
            initialization_point_sha256=cfg['point_checkpoint_sha256'],
            initialization_images_sha256=cfg['image_checkpoint_sha256']),
        code={**cfg['code'], **{name:digest(Path(__file__).parent/name) for name in
             ('export_fusion_evidence.py', 'routing_data.py')}})
    batches = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, collate_fn=export_collate)
    writer = None
    for step, (batch, metadata) in enumerate(batches, 1):
        batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if args.integration_regression:
            integration_regression(model, batch)
            print('PASS real-weight original Fusion/Point/image equivalence; frozen parameters/BN')
            return
        if writer is None:
            keys = [(args.dataset_id, args.source_split, r['video_id'], str(r['window_id'])) for r in data.rows]
            writer = EvidenceWriter(args.output, identity, keys)
        outputs = model(batch)
        logits = torch.stack([outputs[f'{h}_logits'] for h in HEADS], 1).cpu().numpy()
        writer.append(metadata, logits, batch['statistics'].cpu().numpy())
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(step=step, windows=len(writer.rows), total=len(data))), flush=True)
    writer.finish()
    print(json.dumps(dict(status='complete', output=str(args.output), windows=len(data))))


if __name__ == '__main__':
    main()
