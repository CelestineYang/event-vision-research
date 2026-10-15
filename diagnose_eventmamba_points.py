"""Validation-only paired point mismatch diagnostic; no training or test access."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from gcstr_fusion_cached_dataset import CachedFusionDataset
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_secnet_fusion import VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('gcstr_checkpoints/eventmamba_full_seed0_v1/last.pt'))
    parser.add_argument('--output', type=Path, default=Path('logs/eventmamba_point_diagnostic.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(1000000)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    manifest = Path(config['manifest'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint['signature']['manifest_sha256']:
        raise ValueError('Manifest changed since training')
    if config.get('fusion', 'prototype') == 'local':
        from gcstr_local_fusion import GCSTRLocalEventMambaFusion
        model = GCSTRLocalEventMambaFusion(point_aux=config.get('point_aux_weight', 0) > 0)
    else:
        model = GCSTREventMambaFusion(point_aux=config.get('point_aux_weight', 0) > 0)
    model = model.cuda().eval()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    data = CachedFusionDataset(manifest, 'val', dataset_root=config['dataset_root'],
                               point_count=config['point_count'], seed=config['seed'])
    by_class = {}
    for i, row in enumerate(data.rows):
        by_class.setdefault(int(row['label_id']), []).append(i)
    if len(by_class) != 200 or min(map(len, by_class.values())) < 2:
        raise ValueError('Expected at least two validation samples for all 200 classes')
    # Interleaving classes guarantees all batch-local nonzero rolls are cross-class.
    indices = [by_class[c][j] for j in range(2) for c in sorted(by_class)]
    loader = DataLoader(Subset(data, indices), batch_size=16, shuffle=False, num_workers=2,
                        generator=torch.Generator().manual_seed(0))
    shifts = (0, 1, 3, 7)
    logits_all = {s: [] for s in shifts}
    labels_all, ids, donors = [], [], {s: [] for s in shifts}
    torch.manual_seed(1000000)
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            labels = batch['label']
            views = model.encode_views({v: batch[v].cuda() for v in VIEWS})
            features, coords = model.point_encoder.forward_features(batch['events'].cuda())
            for shift in shifts:
                order = torch.arange(len(labels)).roll(shift)
                if shift and (labels == labels[order]).any():
                    raise AssertionError('A donor has the same class as the target')
                gpu_order = order.cuda()
                logits_all[shift].append(model.fuse_features(
                    [f[gpu_order] for f in features], [c[gpu_order] for c in coords], views).cpu())
                donors[shift].extend(batch['sample_id'][i] for i in order.tolist())
            labels_all.append(labels)
            ids.extend(batch['sample_id'])
            if step == 1 or step % 5 == 0:
                print(json.dumps({'batch': step, 'batches': len(loader)}), flush=True)
    labels = torch.cat(labels_all)
    outputs = {s: torch.cat(x) for s, x in logits_all.items()}
    baseline_pred = outputs[0].argmax(1)
    baseline_correct = baseline_pred == labels
    metrics = {}
    for shift, logits in outputs.items():
        pred = logits.argmax(1)
        correct = pred == labels
        metrics[str(shift)] = {
            'top1': correct.float().mean().item(),
            'top5': (logits.topk(5, dim=1).indices == labels[:, None]).any(1).float().mean().item(),
            'loss': torch.nn.functional.cross_entropy(logits, labels).item(),
            'changed_predictions': (pred != baseline_pred).sum().item(),
            'correct_to_wrong': (baseline_correct & ~correct).sum().item(),
            'wrong_to_correct': (~baseline_correct & correct).sum().item(),
            'mean_abs_probability_change': (logits.softmax(1) - outputs[0].softmax(1)).abs().mean().item(),
        }
    report = {'checkpoint': str(args.checkpoint), 'epoch': checkpoint['epoch'],
              'split': 'val', 'samples': len(ids), 'selection': 'first 2 rows per class; class-interleaved',
              'seed': 1000000, 'batch_size': 16,
              'protocol': 'encode points once; jointly permute features and coordinates; fixed image tokens',
              'metrics': metrics,
              'samples_detail': [{'sample_id': sid, 'label': int(labels[i]),
                  'predictions': {str(s): int(outputs[s][i].argmax()) for s in shifts},
                  'point_donors': {str(s): donors[s][i] for s in shifts}} for i, sid in enumerate(ids)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({'samples': len(ids), 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
