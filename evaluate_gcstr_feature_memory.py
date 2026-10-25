"""Fixed-parameter whole-video evaluation and paired prediction comparisons."""
import argparse
import json
from pathlib import Path

import torch

from gcstr_feature_memory_fusion import load_model
from gcstr_video_sequence_dataset import VideoSequenceDataset
from train_gcstr_feature_memory import configuration, identity, run_pass, seed_all, write_json


def compare_predictions(reference, candidate):
    def indexed(path):
        data = json.loads(Path(path).read_text())
        # Native B/G/GM predictions or historical Stage2 video_index.json.
        rows = data if isinstance(data, list) else data['main']
        result = {}
        for row in rows:
            key = row['video_id']
            if key in result:
                raise ValueError('Duplicate prediction video')
            result[key] = dict(label=row.get('label', row.get('label_id')),
                               windows=row.get('windows', len(row.get('window_index', []))),
                               prediction=row['prediction'])
        return result
    a, b = indexed(reference), indexed(candidate)
    if a.keys() != b.keys():
        raise ValueError('Different evaluation video coverage')
    rescued, damaged = [], []
    for video in a:
        if a[video]['label'] != b[video]['label'] or a[video]['windows'] != b[video]['windows']:
            raise ValueError('Different evaluation labels/window coverage')
        old, new = a[video]['prediction'] == a[video]['label'], b[video]['prediction'] == b[video]['label']
        if new and not old:
            rescued.append(video)
        if old and not new:
            damaged.append(video)
    return dict(videos=len(a), rescued=sorted(rescued), damaged=sorted(damaged),
                net_correct=len(rescued)-len(damaged))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluate = commands.add_parser('eval')
    evaluate.add_argument('--config', required=True, type=Path)
    evaluate.add_argument('--checkpoint', required=True, type=Path)
    evaluate.add_argument('--output', required=True, type=Path)
    evaluate.add_argument('--device', default='cuda:0')
    compare = commands.add_parser('compare')
    compare.add_argument('--reference', required=True, type=Path)
    compare.add_argument('--candidate', required=True, type=Path)
    compare.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == 'compare':
        result = compare_predictions(args.reference, args.candidate)
    else:
        config = configuration(args.config)
        seed_all(config['seed'])
        model = load_model(config)
        data = VideoSequenceDataset(config, 'test')
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if saved['identity'] != identity(config, data, model) or saved['smoke']:
            raise ValueError('Evaluation checkpoint identity mismatch or smoke artifact')
        model.load_state_dict(saved['model_state'], strict=True)
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Real network requires CUDA')
        model.to(device)
        seed_all(config['seed']+100000)
        metrics, predictions = run_pass(model, data, device, config)
        result = dict(epoch=saved['epoch'], selection_protocol='original_test_as_val',
                      independent_test=False, metrics=metrics, **predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
