"""Prepare relocated configs and CPU asset audits; never start a training job."""
import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from gcstr_feature_memory_ablation import ARCHITECTURE, SCHEMA, AblationFusion, policy
from gcstr_feature_memory_fusion import load_model as load_original, sha256
from gcstr_video_sequence_dataset import VideoSequenceDataset
from feature_ablation_runtime import configuration, parameter_report, environment, SOURCE_FILES
from train_gcstr_feature_memory import write_json

VARIANTS = {
    'B': ('B', 'base'), 'G': ('G', 'base'), 'GM': ('GM', 'base'),
    'G-FixedGate': ('G', 'fixed_gate'), 'G-PointQuery': ('G', 'point_query'),
    'GM-GateOnly': ('GM', 'gate_only'), 'GM-FeatureOnly': ('GM', 'feature_only'),
    'GM-NoWriteGate': ('GM', 'no_write_gate'), 'GM-FixedUpdate': ('GM', 'fixed_update'),
    'G-GlobalGate': ('G', 'global_gate'), 'G-PointComplement': ('G', 'point_complement'),
    'G-FrameComplement': ('G', 'frame_complement'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--configs', type=Path, default=Path('configs/feature_memory_ablations'))
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.seed < 0:
        p.error('Nonnegative seed required')
    if args.configs.exists() or args.report.exists():
        raise FileExistsError('Use fresh config/report directories')
    root = Path(__file__).parent
    bundle = args.bundle.resolve()
    complete = json.loads((bundle/'COMPLETE.json').read_text())
    if complete['status'] != 'verified':
        raise ValueError('Incomplete asset bundle')
    cfg = json.loads((root/'configs/feature_memory/GM.json').read_text())
    cfg.update(architecture=ARCHITECTURE, schema=SCHEMA, seed=args.seed,
        experiment=str(bundle/'data/experiment'), store=str(bundle/'data/event_store'),
        source_checkpoint=str(bundle/'weights/epoch_040.pt'))
    # Existing strict loader verifies epoch40, full parameter equality, original
    # source hashes, and exact checkpoint SHA. No original pretrained files needed.
    original = load_original(dict(cfg, mode='B'))
    args.configs.mkdir(parents=True)
    args.report.mkdir(parents=True)
    records = {}
    common = None
    for name, (mode, variant) in VARIANTS.items():
        c = dict(cfg, mode=mode, base_mode=mode, ablation_id=variant, **policy(mode, variant))
        path = args.configs/f'{name}.json'
        write_json(path, c)
        configuration(path)
        model = AblationFusion(copy.deepcopy(original.base), mode, args.seed, variant)
        state = model.state_dict()
        if common is None and mode == 'GM':
            common = {k: v.clone() for k, v in state.items()}
        if common is not None:
            for key, value in state.items():
                if key in common:
                    torch.testing.assert_close(value, common[key], rtol=0, atol=0)
        records[name] = dict(config=str(path), base_mode=mode, ablation_id=variant,
            parameters=parameter_report(model, c), formal_status='not_started_by_this_preparation',
            performance=None)
        del model
    data = VideoSequenceDataset(configuration(args.configs/'GM.json'), 'train', 1)
    windows, videos = Counter(), defaultdict(set)
    with (Path(cfg['experiment'])/'window_manifest.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            windows[row['split']] += 1
            videos[row['split']].add(row['video_id'])
    if dict(windows) != dict(train=108066, test=27384) or videos['train'] & videos['test']:
        raise ValueError('Unexpected UCF window coverage/overlap')
    # Read real full-event mmap, augment/project, and sample points on CPU.
    real = data.window(data.groups[sorted(data.groups)[0]][0])
    shapes = {k: list(real[k].shape) for k in ('events', 'cstr3', 'xt_p', 'ty_p')}
    if shapes['events'] != [3, 2048] or any(shapes[v] != [3, 224, 224] for v in ('cstr3', 'xt_p', 'ty_p')):
        raise ValueError('Real input shape mismatch')
    if not all(torch.isfinite(real[k]).all() for k in shapes):
        raise ValueError('Nonfinite real inputs')
    result = dict(status='cpu_assets_verified', bundle=str(bundle), environment=environment(),
        checkpoint=str(cfg['source_checkpoint']), checkpoint_sha256=sha256(cfg['source_checkpoint']),
        source_epoch=40, strict_full_model_load=True, point_encoder=type(original.base.point_encoder).__name__,
        source_code=original.source_config['code'],
        implementation_code={f: sha256(root/f) for f in SOURCE_FILES},
        train_windows=windows['train'], test_windows=windows['test'],
        train_videos=len(videos['train']), test_videos=len(videos['test']),
        train_first_epoch_steps=len(data), real_train_sample_shapes=shapes,
        metadata_and_manifest_identity_verified=True,
        full_event_store_rehash=False, cuda_smoke='pending', variants=records,
        formal_training_started=False)
    write_json(args.report/'assets_and_parameters.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('variants', 'source_code', 'implementation_code')}, indent=2))
    print(json.dumps({k: v['parameters']['trainable'] for k, v in records.items()}, indent=2))


if __name__ == '__main__':
    main()
