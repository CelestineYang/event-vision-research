"""UCF101-DVS official split-1 adapter for the existing whole-recording A model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from gcstr_fusion_cached_dataset import CachedFusionDataset
from gcstr_representations import GCSTRConfig

UCF_ROOT = Path('/remote-home/datasets/Joind-dataset for Action Recognition/UCF101-DVS')
UCF_CONFIG = GCSTRConfig(sensor_width=240, sensor_height=180)


def build_manifest(root, output, *, seed=0):
    root, output = Path(root), Path(output)
    official = {}
    required = {'sample_id', 'split', 'npz_relpath', 'event_relpath', 'label_id', 'group',
                'sensor_width', 'sensor_height'}
    for split in ('train', 'test'):
        with (root / 'splits' / f'{split}.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not required <= set(rows[0]):
            raise ValueError(f'Invalid UCF {split} manifest')
        if {int(r['label_id']) for r in rows} != set(range(101)):
            raise ValueError('Expected all 101 UCF classes')
        for r in rows:
            if r['split'] != split or (int(r['sensor_width']), int(r['sensor_height'])) != (240, 180):
                raise ValueError('Unexpected split or sensor dimensions')
            path = Path(r['npz_relpath'])
            if path.is_absolute() or '..' in path.parts or not (root / path).is_file():
                raise ValueError(f'Invalid/missing NPZ: {path}')
        for key in ('sample_id', 'npz_relpath', 'event_relpath'):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError(f'Duplicate {key}')
        official[split] = rows
    for key in ('sample_id', 'npz_relpath', 'event_relpath'):
        if {r[key] for r in official['train']} & {r[key] for r in official['test']}:
            raise ValueError(f'Official Train/Test overlap: {key}')
    parts = official
    group_sets = {s: {(int(r['label_id']), int(r['group'])) for r in rs} for s, rs in parts.items()}
    for a, b in (('train', 'test'),):
        if group_sets[a] & group_sets[b]:
            raise ValueError(f'{a}/{b} class-group overlap')
    rows = []
    for split, items in parts.items():
        rows.extend({**r, 'source_split': split} for r in items)
    # Preserve existing NPZ paths and metadata verbatim rather than reconstructing.
    fields = list(official['train'][0]) + ['source_split']
    report = {'dataset': 'UCF101-DVS', 'official_split': 1, 'classes': 101,
              'width': 240, 'height': 180, 'seed': seed,
              'validation': False, 'protocol': 'official_train_test_only_v3',
              'samples': {s: len(v) for s, v in parts.items()},
              'class_groups': {s: len(v) for s, v in group_sets.items()},
              'official_manifest_hashes': {s: hashlib.sha256((root / 'splits' / f'{s}.csv').read_bytes()).hexdigest() for s in official}}
    if output.exists():
        with output.open(newline='') as handle:
            if list(csv.DictReader(handle)) != rows:
                raise FileExistsError('Existing manifest differs; use a new output path')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp.csv')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
    report_path = output.with_suffix('.report.json')
    if report_path.exists() and json.loads(report_path.read_text()) != report:
        raise FileExistsError('Existing split report differs')
    if not report_path.exists():
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


class UCF101ADataset(CachedFusionDataset):
    def __init__(self, manifest_path, split, *, dataset_root=UCF_ROOT, **kwargs):
        if split not in ('train', 'test'):
            raise ValueError('This protocol accepts only official Train and Test')
        if kwargs.get('cache_dir') is not None:
            raise ValueError('This experiment uses on-the-fly views, no full-event cache')
        super().__init__(manifest_path, split, dataset_root=dataset_root,
                         config=UCF_CONFIG, imagenet_normalize=True, **kwargs)
        if any(not 0 <= int(r['label_id']) < 101 for r in self.rows):
            raise ValueError('UCF labels must be 0..100')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, default=UCF_ROOT)
    p.add_argument('--output', type=Path, default=Path('ucf101_a_official_split1.csv'))
    args = p.parse_args()
    print(json.dumps(build_manifest(args.dataset_root, args.output), indent=2))
