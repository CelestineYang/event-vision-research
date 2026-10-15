"""Isolated UCF101-DVS audit, video splits, and full-event window images.

No data preparation or training happens on import. Run ``prepare`` explicitly.
Raw ZIP/MAT data are verified against processed NPZ; polarity is never invented.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from gcstr_augmentation import augment_events
from gcstr_representations import GCSTRConfig, _histogram2d, _normalize, _resize_official

ROOT = Path('/remote-home/datasets/Joind-dataset for Action Recognition/UCF101-DVS')
OUTPUT = Path('experiments/ucf101dvs_gcstr_sliding_v1')
VIEWS = ('cstr3', 'xt_p', 'ty_p')
PROTOCOL = dict(window_us=1_000_000, step_us=500_000, min_events=10001,
                max_window_count_exclusive=40, future_point_count=8192,
                polarity='raw pol == 1 positive, raw pol == 0 negative',
                time_unit='microseconds (source CSV *_usec convention)',
                full_windows_only=True, image_size=224, time_bins=224,
                event_sampling='all_events_in_window', augmentation='spatial_train_only')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    base = Path(__file__).resolve().parent
    return {p: sha(base / p) for p in ('ucf101_window_data.py', 'gcstr_representations.py', 'gcstr_augmentation.py')}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def full_window_bounds(first, last):
    """Recording end is last event time; [start,end), no short tail."""
    count = max(0, (int(last) - int(first) - PROTOCOL['window_us']) // PROTOCOL['step_us'] + 1)
    return [(int(first) + i * PROTOCOL['step_us'],
             int(first) + i * PROTOCOL['step_us'] + PROTOCOL['window_us'])
            for i in range(count)] if count < 40 else []


def make_video_splits(root):
    parts = {s: read_csv(root / 'splits' / f'{s}.csv') for s in ('train', 'test')}
    for key in ('sample_id', 'npz_relpath', 'event_relpath'):
        values = [r[key] for rows in parts.values() for r in rows]
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate source video: {key}')
    group_sets = {s: {(r['label_id'], r['group']) for r in rows} for s, rows in parts.items()}
    if group_sets['train'] & group_sets['test']:
        raise ValueError('Official Train/Test group overlap')
    mapping = {}
    for rows in parts.values():
        if {int(r['label_id']) for r in rows} != set(range(101)):
            raise ValueError('Expected all 101 classes in each official split')
        for r in rows:
            label = int(r['label_id'])
            if r['class_name'] in mapping and mapping[r['class_name']] != label:
                raise ValueError('Inconsistent class mapping')
            mapping[r['class_name']] = label
    if len(mapping) != 101:
        raise ValueError('Class names must map one-to-one to labels')
    for split, rows in parts.items():
        for r in rows:
            if r['split'] != split: raise ValueError('Source split mismatch')
            r['source_split'] = split
    result = parts['train'] + parts['test']
    for r in result:
        r['video_id'] = r['sample_id']
    return result, mapping


def distribution(values):
    return dict(count=len(values), min=int(min(values)), max=int(max(values)),
                mean=float(np.mean(values)), median=float(np.median(values))) if values else dict(count=0)


def prepare(root, output, *, seed=0, smoke_per_class=0):
    """Full raw audit, performed only by explicit prepare invocation.

    User override: retain suspicious records; use the existing stable-sorted
    NPZ stream after verifying exact equality to raw events. Missing/invalid
    polarity, unusable coordinates and I/O failures always block.
    """
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, mapping = make_video_splits(root)
    if smoke_per_class:
        selected = []
        for label in range(101):
            selected.extend([r for r in rows if r['split']=='train' and int(r['label_id'])==label][:smoke_per_class])
        rows = selected
    write_csv(output / 'video_splits.csv', rows, list(rows[0]))
    write_json(output / 'class_to_idx.json', mapping)
    config = dict(protocol=PROTOCOL, seed=seed, validation=False,
                  audit_scope='smoke_train_subset' if smoke_per_class else 'full',
                  split_unit='official_video_split', anomaly_policy='retain_suspicious',
                  evaluation_split='test', selection='test_video_nll', independent_test=False,
                  root=str(root), representation_code=code_hash(),
                  source_manifests={s: sha(root / 'splits' / f'{s}.csv') for s in ('train', 'test')})
    records, windows, fatal = [], [], []
    with ExitStack() as stack:
        archives = {}
        for index, row in enumerate(rows):
            record = dict(video_id=row['video_id'], split=row['split'], label_id=int(row['label_id']),
                          source_archive=row['source_archive'], source_path=row['npz_relpath'], flags=[])
            try:
                name = row['source_archive']
                if name not in archives:
                    z = stack.enter_context(ZipFile(root.parent / name))
                    archives[name] = (z, z.namelist())
                z, names = archives[name]
                matches = [n for n in names if n.endswith('/' + row['event_relpath']) or n == row['event_relpath']]
                if len(matches) != 1:
                    raise ValueError('Raw ZIP member not uniquely resolved')
                raw = loadmat(io.BytesIO(z.read(matches[0])))
                if 'pol' not in raw:
                    raise ValueError('BLOCKED: UCF101-DVS polarity unavailable')
                arrays = {k: np.asarray(raw[v]).reshape(-1) for k, v in
                          (('t', 'ts'), ('x', 'x'), ('y', 'y'), ('p', 'pol'))}
                if len({len(a) for a in arrays.values()}) != 1:
                    raise ValueError('Unequal field lengths')
                if not all(np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError('Non-finite raw event fields')
                if not np.isin(arrays['p'], [0, 1]).all():
                    raise ValueError('BLOCKED: invalid raw polarity; no implicit conversion allowed')
                if any(not np.equal(a, np.floor(a)).all() for a in arrays.values()):
                    raise ValueError('Noninteger event fields / unsupported timestamp unit')
                t = arrays['t'].astype(np.int64)
                x, y, p = arrays['x'], arrays['y'], arrays['p']
                width, height = int(row['sensor_width']), int(row['sensor_height'])
                if width <= 0 or height <= 0:
                    raise ValueError('Invalid sensor metadata')
                source = (root / row['npz_relpath']).resolve()
                source.relative_to(root)
                order = np.argsort(t, kind='stable')
                with np.load(source, allow_pickle=False) as data:
                    if not all(k in data and np.array_equal(data[k], a[order]) for k, a in arrays.items()):
                        raise ValueError('Raw MAT / NPZ mismatch (including polarity)')
                record.update(raw_fields=sorted(k for k in raw if not k.startswith('__')),
                              raw_timestamp_dtype=str(arrays['t'].dtype), sensor_width=width, sensor_height=height,
                              polarity_values=np.unique(p).tolist(), events=len(t), source_sha256=sha(source))
                flags = record['flags']
                if not len(t):
                    flags.append('empty')
                else:
                    record.update(raw_backward_jumps=int((np.diff(t) < 0).sum()),
                                  duration_us=int(t.max())-int(t.min()))
                    if np.any(np.diff(t) < 0): flags.append('raw_nonmonotonic')
                    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
                        raise ValueError('Coordinates cannot form valid sensor projections')
                    if x.min() == x.max() or y.min() == y.max(): flags.append('constant_coordinate')
                    if len(np.unique(p)) < 2: flags.append('single_polarity_review_required')
                    if int(t.max())-int(t.min()) != int(row['duration_usec']):
                        raise ValueError('Timestamp span disagrees with microsecond metadata')
                valid = []
                record['candidate_windows'] = 0
                record['low_event_windows'] = 0
                if len(t):
                    t = t[order]  # Verified existing NPZ order; no rollover repair.
                    count = max(0, (int(t[-1])-int(t[0])-1_000_000)//500_000+1)
                    record['candidate_windows'] = count
                    if count >= 40:
                        record['filter_reason'] = 'official_num_windows_ge40'
                    else:
                        for i, (start, end) in enumerate(full_window_bounds(t[0], t[-1])):
                            left, right = np.searchsorted(t, [start, end], side='left')
                            n = int(right-left)
                            if n <= 10000:
                                record['low_event_windows'] += 1
                                continue
                            identity = dict(source_sha256=record['source_sha256'], video_id=row['video_id'],
                                            window_id=i, start_us=start, end_us=end, protocol=PROTOCOL,
                                            sensor_width=width, sensor_height=height, code=config['representation_code'])
                            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                            valid.append(dict(video_id=row['video_id'], window_id=i, start_us=start, end_us=end,
                                              event_count=n, label_id=row['label_id'], split=row['split'],
                                              source_path=str(source), sensor_width=width, sensor_height=height,
                                              source_sha256=record['source_sha256'], cache_key=key))
                record['valid_windows'] = len(valid)
                if not valid:
                    record.setdefault('filter_reason', ','.join(flags) or 'no_full_window_above10000')
                windows.extend(valid)
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                fatal.append(record['video_id'])
            records.append(record)
            if (index+1) % 100 == 0:
                print(f'Audited {index+1}/{len(rows)} videos', flush=True)
    issues = [r['video_id'] for r in records if r['flags']]
    coverage = {}
    for split in ('train', 'test'):
        subset = [w for w in windows if w['split'] == split]
        coverage[split] = dict(original_videos=sum(r['split'] == split for r in rows),
                               retained_videos=len({w['video_id'] for w in subset}), windows=len(subset),
                               original_class_counts=dict(Counter(r['label_id'] for r in rows if r['split'] == split)),
                               retained_class_video_counts=dict(Counter({w['video_id']: w['label_id'] for w in subset}.values())),
                               class_window_counts=dict(Counter(w['label_id'] for w in subset)),
                               event_count=distribution([w['event_count'] for w in subset]),
                               windows_per_video=distribution(list(Counter(w['video_id'] for w in subset).values())))
        if not smoke_per_class and {int(w['label_id']) for w in subset} != set(range(101)):
            fatal.append(f'{split}:missing_classes_after_filtering')
    if smoke_per_class and len(windows)<4:
        fatal.append('insufficient_smoke_windows')
    fields = ['video_id','window_id','start_us','end_us','event_count','label_id','split','source_path',
              'sensor_width','sensor_height','source_sha256','cache_key']
    write_csv(output / 'window_manifest.csv', windows, fields)
    report = dict(status='passed' if not fatal else 'blocked',
                  config=config, fatal=fatal, suspicious_video_ids=issues, coverage=coverage,
                  excluded_videos=[r for r in records if not r.get('valid_windows')], records=records,
                  manifest_sha256=sha(output / 'window_manifest.csv'),
                  video_splits_sha256=sha(output / 'video_splits.csv'),
                  class_mapping_sha256=sha(output / 'class_to_idx.json'))
    write_json(output / 'dataset_audit.json', report)
    print(json.dumps(dict(status=report['status'], coverage=coverage, fatal_count=len(fatal), suspicious=len(issues)), indent=2))
    return report


def checked_audit(output):
    output = Path(output)
    report = json.loads((output / 'dataset_audit.json').read_text())
    if report['status'] != 'passed':
        raise RuntimeError('Data audit blocked; inspect dataset_audit.json before training')
    if report['config']['representation_code'] != code_hash():
        raise RuntimeError('Representation code changed since audit; prepare a new experiment')
    for file, key in [('window_manifest.csv', 'manifest_sha256'), ('video_splits.csv', 'video_splits_sha256'),
                      ('class_to_idx.json', 'class_mapping_sha256')]:
        if sha(output / file) != report[key]:
            raise RuntimeError(f'Audited manifest changed: {file}')
    return report


def window_views(t, x, y, p, start, end, config, views=VIEWS):
    """Fixed window time axis; retain official density normalization/resize.

    XY mean time uses common start/span for both polarities, then the existing
    per-map max normalization. Empty polarity channels remain zero (no events
    are fabricated). XT/TY histograms share the exact [start,end) time domain.
    """
    if end-start != 1_000_000 or not len(t) or np.any(t < start) or np.any(t >= end):
        raise ValueError('Invalid fixed-window event interval')
    if not np.isin(p, [0, 1]).all():
        raise ValueError('Invalid polarity')
    if np.any((x < 0) | (x >= config.sensor_width) | (y < 0) | (y >= config.sensor_height)):
        raise ValueError('Invalid coordinates')
    w, h, bins = config.sensor_width, config.sensor_height, config.time_bins
    pos, neg = p == 1, p == 0
    result = {}
    if 'cstr3' in views:
        def mean(mask):
            sums, counts = np.zeros((w,h)), np.full((w,h), .01)
            np.add.at(sums, (x[mask],y[mask]), (t[mask]-start).astype(np.float64)/(end-start))
            np.add.at(counts, (x[mask],y[mask]), 1.)
            return _normalize(sums/counts)
        density = _normalize(_histogram2d(x,y,(w,h),((0,w-1),(0,h-1))))
        result['cstr3'] = _resize_official(np.stack([mean(neg),density,mean(pos)]), config.image_size)
    for view in ('xt_p', 'ty_p'):
        if view not in views: continue
        if view == 'xt_p':
            a,b,shape,ranges = x,t,(w,bins),((0,w-1),(start,end))
        else:
            a,b,shape,ranges = t,y,(bins,h),((start,end),(0,h-1))
        counts = _normalize(np.stack([_histogram2d(a[m],b[m],shape,ranges) for m in (pos,neg)]))
        result[view] = _resize_official(np.concatenate([counts,np.zeros((1,*shape))]), config.image_size)
    if set(result) != set(views) or any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Invalid projection result')
    return result


class WindowDataset(Dataset):
    def __init__(self, output, split, view, *, require_cache=True, seed=0):
        self.output, self.view, self.require_cache = Path(output), view, require_cache
        self.augmentation = 'spatial' if split == 'train' else 'none'
        self.seed, self.epoch = seed, 0
        if view not in VIEWS or split not in ('train', 'test'):
            raise ValueError('Invalid split/view')
        self.rows = [r for r in read_csv(self.output / 'window_manifest.csv') if r['split'] == split]
        if not self.rows: raise ValueError(f'Empty split {split}')
        self._video = None
        self._arrays = None

    def __len__(self): return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def augmentation_rng(self, row):
        # Independent of worker scheduling/rank/view; reproducible after resume.
        key = f"{self.seed}:{self.epoch}:{row['video_id']}:{row['window_id']}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
        return np.random.default_rng(seed)

    def cache_path(self, row):
        return self.output / 'cache' / self.view / (row['cache_key'] + '.npy')

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.cache_path(row)
        spatial = self.augmentation == 'spatial'
        if not spatial and path.exists():
            image = np.load(path, allow_pickle=False)
        else:
            if not spatial and self.require_cache:
                raise RuntimeError(f'Build window image cache before training: {path}')
            if self._video != row['source_path']:
                if sha(row['source_path']) != row['source_sha256']:
                    raise RuntimeError('Source changed after audit')
                with np.load(row['source_path'], allow_pickle=False) as f:
                    self._arrays = {k: f[k] for k in ('t','x','y','p')}
                self._video = row['source_path']
            a = self._arrays
            start, end = int(row['start_us']), int(row['end_us'])
            left, right = np.searchsorted(a['t'], [start,end], side='left')
            if right-left != int(row['event_count']) or right-left <= 10000:
                raise ValueError('Window event count changed')
            cfg = GCSTRConfig(sensor_width=int(row['sensor_width']), sensor_height=int(row['sensor_height']))
            events = tuple(a[k][left:right] for k in ('t','x','y','p'))
            if spatial:
                events = augment_events(*events,mode='spatial',rng=self.augmentation_rng(row),config=cfg)
            # Keep original fixed temporal bounds; filter raw count only before augmentation.
            image = window_views(*events, start, end, cfg, (self.view,))[self.view]
            if not spatial:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
                with tmp.open('wb') as f: np.save(f, image, allow_pickle=False)
                tmp.replace(path)
        if image.shape != (3,224,224) or not np.isfinite(image).all():
            raise ValueError(f'Corrupt image cache: {path}')
        image = (image - np.array([.485,.456,.406],np.float32)[:,None,None]) / np.array([.229,.224,.225],np.float32)[:,None,None]
        return torch.from_numpy(image), int(row['label_id']), row['video_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','cache'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke-per-class',type=int,default=0,
                        help='Train-only raw audit subset; cannot authorize formal training')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.smoke_per_class<0: parser.error('smoke-per-class must be >=0')
        report = prepare(args.root,args.output,seed=args.seed,smoke_per_class=args.smoke_per_class)
        if report['status'] != 'passed': raise SystemExit(2)
    else:
        checked_audit(args.output)
        # Explicit user override: Test participates in evaluation and selection.
        for view in VIEWS:
            for split in ('test',):
                data = WindowDataset(args.output,split,view,require_cache=False)
                for i in range(len(data)):
                    data[i]
                    if (i+1) % 1000 == 0: print(view,split,i+1,len(data),flush=True)


if __name__ == '__main__': main()
