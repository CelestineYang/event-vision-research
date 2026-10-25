"""Exact-coverage video schedules and aligned, video-consistent augmentation."""
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from gcstr_representations import GCSTRConfig
from ucf101_event_store import EventStore
from ucf101_window_data import checked_audit, read_csv, window_views, VIEWS


def keyed_rng(*parts):
    key = ':'.join(map(str, parts)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))


def sample_spatial_params(seed, epoch, video_id):
    rng = keyed_rng('geometry', seed, epoch, video_id)
    return float(np.deg2rad(rng.uniform(-15, 15))), tuple(rng.uniform(0, .1, 4))


def apply_spatial_params(events, params, rng, config):
    t, x, y, p = (np.asarray(a) for a in events)
    if not len(t):
        raise ValueError('Empty source window')
    angle, (left, right, top, bottom) = params
    xf, yf = x.astype(float) + rng.uniform(-.5, .5, len(x)), y.astype(float) + rng.uniform(-.5, .5, len(y))
    cx, cy = (config.sensor_width-1)/2, (config.sensor_height-1)/2
    dx, dy = xf-cx, yf-cy
    xf, yf = dx*np.cos(angle)-dy*np.sin(angle)+cx, dx*np.sin(angle)+dy*np.cos(angle)+cy
    inside = (xf >= 0) & (xf <= config.sensor_width-1) & (yf >= 0) & (yf <= config.sensor_height-1)
    fallbacks = 0
    if inside.any():
        t, xf, yf, p = t[inside], xf[inside], yf[inside], p[inside]
    else:
        fallbacks += 1
    xmin, xmax = left*config.sensor_width, (1-right)*config.sensor_width
    ymin, ymax = top*config.sensor_height, (1-bottom)*config.sensor_height
    inside = (xf >= xmin) & (xf < xmax) & (yf >= ymin) & (yf < ymax)
    if inside.any():
        t, xf, yf, p = t[inside], xf[inside], yf[inside], p[inside]
    else:
        fallbacks += 1
    # Preserve historical fallback (retain then clip), explicitly count its use.
    x = np.clip(np.rint((xf-xmin)/(xmax-xmin)*(config.sensor_width-1)), 0, config.sensor_width-1).astype(np.int64)
    y = np.clip(np.rint((yf-ymin)/(ymax-ymin)*(config.sensor_height-1)), 0, config.sensor_height-1).astype(np.int64)
    return (t, x, y, p), fallbacks


def video_groups(rows):
    groups = defaultdict(list)
    identities = set()
    for i, row in enumerate(rows):
        identity = (row['video_id'], str(row['window_id']))
        if identity in identities:
            raise ValueError('Duplicate window identity')
        identities.add(identity)
        groups[row['video_id']].append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: int(rows[i]['start_us']))
        times = [int(rows[i]['start_us']) for i in indices]
        if len(times) != len(set(times)) or len({rows[i]['label_id'] for i in indices}) != 1:
            raise ValueError('Invalid video timestamps or labels')
    return dict(groups)


def make_schedule(groups, batch_size, steps, seed, epoch, shuffle):
    if batch_size < 1 or steps < 1:
        raise ValueError('Positive video batch and TBPTT length required')
    videos = sorted(groups)
    if shuffle:
        keyed_rng('schedule', seed, epoch).shuffle(videos)
    schedule = []
    for offset in range(0, len(videos), batch_size):
        batch = videos[offset:offset+batch_size]
        longest = max(len(groups[v]) for v in batch)
        for start in range(0, longest, steps):
            schedule.append(dict(videos=batch, start=start, stop=min(start+steps, longest),
                                 group_start=start == 0, group_end=start+steps >= longest))
    expected = sum(math.ceil(max(len(groups[v]) for v in videos[i:i+batch_size])/steps)
                   for i in range(0, len(videos), batch_size))
    assert len(schedule) == expected
    return schedule


class VideoSequenceDataset(Dataset):
    """One item is one contiguous chunk, NOT an entire video's event tensors.

    DataLoader must use batch_size=None, shuffle=False; only this schedule shuffles.
    Empty lanes contain zeros and valid=False and are never encoded/supervised.
    """
    def __init__(self, config, split, epoch=0, smoke=False):
        self.config, self.split, self.epoch = config, split, epoch
        self.output = Path(config['experiment'])
        self.audit = checked_audit(self.output)
        self.rows = [r for r in read_csv(self.output/'window_manifest.csv') if r['split'] == split]
        self.groups = video_groups(self.rows)
        self.events = EventStore(config['store'], self.output)
        if smoke:
            # Train only; three complete videos, >=2 chunks, plus singleton batch.
            if split != 'train':
                raise ValueError('Smoke cannot read Test')
            chosen = [v for v in sorted(self.groups) if len(self.groups[v]) >= 2*config['tbptt_steps']][:3]
            if len(chosen) != 3:
                raise ValueError('Need three Train videos with >=2 chunks')
            self.groups = {v: self.groups[v] for v in chosen}
        self.windows = sum(map(len, self.groups.values()))
        self.schedule = make_schedule(self.groups, config['video_batch_size_per_gpu'],
                                      config['tbptt_steps'], config['seed'], epoch, split == 'train')
        self.position = {i: (j, indices) for indices in self.groups.values() for j, i in enumerate(indices)}

    def __len__(self):
        return len(self.schedule)

    def window(self, index):
        row = self.rows[index]
        events = self.events.get(row)
        cfg = GCSTRConfig(sensor_width=int(row['sensor_width']), sensor_height=int(row['sensor_height']))
        fallbacks = 0
        epoch = self.epoch if self.split == 'train' else 0
        if self.split == 'train':
            params = sample_spatial_params(self.config['seed'], epoch, row['video_id'])
            events, fallbacks = apply_spatial_params(events, params,
                keyed_rng('jitter', self.config['seed'], epoch, row['video_id'], row['window_id']), cfg)
        t, x, y, p = events
        rng = keyed_rng('points', self.config['seed'], epoch, row['video_id'], row['window_id'])
        indices = np.sort(rng.choice(len(t), self.config['point_count'], replace=len(t)<self.config['point_count']))
        times = np.asarray(t[indices], dtype=np.float64)
        times -= times.min()
        times /= max(times.max(), 1.)
        points = np.stack((times, x[indices]/self.config['point_sensor_width'],
                           y[indices]/self.config['point_sensor_height'])).astype(np.float32)
        cached = {v: self.output/'cache'/v/(row['cache_key']+'.npy') for v in VIEWS}
        if self.split == 'test' and all(path.is_file() for path in cached.values()):
            images = {v: np.load(path, allow_pickle=False) for v, path in cached.items()}
        else:
            images = window_views(*events, int(row['start_us']), int(row['end_us']), cfg, VIEWS)
        j, sequence = self.position[index]
        gap = (int(row['start_us'])-int(self.rows[sequence[j-1]]['start_us']))/1e6 if j else 0.
        result = dict(events=torch.from_numpy(points), label=int(row['label_id']), video_id=row['video_id'],
                      window_id=str(row['window_id']), start_us=int(row['start_us']), end_us=int(row['end_us']),
                      gap_seconds=torch.tensor(gap, dtype=torch.float32), valid=True,
                      video_start=j == 0, video_end=j == len(sequence)-1, augmentation_fallbacks=fallbacks)
        for view, image in images.items():
            if image.shape != (3, 224, 224) or not np.isfinite(image).all():
                raise ValueError('Invalid image payload')
            image = (image-np.array([.485, .456, .406], np.float32)[:, None, None])/np.array([.229, .224, .225], np.float32)[:, None, None]
            result[view] = torch.from_numpy(image.astype(np.float32))
        return result

    def __getitem__(self, index):
        entry = self.schedule[index]
        batches = []
        for time_index in range(entry['start'], entry['stop']):
            samples = [self.window(self.groups[v][time_index]) if time_index < len(self.groups[v]) else None
                       for v in entry['videos']]
            template = next(s for s in samples if s is not None)
            for i, sample in enumerate(samples):
                if sample is None:
                    samples[i] = {k: torch.zeros_like(v) if torch.is_tensor(v) else
                                  '' if isinstance(v, str) else False if isinstance(v, bool) else 0
                                  for k, v in template.items()}
            batches.append(default_collate(samples))
        return dict(batches=batches, group_start=entry['group_start'], group_end=entry['group_end'])
