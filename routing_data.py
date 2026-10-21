"""Versioned frozen-expert evidence; explicit roles and causal sequence batches.

This module has no dependency on the original CUDA models or training entry.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

SCHEMA = 1
HEADS = ('fusion', 'point', 'cstr3', 'xt_p', 'ty_p')
STATS = ('log1p_events_per_pixel_second', 'log1p_event_count',
         'occupied_pixel_fraction', 'point_samples_per_raw_event')
OBSERVATIONS = tuple(f'{h}_{s}' for h in HEADS for s in ('normalized_entropy', 'top2_margin')) + (
    'frame_point_js_divergence_log2', 'image_pair_agreement') + STATS
ROLES = ('gate_train', 'gate_val', 'test', 'unassigned')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def row_key(row):
    return (row['dataset_id'], row['source_split'], row['video_id'], str(row['window_id']))


def video_key(row):
    return (row['dataset_id'], row['source_split'], row['video_id'])


def validate_identity(identity):
    required = ('dataset_id', 'class_to_idx', 'checkpoint_sha256', 'data_manifest_sha256',
                'code', 'normalization', 'sampling', 'export', 'expert_provenance', 'time_unit')
    if any(k not in identity for k in required):
        raise ValueError('Incomplete cache identity')
    classes = identity['class_to_idx']
    if len(classes) < 2 or sorted(classes.values()) != list(range(len(classes))):
        raise ValueError('Class mapping must be contiguous and contain >=2 classes')
    if identity['time_unit'] != 'microseconds':
        raise ValueError('Schema 1 timestamps are explicitly in microseconds')
    if not identity['checkpoint_sha256'] or not identity['code']:
        raise ValueError('Checkpoint and code identity required')


def validate_row(row, identity):
    required = ('dataset_id', 'source_split', 'role', 'video_id', 'window_id',
                'start_us', 'end_us', 'label_id', 'expert_fit_exposure', 'expert_selection_exposure')
    if any(k not in row for k in required):
        raise ValueError('Missing window metadata')
    if row['dataset_id'] != identity['dataset_id'] or row['role'] not in ROLES:
        raise ValueError('Dataset or role mismatch')
    if not 0 <= int(row['label_id']) < len(identity['class_to_idx']):
        raise ValueError('Invalid label')
    if int(row['end_us']) <= int(row['start_us']):
        raise ValueError('Invalid time interval')
    for field in ('expert_fit_exposure', 'expert_selection_exposure'):
        if row[field] not in ('seen', 'unseen', 'unknown'):
            raise ValueError('Exposure must be seen/unseen/unknown, not inferred from role')
    provenance = identity['expert_provenance']
    for phase in ('fit', 'selection'):
        known_seen = provenance.get(f'{phase}_source_splits', [])
        verified_unseen = provenance.get(f'verified_{phase}_unseen_source_splits', [])
        value = row[f'expert_{phase}_exposure']
        if row['source_split'] in known_seen and value != 'seen':
            raise ValueError('Expert-seen source cannot be relabeled as unseen/unknown')
        if value == 'unseen' and row['source_split'] not in verified_unseen:
            raise ValueError('Unverified expert exposure must be unknown, not unseen')


class EvidenceWriter:
    """One mmap array per tensor, row hashes and an atomic completion marker.

    A crashed export leaves a uniquely named .pending directory. Readers reject
    it; no old cache is overwritten. Expected keys make missing rows detectable.
    """
    def __init__(self, output, identity, expected_keys):
        validate_identity(identity)
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=self.output.name + '.pending-', dir=self.output.parent))
        self.identity = identity
        self.expected = {tuple(k) for k in expected_keys}
        if not self.expected or len(self.expected) != len(expected_keys):
            raise ValueError('Empty or duplicate expected windows')
        n, c = len(self.expected), len(identity['class_to_idx'])
        self.logits = np.lib.format.open_memmap(self.directory/'logits.npy', mode='w+', dtype=np.float32, shape=(n, len(HEADS), c))
        self.stats = np.lib.format.open_memmap(self.directory/'stats.npy', mode='w+', dtype=np.float32, shape=(n, len(STATS)))
        self.rows, self.seen = [], set()

    def append(self, rows, logits, stats):
        logits, stats = np.asarray(logits), np.asarray(stats)
        n, offset = len(rows), len(self.rows)
        if logits.shape != (n, len(HEADS), len(self.identity['class_to_idx'])) or stats.shape != (n, len(STATS)):
            raise ValueError('Five-head/class/stat alignment failure')
        if not np.isfinite(logits).all() or not np.isfinite(stats).all():
            raise ValueError('Nonfinite evidence')
        for row in rows:
            validate_row(row, self.identity)
            key = row_key(row)
            if key not in self.expected or key in self.seen:
                raise ValueError('Unexpected or duplicate window')
            self.seen.add(key)
        if offset+n > len(self.expected):
            raise ValueError('Too many rows')
        self.logits[offset:offset+n], self.stats[offset:offset+n] = logits, stats
        self.rows.extend(rows)

    def finish(self):
        if self.seen != self.expected:
            raise ValueError('Incomplete evidence export')
        self.logits.flush()
        self.stats.flush()
        (self.directory/'rows.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in self.rows))
        files = {name: digest(self.directory/name) for name in ('logits.npy', 'stats.npy', 'rows.jsonl')}
        manifest = dict(schema=SCHEMA, status='complete', heads=HEADS, statistics=STATS,
                        observations=OBSERVATIONS, temperature=1.0, identity=self.identity,
                        rows=len(self.rows), expected_keys_sha256=fingerprint(sorted(self.expected)), files=files)
        json_write(self.directory/'manifest.json', manifest)
        if self.output.exists():
            raise FileExistsError(self.output)
        os.rename(self.directory, self.output)
        return self.output


class EvidenceCache:
    def __init__(self, directory, expected_identity=None):
        self.directory = Path(directory)
        if '.pending-' in self.directory.name:
            raise ValueError('Pending cache cannot be read')
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        m = self.manifest
        if m.get('schema') != SCHEMA or m.get('status') != 'complete':
            raise ValueError('Unsupported/incomplete evidence cache')
        if tuple(m['heads']) != HEADS or tuple(m['statistics']) != STATS or tuple(m['observations']) != OBSERVATIONS or m['temperature'] != 1.0:
            raise ValueError('Evidence schema/order/temperature mismatch')
        self.identity = m['identity']
        validate_identity(self.identity)
        if expected_identity is not None and self.identity != expected_identity:
            raise ValueError('Evidence cache identity mismatch')
        if set(m['files']) != {'logits.npy', 'stats.npy', 'rows.jsonl'}:
            raise ValueError('Incomplete file inventory')
        for name, value in m['files'].items():
            if digest(self.directory/name) != value:
                raise ValueError(f'Corrupt evidence file: {name}')
        self.rows = [json.loads(s) for s in (self.directory/'rows.jsonl').read_text().splitlines()]
        self.logits = np.load(self.directory/'logits.npy', mmap_mode='r', allow_pickle=False)
        self.stats = np.load(self.directory/'stats.npy', mmap_mode='r', allow_pickle=False)
        n, c = m['rows'], len(self.identity['class_to_idx'])
        if len(self.rows) != n or self.logits.shape != (n, len(HEADS), c) or self.stats.shape != (n, len(STATS)):
            raise ValueError('Evidence tensor coverage/class count mismatch')
        if self.logits.dtype != np.float32 or self.stats.dtype != np.float32:
            raise ValueError('Evidence dtype mismatch')
        keys, videos = set(), {}
        for row in self.rows:
            validate_row(row, self.identity)
            key = row_key(row)
            if key in keys:
                raise ValueError('Duplicate window')
            keys.add(key)
            value = (int(row['label_id']), row['source_split'], row['role'], row['expert_fit_exposure'], row['expert_selection_exposure'])
            if videos.setdefault((row['dataset_id'], row['video_id']), value) != value:
                raise ValueError('One video crosses source splits, roles or labels')
        if fingerprint(sorted(keys)) != m['expected_keys_sha256']:
            raise ValueError('Missing/unexpected windows')
        for start in range(0, n, 4096):
            if not np.isfinite(self.logits[start:start+4096]).all() or not np.isfinite(self.stats[start:start+4096]).all():
                raise ValueError('Nonfinite cache')
        self.fingerprint = fingerprint(m)


def observations(logits, stats):
    """Only inference-available, class-count-independent features; no labels/IDs."""
    if logits.shape[-2] != len(HEADS) or stats.shape[-1] != len(STATS):
        raise ValueError('Observation schema mismatch')
    if logits.shape[-1] < 2:
        raise ValueError('At least two classes required')
    lp = logits.log_softmax(-1)
    p = lp.exp()
    entropy = -(p*lp).sum(-1)/math.log(logits.shape[-1])
    top = p.topk(2, dim=-1).values
    confidence = torch.stack((entropy, top[..., 0]-top[..., 1]), -1).flatten(-2)
    frame, point = p[..., 2:, :].mean(-2), p[..., 1, :]
    middle = (frame+point)/2
    tiny = torch.finfo(p.dtype).tiny
    js = ((frame*(frame.clamp_min(tiny).log()-middle.clamp_min(tiny).log())).sum(-1)
          +(point*(point.clamp_min(tiny).log()-middle.clamp_min(tiny).log())).sum(-1))/(2*math.log(2))
    pred = p[..., 2:, :].argmax(-1)
    agree = ((pred[..., 0] == pred[..., 1]).float()+(pred[..., 0] == pred[..., 2]).float()
             +(pred[..., 1] == pred[..., 2]).float())/3
    result = torch.cat((confidence, js.unsqueeze(-1), agree.unsqueeze(-1), stats), -1)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite Gate observation')
    return result


class ObservationNormalizer(torch.nn.Module):
    def __init__(self, dimensions=len(OBSERVATIONS)):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimensions))
        self.register_buffer('scale', torch.ones(dimensions))

    def fit(self, data):
        if data.role != 'gate_train':
            raise ValueError('Normalizer must be fitted only on gate_train')
        total = torch.zeros_like(self.mean, dtype=torch.float64)
        squares, count = total.clone(), 0
        for idx in data.indices:
            if data.cache.rows[idx]['source_split'].lower() == 'test':
                raise ValueError('Original Test cannot fit input normalization')
            x = observations(torch.tensor(np.array(data.cache.logits[idx])), torch.tensor(np.array(data.cache.stats[idx]))).double()
            total += x
            squares += x*x
            count += 1
        mean = total/count
        scale = (squares/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
        self.mean.copy_(mean.float())
        self.scale.copy_(scale.float())
        return self

    def forward(self, x):
        return (x-self.mean)/self.scale


class RoutingDataset(Dataset):
    def __init__(self, cache, role, *, sequence=False):
        if role not in ROLES:
            raise ValueError('Unknown role')
        self.cache, self.role, self.sequence = cache, role, sequence
        self.indices = [i for i, r in enumerate(cache.rows) if r['role'] == role]
        if not self.indices:
            raise ValueError(f'No windows for role {role}; supply an explicit role mapping')
        groups = {}
        for idx in self.indices:
            groups.setdefault(video_key(cache.rows[idx]), []).append(idx)
        self.groups = []
        for key in sorted(groups):
            ids = sorted(groups[key], key=lambda i: int(cache.rows[i]['start_us']))
            starts = [int(cache.rows[i]['start_us']) for i in ids]
            if len(starts) != len(set(starts)):
                raise ValueError('Duplicate window start within video')
            self.groups.append(ids)

    def __len__(self):
        return len(self.groups) if self.sequence else len(self.indices)

    def __getitem__(self, index):
        ids = self.groups[index] if self.sequence else [self.indices[index]]
        rows = [self.cache.rows[i] for i in ids]
        starts = torch.tensor([int(r['start_us']) for r in rows], dtype=torch.float64)
        ends = torch.tensor([int(r['end_us']) for r in rows], dtype=torch.float64)
        dt, overlap = torch.zeros(len(ids)), torch.zeros(len(ids))
        if self.sequence and len(ids)>1:
            dt[1:] = ((starts[1:]-starts[:-1])/1e6).float()
            overlap[1:] = ((ends[:-1]-starts[1:]).clamp_min(0)/torch.minimum(ends[:-1]-starts[:-1], ends[1:]-starts[1:])).float()
        logits = torch.tensor(np.array(self.cache.logits[ids]))
        stats = torch.tensor(np.array(self.cache.stats[ids]))
        return dict(logits=logits, observations=observations(logits, stats), dt=dt, overlap=overlap,
                    labels=torch.tensor([int(r['label_id']) for r in rows]), rows=rows)


def collate_sequences(items):
    """All modes use [B,T,...]; window batches simply have T=1."""
    length = max(len(i['rows']) for i in items)
    result = {}
    for key in ('logits', 'observations', 'dt', 'overlap', 'labels'):
        sample = items[0][key]
        value = torch.zeros((len(items), length, *sample.shape[1:]), dtype=sample.dtype)
        for b, item in enumerate(items):
            value[b, :len(item['rows'])] = item[key]
        result[key] = value
    result['mask'] = torch.tensor([[t<len(item['rows']) for t in range(length)] for item in items])
    result['rows'] = [item['rows'] for item in items]
    return result


def validate_training_roles(train, val, *, allow_seen_experts=False):
    if train.role != 'gate_train' or val.role != 'gate_val':
        raise ValueError('Fit requires explicit gate_train and gate_val roles')
    if train.cache.identity != val.cache.identity:
        raise ValueError('Train/validation expert identities differ')
    train_videos = {r['video_id'] for i in train.indices for r in [train.cache.rows[i]]}
    val_videos = {r['video_id'] for i in val.indices for r in [val.cache.rows[i]]}
    if train_videos & val_videos:
        raise ValueError('Gate training/validation videos overlap')
    for data in (train, val):
        if any(data.cache.rows[i]['source_split'].lower() == 'test' for i in data.indices):
            raise ValueError('Original Test cannot fit/select Gate or normalization')
    exposure = {val.cache.rows[i][k] for i in val.indices
                for k in ('expert_fit_exposure', 'expert_selection_exposure')}
    if exposure != {'unseen'} and not allow_seen_experts:
        raise ValueError('Gate validation is expert-seen/unknown; explicit exploratory override required')
