"""DailyAction-DVS full-event cache and EventMamba-compatible base point windows."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
from prepare_chl import sha, write_json

CLASSES = ('bend', 'climb', 'fall', 'get up', 'jump', 'lie', 'lift', 'pick', 'run', 'sit', 'stand', 'walk')
PROTOCOL = 'https://github.com/rhwxmx/EventMamba/blob/main/dataprocess/generate_daily.py'


def load_aedat(path, repair=False, return_audit=False):
    """Decode every DVS128 AEDAT2 record, including the upstream omitted last one."""
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'#!AER-DAT2.0':
            raise ValueError('Expected AEDAT 2.0: '+str(path))
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(pos)
                break
        payload = f.read()
    if not payload or len(payload) % 8:
        raise ValueError('Empty or truncated AEDAT2 payload: '+str(path))
    words = np.frombuffer(payload, dtype='>u4').reshape(-1, 2)
    addr, t = words[:, 0], words[:, 1].astype(np.int64)
    if (addr & 0xffff8000).any():
        raise ValueError('Non-DVS128 address bits; explicit policy required: '+str(path))
    backward = np.flatnonzero(np.diff(t)<0)
    audit = dict(backward_indices=backward.tolist(), wrap_indices=[], reordered_events=0)
    if len(backward) and not repair:
        raise ValueError('Nonmonotonic timestamp; no implicit repair: '+str(path))
    wrap = np.diff(t) < -(2**31)
    audit['wrap_indices'] = np.flatnonzero(wrap).tolist()
    t += np.concatenate(([0], np.cumsum(wrap, dtype=np.int64))) * 2**32
    small = np.flatnonzero(np.diff(t)<0)
    audit['small_backward_us'] = (-np.diff(t)[small]).tolist()
    if len(small) and max(audit['small_backward_us']) > 1000:
        raise ValueError('Unexpected disorder larger than 1 ms: '+str(path))
    t -= t[0]
    events = np.column_stack(((addr & 0xfe) >> 1, (addr & 0x7f00) >> 8, t, addr & 1)).astype(np.int64)
    order = np.argsort(t, kind='stable') if len(small) else np.arange(len(t))
    # Keep the identity of the physical final record for upstream decoder compatibility.
    if order[-1] != len(t)-1 or order[0] != 0:
        raise ValueError('Timestamp repair changes recording endpoints: '+str(path))
    audit['reordered_events'] = int(np.count_nonzero(order != np.arange(len(t))))
    events = events[order]
    return (events, audit) if return_audit else events


def windows(t):
    """t is the EventMamba decoder view (all but final physical record)."""
    if not len(t): return
    for start in range(int(t[0]), int(t[-1])-1500000, 500000):
        end = start+1500000
        yield start, end, int(np.searchsorted(t, start, 'left')), int(np.searchsorted(t, end, 'right'))-1


def sample_points(events, count, rng):
    if len(events) <= 10000: return None
    idx = np.arange(len(events))
    if count > len(events): idx = rng.choice(idx, count, replace=True)
    else:
        rng.shuffle(idx)
        idx = idx[:count]
    idx.sort()
    points = events[idx][:, [2, 0, 1]].copy()
    points[:, 0] //= 1000
    points[:, 0] -= points[:, 0].min()
    duration = points[:, 0].max()
    if duration == 0: raise ValueError('Zero sampled millisecond duration')
    result = points.astype(np.float64)
    result[:, 0] /= duration
    result[:, 1:] /= 128
    return result.astype(np.float32)


def read_records(root, seed):
    found = sorted(p.name for p in root.iterdir() if p.is_dir())
    if found != list(CLASSES): raise ValueError('Unexpected class directories: '+str(found))
    records = []
    rng = random.Random(seed)
    for label, name in enumerate(CLASSES):
        files = sorted((root/name).glob('*.aedat'))
        if not files: raise ValueError('Empty class: '+name)
        train_indices = set(rng.sample(range(len(files)), len(files)*4//5))
        for i, path in enumerate(files):
            records.append(dict(filename=path.relative_to(root).as_posix(), label=label,
                sample_id=f'{label:02d}_{path.stem}', subject=path.stem[:2],
                split='train' if i in train_indices else 'test'))
    for r in records:
        r['event_file'] = f'{r["split"]}/{r["sample_id"]}.npy'
    if len({r['sample_id'] for r in records}) != len(records): raise ValueError('Duplicate sample ID')
    return records


def prepare(args):
    root, out = args.root.resolve(), args.output.resolve()
    records = read_records(root, args.seed)
    if args.split_manifest:
        saved = json.loads(args.split_manifest.read_text())
        if saved != records: raise ValueError('Frozen split differs from input files/seed/protocol')
        records = saved
    if out == root or root in out.parents: raise ValueError('Output must be outside source')
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(dataset='DailyAction-DVS', classes=12, class_names=CLASSES, width=128, height=128,
        timestamp_policy='unwrap uint32; stable sort local disorder <=1ms; audit all changes; preserve source files',
        split_manifest_sha256=sha(args.split_manifest) if args.split_manifest else None,
        timestamp_unit='us', window_us=1500000, step_us=500000, points=2048, seed=args.seed,
        minimum_raw_events_exclusive=10000, radius_filter=False, offline_point_augmentation=False,
        split_policy='sorted class/file order; Python random.Random(seed) sample floor(80%) training recordings per class; remainder test; not subject-disjoint',
        point_time='integer microseconds //1000 then sampled min/max; no epsilon',
        window_boundary='strict end<last; last <=end index excluded',
        decoder_compatibility='cache all physical records; windows use events[:-1] to match upstream decoder',
        frame_events='all events in same eligible point-window slice',
        adaptation='fixed split and window RNG; no offline augmentation copies; no author checkpoint identity claim',
        source_protocol=PROTOCOL, code_sha256=sha(Path(__file__)), numpy=np.__version__,
        smoke_per_split=args.smoke_per_split)
    write_json(out/'config.json', cfg)
    write_json(out/'split_records.json', records)
    audit = {}
    for split in ('train', 'test'):
        rs = [r for r in records if r['split']==split]
        audit[split] = dict(recordings=len(rs), per_class={n:sum(r['label']==i for r in rs) for i,n in enumerate(CLASSES)},
            subjects=sorted({r['subject'] for r in rs}))
    audit['shared_subjects'] = sorted(set(audit['train']['subjects']) & set(audit['test']['subjects']))
    write_json(out/'split_audit.json', audit)
    for split in ('train', 'test'):
        with (out/(split+'_files.jsonl')).open('w') as f:
            for record in records:
                if record['split'] == split: f.write(json.dumps(record)+'\n')
    if args.split_only:
        write_json(out/'status.json', dict(state='split_only', scope='full', preprocessing_complete=False))
        return
    chosen = {s:[r for r in records if r['split']==s] for s in ('train','test')}
    if args.smoke_per_split:
        chosen = {s:rs[:args.smoke_per_split] for s,rs in chosen.items()}
    write_json(out/'samples.json', [r for rs in chosen.values() for r in rs])
    write_json(out/'status.json', dict(state='running'))
    coverage = {}
    repairs = []
    try:
        for split, rs in chosen.items():
            (out/split).mkdir()
            with h5py.File(out/(split+'.h5'), 'x') as h, (out/(split+'_windows.jsonl')).open('w') as manifest:
                h.attrs['complete'] = False
                h.attrs['config_sha256'] = sha(out/'config.json')
                data = h.create_dataset('data', (0,2048,3), maxshape=(None,2048,3), dtype='f4', chunks=(1,2048,3))
                labels = h.create_dataset('label', (0,), maxshape=(None,), dtype='i2')
                marks = h.create_dataset('mark', (0,), maxshape=(None,), dtype='i4')
                n = 0
                excluded = []
                invalid_windows = 0
                for mark, record in enumerate(rs):
                    src = root/record['filename']
                    events, timestamp_audit = load_aedat(src, repair=True, return_audit=True)
                    if timestamp_audit['backward_indices']:
                        repairs.append(dict(filename=record['filename'], **timestamp_audit))
                    dest = out/record['event_file']
                    np.save(dest, events, allow_pickle=False)
                    write_json(dest.with_suffix('.json'), dict(events=len(events), source=str(src),
                        source_sha256=sha(src), cache_sha256=sha(dest), timestamp_origin='first raw event', timestamp_audit=timestamp_audit))
                    valid_count = 0
                    window_count = 0
                    for wid, (start,end,left,right) in enumerate(windows(events[:-1,2])):
                        window_count += 1
                        seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{record["sample_id"]}:{wid}'.encode()).digest()[:4], 'little')
                        points = sample_points(events[left:right], 2048, np.random.RandomState(seed))
                        row = dict(**record, mark=mark, window_id=wid, start_us=start, end_us=end,
                            left=left, right=right, raw_events=max(0,right-left), valid=points is not None,
                            point_row=n if points is not None else None)
                        manifest.write(json.dumps(row)+'\n')
                        if points is not None:
                            for ds in (data,labels,marks): ds.resize(n+1, axis=0)
                            data[n] = points; labels[n] = record['label']; marks[n] = mark
                            n += 1; valid_count += 1
                    invalid_windows += window_count-valid_count
                    if not valid_count:
                        excluded.append(dict(**record, reason='short_recording' if window_count==0 else 'insufficient_events', duration_us=int(events[-2,2]), candidate_windows=window_count))
                    progress = dict(state='running', split=split, recordings=mark+1, total=len(rs), windows=n)
                    write_json(out/'status.json', progress)
                    print(json.dumps(progress), flush=True)
                if n == 0: raise ValueError('Empty split')
                h.attrs['complete'] = True
                coverage[split] = dict(recordings=len(rs), eligible_recordings=len(rs)-len(excluded), valid_windows=n, invalid_windows=invalid_windows, excluded_recordings=excluded)
                write_json(out/'coverage.json', coverage)
                write_json(out/'timestamp_repairs.json', repairs)
        write_json(out/'status.json', dict(state='complete', scope='smoke' if args.smoke_per_split else 'full'))
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, help='Frozen split_records.json to verify and reuse')
    p.add_argument('--split-only', action='store_true', help='Save full recording splits without event preprocessing')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke-per-split', type=int, default=0)
    args = p.parse_args()
    if args.smoke_per_split < 0: p.error('Negative smoke budget')
    prepare(args)
