"""Read-only input audit / window inventory; no filtering, sampling or image cache.

The >10000 rule is only a reported candidate, not an approved filter.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

WINDOW_US = 1_000_000
STRIDE_US = 500_000


def bounds(t):
    if len(t) == 0 or np.any(t[1:] < t[:-1]):
        raise ValueError('Empty or unordered timestamps')
    first, last = int(t[0]), int(t[-1])
    short = last-first < WINDOW_US
    starts = [first] if short else range(first, last-WINDOW_US+1, STRIDE_US)
    result = []
    for i, start in enumerate(starts):
        end = start+WINDOW_US
        left, right = np.searchsorted(t, [start, end], side='left')
        result.append(dict(window_id=i, start_us=start, end_us=end, left=int(left), right=int(right),
                           event_count=int(right-left), short_sample=short))
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(task):
    root, row = task
    result = dict(sample_id=row['sample_id'], split=row['split'], label_id=int(row['label_id']),
                  npz_relpath=row['npz_relpath'])
    try:
        path = Path(root)/row['npz_relpath']
        before = path.stat()
        with np.load(path, allow_pickle=False) as data:
            if not {'t', 'x', 'y', 'p'}.issubset(data.files):
                raise ValueError('Missing t/x/y/p fields')
            # This pass examines t only; prior full-payload audit is not rebranded
            # as a fresh polarity/coordinate validation here.
            t = data['t']
        if t.ndim != 1 or t.dtype.kind not in 'iu' or len(t) != int(row['event_count']):
            raise ValueError('Timestamp shape/type/count differs from manifest')
        if not len(t) or int(t[0]) != int(row['start_time_usec']) or int(t[-1]) != int(row['end_time_usec']):
            raise ValueError('Timestamp endpoints differ from manifest')
        if int(t[-1])-int(t[0]) != int(row['duration_usec']) or int(t[-1])+WINDOW_US > np.iinfo(np.int64).max:
            raise ValueError('Invalid duration or timestamp range')
        windows = bounds(t)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Input changed during inspection')
        result.update(status='ok', windows=windows, event_count=len(t),
                      duration_us=int(t[-1])-int(t[0]), trailing_uncovered_events=len(t)-windows[-1]['right'],
                      source_bytes=after.st_size, source_mtime_ns=after.st_mtime_ns)
    except Exception as error:
        result.update(status='error', error=str(error))
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def summarize(records):
    summary = {}
    for split in ('train', 'test'):
        rows = [r for r in records if r['split'] == split and r['status'] == 'ok']
        counts = np.array([w['event_count'] for r in rows for w in r['windows']], dtype=np.int64)
        if not len(counts):
            summary[split] = dict(samples=len(rows), windows=0)
            continue
        thresholds = {}
        for threshold in (0, 2048, 10000):
            empty = [r for r in rows if not any(w['event_count'] > threshold for w in r['windows'])]
            thresholds[str(threshold)] = dict(comparison='strictly_greater_than',
                retained_windows=int((counts > threshold).sum()),
                samples_without_windows=[r['sample_id'] for r in empty],
                affected_classes=sorted({r['label_id'] for r in empty}))
        summary[split] = dict(samples=len(rows), windows=len(counts), short_samples=sum(r['windows'][0]['short_sample'] for r in rows),
            window_event_quantiles=dict(zip(['min', 'p01', 'p05', 'p50', 'p95', 'max'],
                np.quantile(counts, [0, .01, .05, .5, .95, 1]).tolist())),
            raw_events=sum(r['event_count'] for r in rows),
            trailing_uncovered_events=sum(r['trailing_uncovered_events'] for r in rows),
            threshold_candidates=thresholds)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int, help='Smoke only: first N samples per split')
    args = p.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        p.error('Positive worker/limit values required')
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_rows, sources = [], {}
    ids, paths = set(), set()
    for split in ('train', 'test'):
        path = args.root/'splits'/f'{split}_npz.csv'
        sources[str(path)] = digest(path)
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if {int(r['label_id']) for r in rows} != set(range(200)):
            raise ValueError('Expected all 200 classes')
        for row in rows:
            if row['split'] != split or row['sample_id'] in ids or row['npz_relpath'] in paths:
                raise ValueError('Duplicate/cross-split sample')
            ids.add(row['sample_id']); paths.add(row['npz_relpath'])
            rel = Path(row['npz_relpath'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe input path')
        all_rows.extend(rows[:args.limit] if args.limit else rows)
    config = dict(protocol='dailydvs200_full_event_windows_audit_v1', window_us=WINDOW_US, stride_us=STRIDE_US,
        short_sample='one nominal 1s half-open window; no fabricated events', tail='no extra partial window',
        filter_applied=False, threshold_candidates=[0, 2048, 10000], augmentation=False,
        sampled_points=False, image_cache=False, timestamp_unit='us', workers=args.workers,
        input_manifests=sources, code_sha256=digest(__file__), limit=args.limit,
        audit_scope='timestamp arrays and field presence, not full x/y/p payload audit')
    write_json(args.output/'config.json', config)
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output/'samples.jsonl').open('x') as stream:
        for index, record in enumerate(pool.map(inspect, ((str(args.root), r) for r in all_rows), chunksize=1), 1):
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            if index % 50 == 0 or index == len(all_rows) or index == 1:
                stream.flush()
                progress = dict(status='running', completed=index, total=len(all_rows),
                    errors=sum(r['status'] == 'error' for r in records), seconds=time.monotonic()-started,
                    pid=os.getpid())
                write_json(args.output/'status.json', progress)
                print(json.dumps(progress), flush=True)
    if any(digest(path) != value for path, value in sources.items()):
        raise RuntimeError('Input manifests changed during audit')
    errors = [r for r in records if r['status'] != 'ok']
    report = dict(status='needs_review' if errors else 'complete', splits=summarize(records), errors=errors,
                  filter_applied=False, source_unchanged=True, seconds=time.monotonic()-started)
    write_json(args.output/'summary.json', report)
    write_json(args.output/'status.json', dict(status=report['status'], completed=len(records), total=len(all_rows),
               errors=len(errors), seconds=report['seconds'], pid=os.getpid()))
    print(json.dumps(dict(event='finished', status=report['status'], samples=len(records))), flush=True)


if __name__ == '__main__':
    main()
