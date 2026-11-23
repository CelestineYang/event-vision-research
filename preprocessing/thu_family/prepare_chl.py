"""Portable THU-EACT-50-CHL preprocessing; see README.md before running."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def splits(root, train, test):
    result, seen = [], set()
    for split, path in [('train', train), ('test', test)]:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            name, label = line.rsplit(None, 1)
            # Official lists use ../DVS-action-data-npy/<basename>.
            name = name.replace('\\', '/').split('/')[-1]
            if not name.endswith('.npy') or name in seen:
                raise ValueError('Invalid/duplicate/cross-split sample: ' + name)
            label = int(label)
            if not 0 <= label < 50:
                raise ValueError('Expected official labels 0..49')
            source = root / name
            if not source.is_file():
                raise FileNotFoundError(source)
            seen.add(name)
            result.append(dict(split=split, sample_id=source.stem, filename=name, label=label))
    for split in ('train', 'test'):
        if not any(r['split'] == split for r in result):
            raise ValueError('Empty split')
    return result


def load_events(path):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a):
        raise ValueError('Expected nonempty [N,4] x,y,t,p')
    if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError('Expected finite integer-valued published events')
    if (a[:, :3] < 0).any() or (a[:, 0] >= 346).any() or (a[:, 1] >= 260).any():
        raise ValueError('Invalid CHL coordinates/timestamps')
    if not np.isin(a[:, 3], [0, 1]).all() or (np.diff(a[:, 2].astype(np.float64)) < 0).any():
        raise ValueError('Invalid polarity or unsorted timestamps; no implicit repair')
    return a


def official_windows(t):
    """Same indices as EventMamba uti.get_window_index for sorted nonempty t.

    Strict end < last; end index is last event <= boundary, then excluded by
    Python slicing. Preserve this upstream off-by-one for checkpoint alignment.
    """
    first, last = int(t[0]), int(t[-1])
    for start in range(first, last - 1_000_000, 500_000):
        end = start + 1_000_000
        left = int(np.searchsorted(t, start, side='left'))
        right = int(np.searchsorted(t, end, side='right')) - 1
        yield start, end, left, right


def point_window(events, rng):
    import open3d as o3d
    # Official code casts t,x,y to int64 before radius filtering.
    points = events[:, [2, 0, 1]].astype(np.int64)
    if not len(points):
        return None, 0
    shifted = points.copy()
    shifted[:, 0] -= shifted[:, 0].min()
    if shifted[:, 0].max() == 0:
        return None, 0
    normalized = shifted.astype(np.float64)
    normalized[:, 0] /= shifted[:, 0].max()
    normalized[:, 1] /= 346
    normalized[:, 2] /= 260
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normalized))
    _, indices = cloud.remove_radius_outlier(nb_points=5, radius=0.01)
    points = points[indices]
    count = len(points)
    if count < 100:
        return None, count
    if count < 2048:
        indices = rng.choice(np.arange(count), size=2048, replace=True)
    else:
        indices = np.arange(count)
        rng.shuffle(indices)
        indices = indices[:2048]
    indices.sort()
    points = points[indices].copy()
    points[:, 0] -= points[:, 0].min()
    if points[:, 0].max() == 0:
        return None, count
    result = points.astype(np.float64)
    result[:, 0] /= points[:, 0].max()
    result[:, 1] /= 346
    result[:, 2] /= 260
    return result.astype(np.float32), count


def prepare(args):
    import h5py
    import open3d as o3d
    root, out = args.root.resolve(), args.output.resolve()
    train, test = args.train_list or root/'train.txt', args.test_list or root/'test.txt'
    records = splits(root, train, test)
    if out.exists():
        raise FileExistsError('Use a new output directory; partial runs are retained: '+str(out))
    if out == root or root in out.parents:
        raise ValueError('Output must be outside raw data directory')
    out.mkdir(parents=True)
    config = dict(dataset='THU-EACT-50-CHL', timestamp_unit='us', width=346, height=260,
                  seed=args.seed, points=2048, window_us=1000000, step_us=500000,
                  window_boundary='EventMamba strict end<last; last <=end index excluded',
                  radius=0.01, min_neighbors=5, minimum_filtered_events=100,
                  numpy=np.__version__, open3d=o3d.__version__, train_list_sha256=sha(train),
                  test_list_sha256=sha(test), code_sha256=sha(Path(__file__)),
                  adaptation='per-window deterministic RNG; full raw polarity retained for frames',
                  checkpoint_compatibility='input protocol only; actual checkpoint not inspected',
                  frame_events='all raw events in SAME official index slice, no radius filter',
                  smoke_per_split=args.smoke_per_split)
    write_json(out/'config.json', config)
    write_json(out/'samples.json', records)
    write_json(out/'split_audit.json', {
        split: dict(videos=sum(r['split']==split for r in records),
                    labels=sorted({r['label'] for r in records if r['split']==split}),
                    missing_labels=sorted(set(range(50))-{r['label'] for r in records if r['split']==split}))
        for split in ('train','test')})
    try:
        for split in ('test', 'train'):  # same split processing order as upstream
            chosen = [r for r in records if r['split'] == split]
            if args.smoke_per_split:
                chosen = chosen[:args.smoke_per_split]
            folder = out/split
            folder.mkdir()
            with h5py.File(out/(split+'.h5'), 'x') as h5, (out/(split+'_windows.jsonl')).open('w') as manifest:
                h5.attrs['complete'] = False
                h5.attrs['config_sha256'] = sha(out/'config.json')
                data = h5.create_dataset('data', shape=(0,2048,3), maxshape=(None,2048,3), dtype='f4', chunks=(1,2048,3))
                labels = h5.create_dataset('label', shape=(0,), maxshape=(None,), dtype='i2')
                marks = h5.create_dataset('mark', shape=(0,), maxshape=(None,), dtype='i4')
                n = 0
                for mark, sample in enumerate(chosen):
                    src = root/sample['filename']
                    events = load_events(src)
                    # Store full published sequence once; overlapping windows share it.
                    dest = folder/(sample['sample_id']+'.npy')
                    np.save(dest, events, allow_pickle=False)
                    restored = np.load(dest, mmap_mode='r')
                    if restored.dtype != events.dtype or not np.array_equal(events, restored):
                        raise RuntimeError('Raw cache round-trip failed')
                    write_json(folder/(sample['sample_id']+'.json'), dict(**sample, mark=mark,
                               events=len(events), source_sha256=sha(src), cache_sha256=sha(dest)))
                    for window_id, (start,end,left,right) in enumerate(official_windows(events[:,2])):
                        seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{sample["sample_id"]}:{window_id}'.encode()).digest()[:4], 'little')
                        points, filtered = point_window(events[left:right], np.random.RandomState(seed))
                        row = dict(**sample, mark=mark, window_id=window_id, start_us=start, end_us=end,
                                   left=left, right=right, raw_events=max(0,right-left), filtered_events=filtered,
                                   point_row=n if points is not None else None,
                                   valid=points is not None, event_file=f'{split}/{dest.name}')
                        manifest.write(json.dumps(row)+'\n')
                        if points is not None:
                            for ds in (data, labels, marks): ds.resize(n+1, axis=0)
                            data[n], labels[n], marks[n] = points, sample['label'], mark
                            n += 1
                    print(json.dumps(dict(split=split, video=mark+1, total=len(chosen), valid_windows=n)), flush=True)
                    write_json(out/'status.json', dict(state='running', split=split, videos=mark+1, windows=n))
                if n == 0:
                    raise ValueError('No eligible windows in '+split)
                h5.attrs['complete'] = True
        write_json(out/'status.json', dict(state='complete', scope='smoke' if args.smoke_per_split else 'full'))
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=repr(exc)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Directory containing original CHL NPY files')
    p.add_argument('--train-list', type=Path)
    p.add_argument('--test-list', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke-per-split', type=int, default=0)
    a = p.parse_args()
    if a.smoke_per_split < 0: p.error('smoke-per-split must be nonnegative')
    prepare(a)


if __name__ == '__main__':
    main()
