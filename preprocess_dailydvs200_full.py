"""Full-event mmap and three full-window gCSTR views; CPU-only, resumable.

Retains every window from the approved 1s/0.5s inventory, including short samples.
No downsampling, threshold filtering, augmentation, training, or raw-file edits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from gcstr_representations import GCSTRConfig
from prepare_dailydvs200_windows import bounds
from ucf101_window_data import VIEWS, code_hash, sha, window_views, write_csv, write_json

FIELDS = ('t', 'x', 'y', 'p')


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    tmp.replace(path)


def window_rows(row, record, source_sha, root):
    result = []
    for window in record['windows']:
        result.append(dict(video_id=row['sample_id'], sample_id=row['sample_id'], split=row['split'],
            label_id=int(row['label_id']), class_name=row['class_name'],
            window_id=window['window_id'], start_us=window['start_us'], end_us=window['end_us'],
            event_count=window['event_count'], short_sample=window['short_sample'],
            sensor_width=int(row['sensor_width']), sensor_height=int(row['sensor_height']),
            cache_key=f"{row['sample_id']}_w{window['window_id']:04d}",
            source_path=str(Path(root)/row['npz_relpath']), source_sha256=source_sha))
    return result


def prepare_one(task):
    root, output, row, record = task
    output = Path(output)
    folder = output/'event_store/videos'/row['sample_id']
    metadata_path = folder/'metadata.json'
    source = Path(root)/row['npz_relpath']
    stat = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (record['source_bytes'], record['source_mtime_ns']):
        raise RuntimeError(f"Source changed since inventory: {row['sample_id']}")
    source_sha = sha(source)
    rows = window_rows(row, record, source_sha, root)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        expected = {str(w['window_id']): {k: w[k] for k in ('start_us', 'end_us', 'event_count', 'left', 'right')}
                    for w in record['windows']}
        if metadata['source_sha256'] != source_sha or metadata['windows'] != expected:
            raise RuntimeError('Committed sample identity changed')
        if sha(folder/'events.npy') != metadata['events_sha256']:
            raise RuntimeError('Committed mmap checksum mismatch')
        for relative, expected_sha in metadata['image_sha256'].items():
            if sha(output/'experiment'/relative) != expected_sha:
                raise RuntimeError('Committed image checksum mismatch')
        if len(metadata['image_sha256']) != len(rows)*3:
            raise RuntimeError('Incomplete committed image coverage')
        return dict(rows=rows, metadata_sha256=sha(metadata_path), source_sha256=source_sha,
                    sample_id=row['sample_id'], events=metadata['events'], resumed=True)
    with np.load(source, allow_pickle=False) as data:
        arrays = {k: data[k] for k in FIELDS}
    n = len(arrays['t'])
    if n != record['event_count'] or any(a.ndim != 1 or len(a) != n or a.dtype.kind not in 'iub' for a in arrays.values()):
        raise ValueError('Invalid event arrays')
    cfg = GCSTRConfig(sensor_width=int(row['sensor_width']), sensor_height=int(row['sensor_height']))
    if (np.any(arrays['t'][1:] < arrays['t'][:-1]) or not np.isin(arrays['p'], [0, 1]).all() or
        np.any(arrays['x'] >= cfg.sensor_width) or np.any(arrays['y'] >= cfg.sensor_height) or
        np.any(arrays['x'] < 0) or np.any(arrays['y'] < 0)):
        raise ValueError('Invalid coordinates/polarity/timestamp order')
    if bounds(arrays['t']) != record['windows']:
        raise ValueError('Window inventory does not match source timestamps')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder/'events.npy.tmp'
    dtype = np.dtype([(k, arrays[k].dtype) for k in FIELDS], align=False)
    mmap = np.lib.format.open_memmap(temporary, mode='w+', dtype=dtype, shape=(n,))
    for k in FIELDS:
        mmap[k] = arrays[k]
    mmap.flush()
    del mmap
    mmap = np.load(temporary, mmap_mode='r', allow_pickle=False)
    for k in FIELDS:
        if mmap[k].dtype != arrays[k].dtype or not np.array_equal(mmap[k], arrays[k]):
            raise RuntimeError('Mmap changed raw event values/dtypes')
    del mmap
    temporary.replace(folder/'events.npy')
    image_hashes = {}
    for window, manifest_row in zip(record['windows'], rows):
        left, right = window['left'], window['right']
        if right <= left:
            raise ValueError('Empty window requires explicit protocol decision, not silent removal')
        images = window_views(*(arrays[k][left:right] for k in FIELDS),
                              window['start_us'], window['end_us'], cfg, VIEWS)
        for view, image in images.items():
            if image.shape != (3, 224, 224) or not np.isfinite(image).all():
                raise ValueError('Invalid projected image')
            image = np.asarray(image, dtype=np.float32)
            relative = Path('cache')/view/(manifest_row['cache_key']+'.npy')
            path = output/'experiment'/relative
            save_array(path, image)
            loaded = np.load(path, allow_pickle=False)
            if not np.array_equal(loaded, image):
                raise RuntimeError('Saved projection differs from rendering')
            image_hashes[str(relative)] = sha(path)
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError('Source changed during preprocessing')
    metadata = dict(video_id=row['sample_id'], split=row['split'], source_path=str(source), source_sha256=source_sha,
        events=n, field_dtypes={k: str(arrays[k].dtype) for k in FIELDS}, events_sha256=sha(folder/'events.npy'),
        windows={str(w['window_id']): {k: w[k] for k in ('start_us', 'end_us', 'event_count', 'left', 'right')}
                 for w in record['windows']}, image_sha256=image_hashes,
        verification='whole-stream dtype/value equality; every saved projection reloaded and compared exactly')
    # Only publish a sample commit after its mmap AND all images pass validation.
    write_json(metadata_path, metadata)
    return dict(rows=rows, metadata_sha256=sha(metadata_path), source_sha256=source_sha,
                sample_id=row['sample_id'], events=n, resumed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200'))
    p.add_argument('--inventory', type=Path, default=Path('/remote-home/datasets/event_project_cache/dailydvs200_windows_audit_1s_stride05_20261025_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Train first sample and all short samples; never training')
    args = p.parse_args()
    if args.workers < 1:
        p.error('workers must be positive')
    records = [json.loads(line) for line in (args.inventory/'samples.jsonl').read_text().splitlines()]
    report = json.loads((args.inventory/'summary.json').read_text())
    inventory_config = json.loads((args.inventory/'config.json').read_text())
    if report['status'] != 'complete' or report['errors'] or inventory_config['limit'] is not None:
        raise ValueError('Require successful full inventory')
    for path, expected in inventory_config['input_manifests'].items():
        if sha(path) != expected:
            raise ValueError('Input manifest changed since inventory')
    rows = []
    for split in ('train', 'test'):
        with (args.root/'splits'/f'{split}_npz.csv').open(newline='') as stream:
            part = list(csv.DictReader(stream))
        if any(r['split'] != split for r in part) or {int(r['label_id']) for r in part} != set(range(200)):
            raise ValueError('Invalid official split/class coverage')
        rows.extend(part)
    lookup = {r['sample_id']: r for r in records}
    if len(lookup) != len(records) or len({r['sample_id'] for r in rows}) != len(rows) or set(lookup) != {r['sample_id'] for r in rows}:
        raise ValueError('Inventory sample coverage mismatch')
    if len({r['npz_relpath'] for r in rows}) != len(rows):
        raise ValueError('Duplicate/cross-split source paths')
    for row in rows:
        rec = lookup[row['sample_id']]
        if rec['status'] != 'ok' or (rec['split'], rec['label_id'], rec['npz_relpath']) != (row['split'], int(row['label_id']), row['npz_relpath']):
            raise ValueError('Inventory identity mismatch')
    if args.smoke:
        rows = [r for i, r in enumerate(rows) if i == 0 or lookup[r['sample_id']]['windows'][0]['short_sample']]
    config = dict(dataset='DailyDVS-200', protocol='daily_full_events_1s_stride05_all_windows_v1',
        root=str(args.root.resolve()), inventory_sha256=sha(args.inventory/'samples.jsonl'),
        inventory_manifests=inventory_config['input_manifests'], window_us=1000000, stride_us=500000,
        filter_applied=False, short_sample='one nominal 1s half-open window', tail='no extra partial window',
        views=list(VIEWS), image_dtype='float32', image_shape=[3,224,224], image_normalization='raw rendered values; ImageNet normalization on read',
        augmentation=False, point_sampling='online, intended 2048; no sampled point cache',
        selection_protocol='not set by preprocessing', smoke=args.smoke,
        representation_code=code_hash(), code={name:sha(Path(__file__).parent/name) for name in
          ('preprocess_dailydvs200_full.py','prepare_dailydvs200_windows.py','ucf101_event_store.py')})
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output/'COMPLETE.json').exists():
        raise FileExistsError('Preprocessing already complete; refusing to overwrite completed artifacts')
    if args.resume:
        if json.loads((args.output/'config.json').read_text()) != config:
            raise ValueError('Resume config/code/input identity mismatch')
    elif any(f.name != 'run.lock' for f in args.output.iterdir()):
        raise FileExistsError('Fresh preprocessing requires empty output')
    write_json(args.output/'config.json', config)
    experiment, store = args.output/'experiment', args.output/'event_store'
    experiment.mkdir(exist_ok=True); store.mkdir(exist_ok=True)
    write_json(store/'summary.json', dict(status='preparing', rendering_validation='pending'))
    write_json(experiment/'dataset_audit.json', dict(status='preparing'))
    started, manifests, videos, events = time.monotonic(), [], {}, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = ((str(args.root),str(args.output),r,lookup[r['sample_id']]) for r in rows)
        for i, result in enumerate(pool.map(prepare_one,tasks,chunksize=1),1):
            manifests.extend(result['rows']); events += result['events']
            videos[result['sample_id']] = dict(directory='videos/'+result['sample_id'], metadata_sha256=result['metadata_sha256'])
            if i == 1 or i % 10 == 0 or i == len(rows):
                progress = dict(status='running', samples=i,total_samples=len(rows),windows=len(manifests),
                    images=len(manifests)*3,events=events,seconds=time.monotonic()-started,pid=os.getpid())
                write_json(args.output/'status.json',progress)
                print(json.dumps(progress),flush=True)
    expected_windows = sum(len(lookup[r['sample_id']]['windows']) for r in rows)
    if len(manifests) != expected_windows or len({(r['video_id'],r['window_id']) for r in manifests}) != expected_windows:
        raise RuntimeError('Missing or duplicate windows')
    write_csv(experiment/'window_manifest.csv',manifests,list(manifests[0]))
    write_csv(experiment/'video_splits.csv',rows,list(rows[0]))
    write_json(experiment/'class_to_idx.json',{r['class_name']:int(r['label_id']) for r in rows})
    write_json(store/'videos.json',videos)
    hashes=dict(manifest_sha256=sha(experiment/'window_manifest.csv'), video_splits_sha256=sha(experiment/'video_splits.csv'),
                class_mapping_sha256=sha(experiment/'class_to_idx.json'))
    write_json(experiment/'dataset_audit.json',dict(status='passed',config=config,**hashes,
        scope='NPZ fields/types/polarity/coordinates/timestamps; exact mmap values; every saved image verified',
        raw_aedat4_redecoded=False,samples=len(rows),windows=len(manifests),images=len(manifests)*3))
    write_json(store/'summary.json',dict(status='complete',rendering_validation='passed',
        window_manifest_sha256=hashes['manifest_sha256'],video_splits_sha256=hashes['video_splits_sha256'],
        videos_index_sha256=sha(store/'videos.json'),fields=list(FIELDS),format_version=1,
        videos=len(rows),windows=len(manifests),events=events,layout='unaligned structured NPY, original field dtypes'))
    # Verify public readers against the final commit, including the short sample.
    from ucf101_event_store import EventStore
    from ucf101_window_data import checked_audit
    checked_audit(experiment)
    reader=EventStore(store,experiment)
    examples=[manifests[0],manifests[-1]]+[r for r in manifests if r['short_sample']]
    for row in examples:
        payload=reader.get(row)
        assert len(payload[0]) == row['event_count']
        regenerated=window_views(*payload,row['start_us'],row['end_us'],
                                GCSTRConfig(sensor_width=row['sensor_width'],sensor_height=row['sensor_height']),VIEWS)
        for v in VIEWS:
            assert np.array_equal(regenerated[v],np.load(experiment/'cache'/v/(row['cache_key']+'.npy')))
    final=dict(status='complete',samples=len(rows),windows=len(manifests),images=len(manifests)*3,
               events=events,seconds=time.monotonic()-started,pid=os.getpid(),
               splits={s:dict(samples=sum(r['split']==s for r in rows),windows=sum(r['split']==s for r in manifests)) for s in ('train','test')})
    write_json(args.output/'status.json',final)
    write_json(args.output/'COMPLETE.json',final)
    print(json.dumps(final),flush=True)


if __name__=='__main__':
    main()
