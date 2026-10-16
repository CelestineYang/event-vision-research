"""Prepare complete eligible video streams once; never change or restart training.

Artifacts go to a dedicated cache directory, not the original dataset files.
Per-video commits support --resume after interruption. A complete summary is
published only after all videos/indices and rendering checks have passed.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from ucf101_event_store import FIELDS
from ucf101_window_data import checked_audit, read_csv, sha, write_csv, write_json

DEFAULT_STORE = Path('/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_events_v1')


def video_directory(video):
    return 'videos/'+hashlib.sha256(video.encode()).hexdigest()


def prepare_video(rows,store):
    first=rows[0]
    folder=store/video_directory(first['video_id'])
    metadata_path=folder/'metadata.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        expected={str(r['window_id']):(int(r['start_us']),int(r['end_us']),int(r['event_count'])) for r in rows}
        actual={k:(w['start_us'],w['end_us'],w['event_count']) for k,w in metadata['windows'].items()}
        if metadata['source_sha256']!=first['source_sha256'] or expected!=actual:
            raise RuntimeError('Existing per-video cache identity mismatch')
        if sha(folder/'events.npy')!=metadata['events_sha256']:
            raise RuntimeError('Existing event cache is corrupt')
        return metadata
    if sha(first['source_path'])!=first['source_sha256']:
        raise RuntimeError('NPZ source changed since audit')
    with np.load(first['source_path'],allow_pickle=False) as f:
        arrays={k:f[k] for k in FIELDS}
    lengths={len(a) for a in arrays.values()}
    if len(lengths)!=1 or not len(arrays['t']) or any(a.ndim!=1 for a in arrays.values()):
        raise ValueError('Invalid source arrays')
    if np.any(arrays['t'][1:]<arrays['t'][:-1]) or not np.isin(arrays['p'],[0,1]).all():
        raise ValueError('Invalid timestamp order or polarity')
    if any(not np.isfinite(a).all() or a.dtype.kind not in 'iub' for a in arrays.values()):
        raise ValueError('Unsupported event dtype')
    width,height=int(first['sensor_width']),int(first['sensor_height'])
    if np.any((arrays['x']<0)|(arrays['x']>=width)|(arrays['y']<0)|(arrays['y']>=height)):
        raise ValueError('Invalid source coordinates')
    folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/'events.npy.tmp'
    dtype=np.dtype([(k,arrays[k].dtype) for k in FIELDS],align=False)
    target=np.lib.format.open_memmap(tmp,mode='w+',dtype=dtype,shape=(len(arrays['t']),))
    for k in FIELDS: target[k]=arrays[k]
    target.flush()
    # Exact whole-stream verification, not merely a few window samples.
    for k in FIELDS:
        if target[k].dtype!=arrays[k].dtype or not np.array_equal(target[k],arrays[k]):
            raise RuntimeError(f'Event field changed: {k}')
    del target
    tmp.replace(folder/'events.npy')
    metadata=dict(video_id=first['video_id'],split=first['split'],source_path=first['source_path'],
                  source_sha256=first['source_sha256'],events=len(arrays['t']),
                  field_dtypes={k:str(arrays[k].dtype) for k in FIELDS},
                  events_sha256=sha(folder/'events.npy'),windows={})
    for row in rows:
        if (row['source_sha256'],row['source_path'],row['split'])!=(first['source_sha256'],first['source_path'],first['split']):
            raise ValueError('Source video spans splits or sources')
        start,end=int(row['start_us']),int(row['end_us'])
        left,right=np.searchsorted(arrays['t'],[start,end],side='left')
        if end-start!=1_000_000 or right-left!=int(row['event_count']) or right-left<=10000:
            raise ValueError('Window bounds/count changed')
        metadata['windows'][str(row['window_id'])]=dict(start_us=start,end_us=end,event_count=int(right-left),left=int(left),right=int(right))
    write_json(metadata_path,metadata)
    return metadata


def verify_rendering(experiment,store,*,samples=32):
    import torch
    from train_ucf101_window_joint import ThreeViewDataset
    from ucf101_event_store import EventStore,PreparedThreeViewDataset
    metrics={}
    for split in ('train','test'):
        old=ThreeViewDataset(experiment,split,seed=0)
        new=PreparedThreeViewDataset(experiment,split,store,seed=0,_allow_pending=True)
        old.set_epoch(7);new.set_epoch(7)
        indices=np.random.default_rng(42).choice(len(old),min(samples,len(old)),replace=False)
        for i in indices:
            row=old.rows[int(i)]
            events=new.events.get(row)
            with np.load(row['source_path'],allow_pickle=False) as f:
                left,right=np.searchsorted(f['t'],[int(row['start_us']),int(row['end_us'])],side='left')
                for k,a in zip(FIELDS,events):
                    np.testing.assert_array_equal(a,f[k][left:right])
            a,label_a,id_a=old[int(i)];b,label_b,id_b=new[int(i)]
            if (label_a,id_a)!=(label_b,id_b): raise RuntimeError('Sample identity mismatch')
            for view in a: torch.testing.assert_close(a[view],b[view],rtol=0,atol=0)
        # Informative current-load CPU comparison; not a promise of epoch speedup.
        timings={}
        for name,data in [('npz',old),('memory_map',new)]:
            begin=time.perf_counter()
            for i in indices: data[int(i)]
            timings[name]=time.perf_counter()-begin
        metrics[split]=dict(windows=len(indices),all_views_bitwise_equal=True,cpu_seconds=timings,
                            note='Test baseline reads image cache; prepared reader renders events online')
        print('Rendering verified',split,len(indices),timings,flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    p.add_argument('--store',type=Path,default=DEFAULT_STORE)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-samples',type=int,default=32)
    args=p.parse_args()
    if args.verify_samples<1: p.error('verify-samples must be positive')
    audit=checked_audit(args.experiment)
    if audit['config'].get('audit_scope')!='full': raise ValueError('Full audit required')
    rows=read_csv(args.experiment/'window_manifest.csv')
    grouped=defaultdict(list)
    for row in rows: grouped[row['video_id']].append(row)
    signature=dict(window_manifest_sha256=sha(args.experiment/'window_manifest.csv'),
                   video_splits_sha256=sha(args.experiment/'video_splits.csv'),
                   fields=list(FIELDS),layout='unaligned structured NPY, original field dtypes',
                   format_version=1,experiment=str(args.experiment.resolve()),
                   code={n:sha(Path(__file__).parent/n) for n in ['preprocess_ucf101_event_store.py','ucf101_event_store.py']})
    args.store.mkdir(parents=True,exist_ok=True)
    # Conservative uncompressed 4*int64 upper bound, one copy per eligible video.
    video_rows={r['video_id']:r for r in read_csv(args.experiment/'video_splits.csv')}
    upper_bytes=sum(int(video_rows[v]['event_count'])*32 for v in grouped)
    if shutil.disk_usage(args.store).free < upper_bytes+2*1024**3:
        raise RuntimeError('Insufficient disk space for conservative full-event cache estimate')
    with (args.store/'.preprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signature_path=args.store/'signature.json'
        if signature_path.exists():
            if not args.resume or json.loads(signature_path.read_text())!=signature:
                raise ValueError('Existing store requires --resume with an identical signature')
        else:
            if any(x.name!='.preprocess.lock' for x in args.store.iterdir()): raise ValueError('Use a fresh dedicated cache directory')
            write_json(signature_path,signature)
        index={};total_events=0;started=time.perf_counter()
        status_path=args.experiment/'event_preprocess_status.json'
        state=dict(pid=os.getpid(),store=str(args.store),status='running',phase='videos',
                   total_videos=len(grouped),total_windows=len(rows),completed_videos=0)
        def status(**updates):
            state.update(updates,updated_at=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-started)
            write_json(status_path,state)
        status()
        try:
            for i,(video,items) in enumerate(grouped.items(),1):
                metadata=prepare_video(items,args.store)
                folder=args.store/video_directory(video)
                index[video]=dict(directory=video_directory(video),metadata_sha256=sha(folder/'metadata.json'))
                total_events+=metadata['events']
                if i%50==0 or i==len(grouped):
                    status(completed_videos=i,events=total_events)
                    print(json.dumps(state),flush=True)
            write_json(args.store/'videos.json',index)
            # The complete data index is ready; final rendering validation is still pending.
            summary=dict(signature,status='validating',rendering_validation='pending',videos=len(index),
                         windows=len(rows),events=total_events,videos_index_sha256=sha(args.store/'videos.json'))
            # EventStore allows reading for validation; no training entry auto-selects this store.
            write_json(args.store/'summary.json',summary)
            status(phase='rendering_validation')
            rendering=verify_rendering(args.experiment,args.store,samples=args.verify_samples)
            summary.update(status='complete',rendering_validation='passed',rendering=rendering,seconds=time.perf_counter()-started)
            write_json(args.store/'summary.json',summary)
            write_json(args.experiment/'event_preprocess_report.json',summary)
            status(status='complete',phase='complete')
        except Exception as exc:
            status(status='failed',error=f'{type(exc).__name__}: {exc}')
            if (args.store/'summary.json').exists():
                summary=json.loads((args.store/'summary.json').read_text())
                summary.update(status='failed',error=str(exc));write_json(args.store/'summary.json',summary)
            raise


if __name__=='__main__': main()
