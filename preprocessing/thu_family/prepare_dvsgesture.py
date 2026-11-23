"""Decode DVS128 Gesture AEDAT 3.1 into aligned full-event/point windows.

Reference: EventMamba dataprocess/generate_dvsgesture.py and uti.py.
No radius filtering and no offline augmented duplicate point rows.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import h5py
import numpy as np
from prepare_chl import sha, write_json


def load_aedat(path, require_monotonic=True):
    """Read valid polarity packets, retaining absolute microsecond timestamps."""
    parts = []
    source_id = None
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'#!AER-DAT3.1':
            raise ValueError('Expected AEDAT 3.1: ' + str(path))
        while True:
            line = f.readline()
            if not line or not line.startswith(b'#'):
                raise ValueError('Missing AEDAT end-header marker')
            if line.strip() == b'#!END-HEADER': break
        while True:
            header = f.read(28)
            if not header: break
            if len(header) != 28: raise ValueError('Truncated AEDAT packet header')
            kind, source, size, offset, overflow, capacity, number, valid_count = struct.unpack('<HHIIIIII', header)
            if size < 4 or number > capacity or valid_count > number or capacity * size > 512*1024*1024:
                raise ValueError('Invalid AEDAT packet dimensions')
            payload = f.read(capacity * size)
            if len(payload) != capacity * size: raise ValueError('Truncated AEDAT payload')
            if kind != 1: continue
            if size < 8 or offset < 4 or offset+4 > size: raise ValueError('Invalid polarity layout')
            if source_id is None: source_id = source
            if source != source_id: raise ValueError('Multiple polarity sources are unsupported')
            data = np.ndarray((number,), dtype='<u4', buffer=payload, strides=(size,))
            times = np.ndarray((number,), dtype='<u4', buffer=payload, offset=offset, strides=(size,))
            valid = (data & 1) != 0
            if int(valid.sum()) != valid_count: raise ValueError('AEDAT validity count mismatch')
            data, times = data[valid], times[valid]
            if (times >= 2**31).any(): raise ValueError('Invalid 31-bit timestamp')
            x, y = (data >> 17) & 0x7fff, (data >> 2) & 0x7fff
            if (x >= 128).any() or (y >= 128).any(): raise ValueError('DVS128 coordinate out of bounds')
            t = (np.int64(overflow) << 31) | times.astype(np.int64)
            if len(t): parts.append(np.column_stack((x, y, t, (data >> 1) & 1)))
    if not parts: raise ValueError('No valid polarity events')
    events = np.concatenate(parts).astype(np.int64, copy=False)
    if require_monotonic and (np.diff(events[:,2]) < 0).any():
        raise ValueError('Nonmonotonic timestamps; no implicit repair')
    return events


def segment_bounds(events, start, end, recording_monotonic=True):
    """Match annotation boolean masking without sorting the raw recording."""
    if recording_monotonic:
        a,b = np.searchsorted(events[:,2],[start,end],side='left')
    else:
        indices = np.flatnonzero((events[:,2]>=start)&(events[:,2]<end))
        if not len(indices): return 0,0
        if (np.diff(indices)!=1).any():
            raise ValueError('Annotation events are noncontiguous in raw recording')
        a,b = int(indices[0]),int(indices[-1])+1
    if (np.diff(events[a:b,2])<0).any():
        raise ValueError('Nonmonotonic timestamps inside annotated segment; no implicit repair')
    return int(a),int(b)


def windows(t):
    """Preserve EventMamba's strict end<last and excluded last <=end event."""
    if not len(t): return
    for start in range(int(t[0]), int(t[-1])-500000, 250000):
        end = start+500000
        yield start, end, int(np.searchsorted(t,start,'left')), int(np.searchsorted(t,end,'right'))-1


def sample_points(events, count, rng):
    if len(events) <= 100: return None
    indices = np.arange(len(events))
    if count > len(events): indices = rng.choice(indices,size=count,replace=True)
    else:
        rng.shuffle(indices)
        indices = indices[:count]
    indices.sort()
    # Match upstream float32 conversion before time normalization, including epsilon.
    points = events[indices][:,[2,0,1]].astype(np.float32)
    points[:,0] = (points[:,0]-points[:,0].min())/(points[:,0].max()-points[:,0].min()+1e-6)
    points[:,1:] /= 128
    return points


def read_records(root, classes):
    records, seen, subjects, audit = [], set(), {}, {}
    for split in ('train','test'):
        names=(root/'splits'/f'trials_to_{split}.txt').read_text().split()
        subjects[split]=set()
        excluded=0; overlaps=[]
        for name in names:
            if Path(name).name != name or not name.endswith('.aedat') or name in seen:
                raise ValueError('Duplicate/unsafe trial: '+name)
            seen.add(name); subject=name.split('_')[0];subjects[split].add(subject)
            raw=root/'raw/events'/name
            annotation=root/'annotations'/(Path(name).stem+'_labels.csv')
            if not raw.is_file() or not annotation.is_file(): raise FileNotFoundError(name)
            with annotation.open(newline='') as annotation_file:
                rows=list(csv.DictReader(annotation_file))
            intervals=[]; occurrences={}
            for row_id,row in enumerate(rows):
                label,start,end=int(row['class']),int(row['startTime_usec']),int(row['endTime_usec'])
                if not 1<=label<=11 or not 0<=start<end: raise ValueError('Invalid annotation: '+name)
                occurrence=occurrences.get(label,0);occurrences[label]=occurrence+1
                for old_start,old_end,old_id in intervals:
                    if start<old_end and old_start<end:overlaps.append([name,old_id,row_id])
                intervals.append((start,end,row_id))
                if label>classes:excluded+=1;continue
                records.append(dict(split=split, sample_id=f'{Path(name).stem}_c{label:02d}_s{occurrence:02d}',
                    filename=name,subject=subject,label=label-1,official_label=label,
                    annotation_row=row_id,occurrence=occurrence,segment_start_us=start,segment_end_us=end,
                    event_file=f'{split}/{Path(name).stem}.npy'))
        audit[split]=dict(trials=len(names),subjects=sorted(subjects[split]),
            segments=sum(r['split']==split for r in records),excluded_other_segments=excluded,
            annotation_overlaps=overlaps)
    if subjects['train'] & subjects['test']:raise ValueError('Cross-split subject leakage')
    return records,audit


def prepare(args):
    root,out=args.root.resolve(),args.output.resolve()
    records,audit=read_records(root,args.classes)
    if out==root or root in out.parents:raise ValueError('Output must be outside source directory')
    out.mkdir(parents=True,exist_ok=False)
    cfg=dict(dataset='DVS128Gesture',classes=args.classes,width=128,height=128,timestamp_unit='us',
        window_us=500000,step_us=250000,points=args.points,seed=args.seed,
        window_boundary='EventMamba strict end<last; last <=end index excluded',
        segment_boundary='annotation start <= t < end',minimum_raw_events_exclusive=100,
        radius_filter=False,offline_point_augmentation=False,
        timestamp_policy='preserve raw recording order; audit global disorder; require monotonic contiguous annotation segments',
        frame_events='all raw events in same valid point-window slice',
        class_policy='exclude official class 11' if args.classes==10 else 'retain all 11 classes',
        adaptation='deterministic per-window RNG; no offline crop duplicates; no checkpoint compatibility guarantee',
        code_sha256=sha(Path(__file__)),numpy=np.__version__,smoke_per_split=args.smoke_per_split,
        source_protocol='https://github.com/rhwxmx/EventMamba/blob/main/dataprocess/generate_dvsgesture.py',
        **{s+'_list_sha256':sha(root/'splits'/f'trials_to_{s}.txt') for s in ('train','test')})
    write_json(out/'config.json',cfg);write_json(out/'split_audit.json',audit)
    chosen={s:[r for r in records if r['split']==s] for s in ('train','test')}
    if args.smoke_per_split:chosen={s:rs[:args.smoke_per_split] for s,rs in chosen.items()}
    write_json(out/'samples.json',[r for rs in chosen.values() for r in rs])
    write_json(out/'status.json',dict(state='running'))
    try:
        for split,rs in chosen.items():
            (out/split).mkdir()
            with h5py.File(out/(split+'.h5'),'x') as h, (out/(split+'_windows.jsonl')).open('w') as manifest:
                h.attrs['complete']=False;h.attrs['config_sha256']=sha(out/'config.json')
                data=h.create_dataset('data',(0,args.points,3),maxshape=(None,args.points,3),dtype='f4',chunks=(1,args.points,3))
                labels=h.create_dataset('label',(0,),maxshape=(None,),dtype='i2')
                marks=h.create_dataset('mark',(0,),maxshape=(None,),dtype='i4')
                previous=None;events=None;n=0
                for mark,record in enumerate(rs):
                    if record['filename']!=previous:
                        previous=record['filename'];src=root/'raw/events'/previous
                        events=load_aedat(src,require_monotonic=False);dest=out/record['event_file']
                        backward_count=int((np.diff(events[:,2])<0).sum())
                        recording_monotonic=backward_count==0
                        np.save(dest,events,allow_pickle=False)
                        if not np.array_equal(np.load(dest,mmap_mode='r'),events):raise RuntimeError('Cache round-trip failure')
                        annotation=root/'annotations'/(Path(previous).stem+'_labels.csv')
                        write_json(dest.with_suffix('.json'),dict(events=len(events),source_sha256=sha(src),
                            cache_sha256=sha(dest),annotation_sha256=sha(annotation),source=str(src),
                            timestamp_backward_pairs=backward_count))
                    a,b=segment_bounds(events,record['segment_start_us'],record['segment_end_us'],recording_monotonic)
                    valid_count=0
                    for window_id,(start,end,left,right) in enumerate(windows(events[a:b,2])):
                        left+=int(a);right+=int(a)
                        seed=int.from_bytes(hashlib.sha256(f'{args.seed}:{record["sample_id"]}:{window_id}'.encode()).digest()[:4],'little')
                        points=sample_points(events[left:right],args.points,np.random.RandomState(seed))
                        row=dict(**record,mark=mark,window_id=window_id,start_us=start,end_us=end,left=left,right=right,
                            raw_events=max(0,right-left),valid=points is not None,point_row=n if points is not None else None)
                        manifest.write(json.dumps(row)+'\n')
                        if points is not None:
                            for ds in (data,labels,marks):ds.resize(n+1,axis=0)
                            data[n]=points;labels[n]=record['label'];marks[n]=mark;n+=1;valid_count+=1
                    if not valid_count:raise ValueError('Segment has no eligible windows: '+record['sample_id'])
                    write_json(out/'status.json',dict(state='running',split=split,segments=mark+1,windows=n))
                    print(json.dumps(dict(split=split,segment=mark+1,total=len(rs),valid_windows=n)),flush=True)
                if not n:raise ValueError('Empty split: '+split)
                h.attrs['complete']=True
        write_json(out/'status.json',dict(state='complete',scope='smoke' if args.smoke_per_split else 'full'))
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=repr(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--classes',type=int,choices=(10,11),required=True)
    p.add_argument('--points',type=int,choices=(1024,2048),required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke-per-split',type=int,default=0)
    args=p.parse_args()
    if args.smoke_per_split<0:p.error('Negative smoke budget')
    prepare(args)


if __name__=='__main__':main()
