"""Adopt user-supplied video split lists without modifying previous experiments.

Reuse the complete prior raw audit/windows for known videos. Read original MAT
for previously omitted records; materialize only missing nonempty NPZ locally.
No training, image-cache generation, or process signals occur here.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

from ucf101_window_data import (ROOT, PROTOCOL, checked_audit, code_hash,
                                distribution, full_window_bounds, read_csv,
                                sha, write_csv, write_json)

NEW_OUTPUT = Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2')


def parse_lists(train_path,test_path,mapping):
    rows=[]
    for split,path in [('train',train_path),('test',test_path)]:
        for number,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
            if not line.strip(): continue
            value=ast.literal_eval(line)
            if not isinstance(value,(list,tuple)) or len(value)!=2:
                raise ValueError(f'Invalid split entry: {path}:{number}')
            original,label=value
            parts=[p for p in original.replace('\\','/').split('/') if p]
            category,filename=parts[-2:]
            if mapping.get(category)!=label:
                raise ValueError(f'Class label mismatch: {path}:{number}')
            match=re.fullmatch(r'v_.+_g(\d+)_c(\d+)\.mat',filename)
            if not match: raise ValueError(f'Invalid MAT filename: {filename}')
            rows.append(dict(video_id=f'{int(label):03d}_{Path(filename).stem}',split=split,
                             label_id=int(label),class_name=category,group=int(match[1]),clip=int(match[2]),
                             event_relpath=category+'/'+filename))
    for key in ('video_id','event_relpath'):
        if len({r[key] for r in rows})!=len(rows):
            raise ValueError(f'Duplicate/overlapping video in supplied lists: {key}')
    for split in ('train','test'):
        if {r['label_id'] for r in rows if r['split']==split}!=set(range(101)):
            raise ValueError(f'Expected 101 classes in {split}')
    return rows


def prepare(args):
    old=checked_audit(args.previous)
    mapping=json.loads((args.previous/'class_to_idx.json').read_text())
    rows=parse_lists(args.train_list,args.test_list,mapping)
    metadata={r['event_relpath']:r for r in read_csv(args.root/'splits/manifest.csv')}
    old_records={r['video_id']:r for r in old['records']}
    old_windows={}
    for w in read_csv(args.previous/'window_manifest.csv'):
        old_windows.setdefault(w['video_id'],[]).append(w)
    sizes={(int(r['sensor_width']),int(r['sensor_height'])) for r in metadata.values()}
    if len(sizes)!=1: raise ValueError('Sensor metadata is inconsistent')
    width,height=next(iter(sizes))
    if args.output.exists(): raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    inputs=args.output/'split_inputs';inputs.mkdir()
    shutil.copyfile(args.train_list,inputs/'train_ucf.txt')
    shutil.copyfile(args.test_list,inputs/'test_ucf.txt')
    config=dict(protocol=PROTOCOL,seed=0,validation=False,audit_scope='full',
                split_unit='user_supplied_video_lists',anomaly_policy='retain_suspicious',
                evaluation_split='test',selection='test_video_nll',independent_test=False,
                root=str(args.root.resolve()),representation_code=code_hash(),
                source_manifests={s:sha(p) for s,p in [('train',inputs/'train_ucf.txt'),('test',inputs/'test_ucf.txt')]},
                prepare_code_sha256=sha(__file__),prior_audit_sha256=sha(args.previous/'dataset_audit.json'),
                audit_method='Reuse prior complete raw/NPZ audit for known sources; audit omitted MAT directly',
                group_overlap_allowed=True,fresh_model_initialization_required=True)
    windows=[];records=[];converted=0;empty=0;errors=[]
    transitions=Counter()
    with ExitStack() as stack:
        archive_index=None
        for i,row in enumerate(rows):
            key=row['event_relpath']
            record=dict(video_id=row['video_id'],split=row['split'],label_id=row['label_id'],flags=[])
            try:
                if key in metadata:
                    source=metadata[key];source_id=source['sample_id']
                    source_path=(args.root/source['npz_relpath']).resolve()
                    if not source_path.is_file(): raise FileNotFoundError(source_path)
                    record.update(old_records[source_id])
                    record.update(video_id=row['video_id'],split=row['split'],reused_audit_video_id=source_id)
                    proposed=[dict(w) for w in old_windows.get(source_id,[])]
                    row.update(source_path=str(source_path),event_count=source['event_count'],
                               duration_usec=source['duration_usec'],status='existing_npz',old_split=source['split'])
                    transitions[source['split']+'->'+row['split']]+=1
                else:
                    if archive_index is None:
                        archive_index={}
                        for path in sorted(args.root.parent.glob('Yin Bi - UCF101-DVS-Part*.zip')):
                            z=stack.enter_context(ZipFile(path))
                            for member in z.namelist():
                                if member.lower().endswith('.mat'):
                                    rel='/'.join(member.replace('\\','/').split('/')[-2:])
                                    if rel in archive_index: raise ValueError('Duplicate MAT in archives')
                                    archive_index[rel]=(z,member,path.name)
                    z,member,archive_name=archive_index[key]
                    raw=loadmat(io.BytesIO(z.read(member)))
                    if 'pol' not in raw: raise ValueError('BLOCKED: polarity unavailable')
                    arrays={k:np.asarray(raw[v]).reshape(-1) for k,v in [('t','ts'),('x','x'),('y','y'),('p','pol')]}
                    if len({len(a) for a in arrays.values()})!=1: raise ValueError('Mismatched event lengths')
                    if not all(np.isfinite(a).all() and np.equal(a,np.floor(a)).all() for a in arrays.values()):
                        raise ValueError('Invalid raw event fields')
                    if not np.isin(arrays['p'],[0,1]).all(): raise ValueError('Invalid raw polarity')
                    t=arrays['t'].astype(np.int64);x=arrays['x'];y=arrays['y'];p=arrays['p']
                    if np.any((x<0)|(x>=width)|(y<0)|(y>=height)): raise ValueError('Out-of-range coordinates')
                    duration=int(t.max())-int(t.min()) if len(t) else 0
                    row.update(source_path='',event_count=len(t),duration_usec=duration,
                               status='empty_raw' if not len(t) else 'converted_npz',old_split='unlisted')
                    record.update(source_archive=archive_name,raw_timestamp_dtype=str(arrays['t'].dtype),
                                  events=len(t),duration_us=duration,polarity_values=np.unique(p).tolist(),
                                  sensor_width=width,sensor_height=height)
                    proposed=[]
                    if not len(t):
                        empty+=1;record['filter_reason']='empty_raw';record['flags'].append('empty')
                    else:
                        if np.any(np.diff(t)<0): record['flags'].append('raw_nonmonotonic')
                        if x.min()==x.max() or y.min()==y.max(): record['flags'].append('constant_coordinate')
                        if len(np.unique(p))<2: record['flags'].append('single_polarity_review_required')
                        order=np.argsort(t,kind='stable');t=t[order]
                        source_path=(args.output/'processed'/f"{row['label_id']:03d}"/(Path(key).stem+'.npz')).resolve()
                        source_path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(source_path,t=t,x=x[order].astype(np.int16),
                                            y=y[order].astype(np.int16),p=p[order].astype(np.uint8))
                        row['source_path']=str(source_path);converted+=1
                        record['source_sha256']=sha(source_path)
                        count=max(0,(int(t[-1])-int(t[0])-1000000)//500000+1)
                        record['candidate_windows']=count
                        record['low_event_windows']=0
                        if count>=40: record['filter_reason']='official_num_windows_ge40'
                        for window_id,(start,end) in enumerate(full_window_bounds(t[0],t[-1])):
                            left,right=np.searchsorted(t,[start,end],side='left');n=int(right-left)
                            if n<=10000:
                                record['low_event_windows']+=1;continue
                            proposed.append(dict(window_id=window_id,start_us=start,end_us=end,event_count=n,
                                                 sensor_width=width,sensor_height=height,source_sha256=record['source_sha256']))
                for w in proposed:
                    w.update(video_id=row['video_id'],split=row['split'],label_id=row['label_id'],source_path=row['source_path'])
                    identity={k:w[k] for k in ('video_id','window_id','start_us','end_us','source_sha256')}
                    identity.update(split_inputs=config['source_manifests'],protocol=PROTOCOL,code=code_hash())
                    w['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                    windows.append(w)
                record['valid_windows']=len(proposed)
                record['source_path']=row['source_path']
                if not proposed: record.setdefault('filter_reason','no_full_window_above10000')
            except Exception as exc:
                errors.append(dict(video_id=row['video_id'],error=str(exc)))
                record['error']=str(exc)
                row.update(source_path=row.get('source_path',''),event_count=row.get('event_count',''),
                           duration_usec=row.get('duration_usec',''),status='error',old_split=row.get('old_split','unknown'))
            records.append(record)
            if (i+1)%1000==0: print('Prepared',i+1,'/',len(rows),flush=True)
    coverage={}
    for split in ('train','test'):
        subset=[w for w in windows if w['split']==split]
        coverage[split]=dict(original_videos=sum(r['split']==split for r in rows),
                             retained_videos=len({w['video_id'] for w in subset}),windows=len(subset),
                             original_class_counts=dict(Counter(r['label_id'] for r in rows if r['split']==split)),
                             retained_class_video_counts=dict(Counter({w['video_id']:w['label_id'] for w in subset}.values())),
                             class_window_counts=dict(Counter(w['label_id'] for w in subset)),
                             event_count=distribution([int(w['event_count']) for w in subset]),
                             windows_per_video=distribution(list(Counter(w['video_id'] for w in subset).values())))
        if {int(w['label_id']) for w in subset}!=set(range(101)): errors.append(dict(split=split,error='Missing class after filtering'))
    groups={s:{(r['label_id'],r['group']) for r in rows if r['split']==s} for s in ('train','test')}
    report=dict(status='blocked' if errors else 'passed',config=config,fatal=errors,coverage=coverage,
                records=records,excluded_videos=[r for r in records if not r.get('valid_windows')],
                suspicious_video_ids=[r['video_id'] for r in records if r['flags']],
                converted_npz=converted,empty_videos=empty,old_to_new=dict(transitions),
                shared_class_groups=len(groups['train']&groups['test']))
    write_csv(args.output/'video_splits.csv',rows,list(rows[0]))
    fields=['video_id','window_id','start_us','end_us','event_count','label_id','split','source_path',
            'sensor_width','sensor_height','source_sha256','cache_key']
    write_csv(args.output/'window_manifest.csv',windows,fields)
    write_json(args.output/'class_to_idx.json',mapping)
    report.update(manifest_sha256=sha(args.output/'window_manifest.csv'),
                  video_splits_sha256=sha(args.output/'video_splits.csv'),
                  class_mapping_sha256=sha(args.output/'class_to_idx.json'))
    write_json(args.output/'dataset_audit.json',report)
    print(json.dumps({k:report[k] for k in ['status','converted_npz','empty_videos','shared_class_groups','fatal']},indent=2),flush=True)
    print({s:{k:v[k] for k in ['original_videos','retained_videos','windows']} for s,v in coverage.items()},flush=True)
    return 0 if not errors else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--train-list',type=Path,default=Path('train_ucf.txt'))
    p.add_argument('--test-list',type=Path,default=Path('test_ucf(1).txt'))
    p.add_argument('--previous',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_v1'))
    p.add_argument('--output',type=Path,default=NEW_OUTPUT)
    return prepare(p.parse_args())


if __name__=='__main__': raise SystemExit(main())
