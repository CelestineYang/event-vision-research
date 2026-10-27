"""THU exact video schedules; shared event augmentation precedes both branches."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from gcstr_representations import GCSTRConfig
from gcstr_video_sequence_dataset import (VideoSequenceDataset,video_groups,make_schedule,keyed_rng,
    sample_spatial_params,apply_spatial_params)
from thu_chl_render import render


def point_input(events,rng,count=2048):
    # Match historical THU radius filter; keep sparse Test windows with explicit fallback.
    t,x,y,p=events
    if not len(t):raise ValueError('Empty window requires explicit handling')
    xyz=np.column_stack([t,x,y]).astype(np.int64);normalized=xyz.astype(np.float64)
    normalized[:,0]-=normalized[:,0].min();normalized[:,0]/=max(normalized[:,0].max(),1)
    normalized[:,1]/=346;normalized[:,2]/=260
    # Open3D radius search is strict distance < radius, and count includes self.
    counts=cKDTree(normalized).query_ball_point(normalized,np.nextafter(.01,0),return_length=True,workers=1)
    keep=np.flatnonzero(counts>5)
    fallback=len(keep)<100
    pool=xyz if fallback else xyz[keep]
    idx=np.sort(rng.choice(len(pool),count,replace=len(pool)<count))
    points=pool[idx].astype(np.float64);points[:,0]-=points[:,0].min()
    points[:,0]/=max(points[:,0].max(),1);points[:,1]/=346;points[:,2]/=260
    return points.T.astype(np.float32),int(fallback)


class THUVideoSequenceDataset(VideoSequenceDataset):
    def __init__(self,config,split,epoch=0,smoke=False):
        self.config,self.split,self.epoch=config,split,epoch;self.store=Path(config['store'])
        if json.loads((self.store/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete source')
        self.rows=[]
        for line in (self.store/f'{split}_windows.jsonl').read_text().splitlines():
            r=json.loads(line)
            if split=='train' and not r['valid']:continue
            if r['right']<=r['left']:raise ValueError('Empty test window must not silently disappear')
            r.update(video_id=r['sample_id'],label_id=r['label']);self.rows.append(r)
        self.groups=video_groups(self.rows)
        samples=json.loads((self.store/'samples.json').read_text())
        expected={r['sample_id'] for r in samples if r['split']==split}
        if set(self.groups)!=expected:raise ValueError('Recording coverage mismatch')
        if smoke:
            if split!='train':raise ValueError('Smoke only reads Train')
            candidates=[v for v in sorted(self.groups) if len(self.groups[v])>=2*config['tbptt_steps']]
            if len(candidates)<3:raise ValueError('Need three long Train recordings for smoke')
            self.groups={v:self.groups[v] for v in candidates[:3]}
        self.windows=sum(map(len,self.groups.values()))
        # Three-video smoke uses two lanes to exercise a singleton tail as well.
        batch_size=min(2,config['video_batch_size_per_gpu']) if smoke else config['video_batch_size_per_gpu']
        self.schedule=make_schedule(self.groups,batch_size,config['tbptt_steps'],config['seed'],epoch,split=='train')
        self.position={i:(j,indices) for indices in self.groups.values() for j,i in enumerate(indices)}
        self.opened=OrderedDict()

    def __getstate__(self):
        state=self.__dict__.copy();state['opened']=OrderedDict();return state

    def window(self,index):
        row=self.rows[index];name=row['event_file']
        if name not in self.opened:
            self.opened[name]=np.load(self.store/name,mmap_mode='r',allow_pickle=False)
            if len(self.opened)>8:self.opened.popitem(last=False)
        self.opened.move_to_end(name)
        a=self.opened[name][row['left']:row['right']]
        events=(a[:,2],a[:,0],a[:,1],a[:,3]);cfg=GCSTRConfig(sensor_width=346,sensor_height=260);fallback=0
        epoch=self.epoch if self.split=='train' else 0
        if self.split=='train':
            events,fallback=apply_spatial_params(events,sample_spatial_params(self.config['seed'],epoch,row['video_id']),
                keyed_rng('jitter',self.config['seed'],epoch,row['video_id'],row['window_id']),cfg)
        points,point_fallback=point_input(events,keyed_rng('points',self.config['seed'],epoch,row['video_id'],row['window_id']),self.config['point_count'])
        t,x,y,p=events
        images=render(np.column_stack([x,y,t,p]),row['start_us'],row['end_us'],cfg)
        j,sequence=self.position[index]
        gap=(row['start_us']-self.rows[sequence[j-1]]['start_us'])/1e6 if j else 0.
        result=dict(events=torch.from_numpy(points),label=row['label'],video_id=row['video_id'],window_id=str(row['window_id']),
            start_us=row['start_us'],end_us=row['end_us'],gap_seconds=torch.tensor(gap,dtype=torch.float32),valid=True,
            video_start=j==0,video_end=j==len(sequence)-1,augmentation_fallbacks=fallback+point_fallback)
        for view,image in images.items():
            if image.shape!=(3,224,224) or not np.isfinite(image).all():raise ValueError('Invalid frame')
            result[view]=torch.from_numpy(((image-np.array([.485,.456,.406],dtype='f4')[:,None,None])/
                np.array([.229,.224,.225],dtype='f4')[:,None,None]).astype('f4'))
        return result
