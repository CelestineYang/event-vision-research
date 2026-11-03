"""prepared_v1 NPZ reader with bounded per-worker recording LRU."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch.utils.data import Dataset
from gcstr_augmentation import augment_events
from gcstr_representations import GCSTRConfig
from ucf101_window_data import VIEWS,window_views,sha

def identity(root):
 root=Path(root)
 status=json.loads((root/'status.json').read_text());cfg=json.loads((root/'config.json').read_text())
 if status.get('state')!='complete' or status.get('scope')!='full' or cfg.get('classes')!=200 or (cfg.get('width'),cfg.get('height'))!=(320,240):raise ValueError('Expected complete DailyDVS-200 prepared_v1')
 return {n:sha(root/n) for n in ['config.json','train_sources.jsonl','test_sources.jsonl','train_windows.jsonl','test_windows.jsonl']}

class PreparedDailyViews(Dataset):
 def __init__(self,root,split,*,seed=0,cache_mib=256,frames=None):
  self.root=Path(root);self.split=split;self.seed=seed;self.epoch=0;self.augmentation='spatial' if split=='train' else 'none'
  self.identity=identity(root);self.budget=int(cache_mib*2**20)
  if self.budget<0:raise ValueError('cache_mib must be nonnegative')
  self.opened=OrderedDict();self.bytes=0;self.loads=0
  self.sources={r['sample_id']:r for r in map(json.loads,(self.root/f'{split}_sources.jsonl').read_text().splitlines())}
  self.rows=[r|{'video_id':r['sample_id'],'label_id':r['label']} for r in map(json.loads,(self.root/f'{split}_windows.jsonl').read_text().splitlines()) if r['valid']]
  if any(not 0<=r['label']<200 for r in self.rows):raise ValueError('Invalid labels')
  self.frames=Path(frames) if frames else None
  if self.frames:
   meta=json.loads((self.frames/'COMPLETE.json').read_text())
   if meta['status']!='complete' or meta['scope']!='full' or meta['identity']!=self.identity:raise ValueError('Incomplete/stale frame cache')
 def __len__(self):return len(self.rows)
 def set_epoch(self,n):self.epoch=n
 def __getstate__(self):
  s=self.__dict__.copy();s.update(opened=OrderedDict(),bytes=0,loads=0);return s
 def events(self,row):
  name=row['event_file'];path=Path(name);path=path if path.is_absolute() else self.root/path
  if name in self.opened:a=self.opened.pop(name);self.opened[name]=a
  else:
   if sha(path)!=self.sources[row['sample_id']]['sha256']:raise ValueError('Source checksum mismatch')
   with np.load(path,allow_pickle=False) as z:
    t=z['t'].astype(np.int64);origin=int(t[0]);t-=origin
    a=(t,z['x'].astype(np.int64),z['y'].astype(np.int64),z['p'].astype(np.int64))
   if origin!=self.sources[row['sample_id']]['timestamp_origin_us'] or any(len(v)!=len(t) for v in a) or np.any(np.diff(t)<0):raise ValueError('Source metadata mismatch')
   size=sum(v.nbytes for v in a);self.loads+=1
   while self.opened and self.bytes+size>self.budget:
    _,old=self.opened.popitem(last=False);self.bytes-=sum(v.nbytes for v in old)
   if size<=self.budget:self.opened[name]=a;self.bytes+=size
  e=tuple(v[row['left']:row['right']] for v in a)
  if len(e[0])!=row['raw_events']:raise ValueError('Window count mismatch')
  return e
 def images(self,row):
  if self.augmentation=='none' and self.frames:
   images={v:np.load(self.frames/self.split/v/f"{row['point_row']}.npy",allow_pickle=False) for v in VIEWS}
  else:
   events=self.events(row);cfg=GCSTRConfig(sensor_width=320,sensor_height=240)
   if self.augmentation=='spatial':
    seed=int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["sample_id"]}:{row["window_id"]}'.encode()).digest()[:8],'little')
    events=augment_events(*events,mode='spatial',rng=np.random.default_rng(seed),config=cfg)
   images=window_views(*events,row['start_us'],row['end_us'],cfg)
  if any(a.shape!=(3,224,224) or a.dtype!=np.float32 or not np.isfinite(a).all() or a.min()<0 or a.max()>1 for a in images.values()):raise ValueError('Invalid projection cache')
  return images
 def __getitem__(self,i):
  r=self.rows[i];images=self.images(r);mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None]
  return {v:torch.from_numpy((a-mean)/std) for v,a in images.items()},r['label'],r['sample_id']
