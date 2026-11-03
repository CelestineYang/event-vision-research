"""DVSAction shared augmented events for training, verified frame cache for eval."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'thu_preprocessing'))
from gcstr_augmentation import augment_events
from gcstr_representations import GCSTRConfig
# Qualified import avoids colliding with the THU module of the same basename.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dvsaction_preprocessing.render_frames import render,crop,VIEWS

class Windows(Dataset):
 def __init__(self,store,frames,split,seed=0):
  self.store,self.frames,self.split,self.seed,self.epoch=Path(store),Path(frames),split,seed,0
  for p in [self.store,self.frames]:
   assert json.loads((p/'status.json').read_text())=={'status':'complete','scope':'full'}
  metadata=json.loads((self.frames/'config.json').read_text())
  for n,h in metadata['source_hashes'].items():
   assert hashlib.sha256((self.store/n).read_bytes()).hexdigest()==h,'Frame/data mismatch'
  self.rows=[json.loads(x) for x in (self.frames/(split+'_frames.jsonl')).read_text().splitlines()]
  for row in self.rows:row['sample_id']=row['filename']
  self.opened=OrderedDict()
 def __len__(self):return len(self.rows)
 def __getstate__(self):
  d=self.__dict__.copy();d['opened']=OrderedDict();return d
 def __getitem__(self,index):
  row=self.rows[index]
  if self.split=='train':
   name=row['event_file']
   if name not in self.opened:
    self.opened[name]=np.load(self.store/name,mmap_mode='r',allow_pickle=False)
    if len(self.opened)>8:self.opened.popitem(last=False)
   self.opened.move_to_end(name)
   events=crop(self.opened[name][row['raw_left']:row['raw_right']])
   seed=int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["sample_id"]}:{row["window_id"]}'.encode()).digest()[:8],'little')
   if len(events):
    t,x,y,p=augment_events(*[events[:,j] for j in range(4)],mode='spatial',rng=np.random.default_rng(seed),config=GCSTRConfig(sensor_width=150,sensor_height=150))
    # Invert the ROI mapping solely to reuse the exact cache renderer.
    events=np.column_stack([t,x+100,209-y,p])
   views,_=render(events,row['start_us'],row['end_us'])
  else:views={v:np.load(self.frames/row['frames'][v],allow_pickle=False) for v in VIEWS}
  mean=np.array([.485,.456,.406],dtype='f4')[:,None,None];std=np.array([.229,.224,.225],dtype='f4')[:,None,None]
  return {v:torch.from_numpy(np.asarray((views[v]-mean)/std,dtype='f4')) for v in VIEWS},row['label'],row['sample_id']
