"""Render three-view caches for prepared_v1; resume validates saved checksums."""
import argparse,json,fcntl
from pathlib import Path
import numpy as np
from dailydvs200_prepared_data import PreparedDailyViews,identity
from ucf101_window_data import VIEWS,sha,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 config={'identity':identity(a.data),'scope':'smoke' if a.smoke else 'full','code':{n:sha(Path(__file__).parent/n) for n in ['render_dailydvs200_prepared.py','dailydvs200_prepared_data.py','ucf101_window_data.py','gcstr_representations.py']}}
 if not a.resume:a.output.mkdir(parents=True,exist_ok=False)
 lock=(a.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.resume:
  if json.loads((a.output/'config.json').read_text())!=config:raise ValueError('Resume identity changed')
 else:write_json(a.output/'config.json',config)
 (a.output/'COMPLETE.json').unlink(missing_ok=True)
 write_json(a.output/'status.json',{'status':'running'})
 counts={}
 for split in ['train','test']:
  ds=PreparedDailyViews(a.data,split);ds.augmentation='none'
  if a.smoke:ds.rows=ds.rows[:3]
  records=[]
  for v in VIEWS:(a.output/split/v).mkdir(parents=True,exist_ok=True)
  for i,row in enumerate(ds.rows):
   receipt=a.output/split/f"{row['point_row']}.json"
   if receipt.exists():
    rec=json.loads(receipt.read_text())
    if rec['window']!=row:raise ValueError('Window receipt mismatch')
    for v,checksum in rec['hashes'].items():
     if sha(a.output/split/v/f"{row['point_row']}.npy")!=checksum:raise ValueError('Cached frame checksum mismatch')
   else:
    images=ds.images(row);hashes={}
    for v,img in images.items():
     f=a.output/split/v/f"{row['point_row']}.npy";tmp=f.with_suffix('.tmp')
     with tmp.open('wb') as stream:np.save(stream,img,allow_pickle=False)
     tmp.replace(f)
     if not np.array_equal(img,np.load(f,allow_pickle=False)):raise ValueError('Frame write mismatch')
     hashes[v]=sha(f)
    rec={'window':row,'hashes':hashes};write_json(receipt,rec)
   records.append(rec)
   if (i+1)%1000==0:print(split,i+1,flush=True)
  with (a.output/f'{split}_frames.jsonl').open('w') as f:
   for rec in records:f.write(json.dumps(rec)+'\n')
  counts[split]=len(records)
 write_json(a.output/'COMPLETE.json',config|{'status':'complete','windows':counts,'normalization':'float32 [3,224,224], [0,1], no ImageNet normalization or augmentation'})
 write_json(a.output/'status.json',{'status':'complete','scope':config['scope']});print(counts)
if __name__=='__main__':main()
