"""Read-only end-to-end artifact verification after preprocessing."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,h5py
p=Path(sys.argv[1]);status=json.loads((p/'status.json').read_text());assert status=={'status':'complete','scope':'full'}
result={}
for split in ['train','test']:
 files=[json.loads(x) for x in (p/(split+'_files.jsonl')).read_text().splitlines()]
 windows=[json.loads(x) for x in (p/(split+'_windows.jsonl')).read_text().splitlines()]
 valid=[w for w in windows if w['valid']]
 assert [w['point_row'] for w in valid]==list(range(len(valid)))
 bymark={i['mark']:i for i in files}
 with h5py.File(p/(split+'.h5'),'r') as h:
  assert h.attrs['complete'] and h['data'].shape==(len(valid),2048,3)
  for i in files:
   ev=np.load(p/i['event_file'],mmap_mode='r');points=np.load(p/i['snn_file'],mmap_mode='r');lo,hi=i['selected_event_indices']
   ws=[w for w in windows if w['mark']==i['mark']]
   assert len(ws)==i['windows']
   for w in ws:
    assert lo<=w['raw_left']<=w['raw_right']<=hi
    assert len(points[w['left']:w['right']])==w['snn_points']
    if not w['valid']:continue
    j=w['point_row'];d=h['data'][j]
    assert h['label'][j]==w['label']==i['label'] and h['mark'][j]==i['mark']
    assert np.isfinite(d).all() and d.min()>=0 and d.max()<=1
    seed=int.from_bytes(hashlib.sha256(f"0:{split}:{i['filename']}:{w['window_id']}".encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);block=points[w['left']:w['right']];ix=np.arange(len(block));rng.shuffle(ix);sample=block[np.sort(ix[:2048])].copy();sample[:,0]-=sample[:,0].min()
    expected=sample.astype(np.float64);expected[:,0]/=sample[:,0].max();expected[:,1:]/=150
    assert np.array_equal(expected.astype(np.float32),d)
 result[split]={'recordings':len(files),'valid_windows':len(valid),'all_point_rows_recomputed_exactly':True,'raw_index_bounds_checked':True,'zero_valid_recordings':[i['filename'] for i in files if i['valid_windows']==0]}
(p/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
