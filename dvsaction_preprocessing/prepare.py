"""DVSAction author-compatible point preprocessing with full-event provenance."""
import argparse, ast, contextlib, hashlib, io, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import h5py

SOURCE=Path(__file__).with_name('upstream_generate_action.py')
ns={'np':np,'cv2':cv2,'os':os,'struct':__import__('struct')}
tree=ast.parse(SOURCE.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))],type_ignores=[]),str(SOURCE),'exec'),ns)
CLASSES=['arm crossing','get-up','kicking','picking up','jumping','sit-down','throwing','turning around','walking','waving']
def dump(p,x):
 p.write_text(json.dumps(x,indent=2)+'\n')
def decode(p):
 with p.open('rb') as f:
  while True:
   pos=f.tell(); line=f.readline()
   if not line.startswith(b'#'): f.seek(pos); break
  b=f.read()
 if len(b)%8: raise ValueError(f'Incomplete record: {p}')
 a=np.frombuffer(b,dtype='>u4').reshape(-1,2)
 x=345-((a[:,0].astype(np.int64)&0x003ff000)>>12); y=(a[:,0]&0x7fc00000)>>22
 good=((a[:,0]&0x80000000)==0)&(x>=0)&(x<346)&(y<260)
 ev=np.column_stack([a[:,1],x,y,1-((a[:,0]&0x800)>>11).astype(np.int64)])[good]
 # Author decoder skips the final physical record, not necessarily final DVS event.
 compat=ev[:-1] if len(a) and good[-1] else ev
 return ev,compat

def process(job):
 root,out,split,rel,mark=job; root=Path(root); out=Path(out)
 ev,compat=decode(root/rel)
 if len(ev)==0: raise ValueError('Empty events: '+rel)
 raw_decreases=np.flatnonzero(np.diff(ev[:,0])<0)
 sid=f'{mark:04d}'; dest=out/split; dest.mkdir(exist_ok=True)
 np.save(dest/(sid+'_events.npy'),ev)
 lo=len(compat)//2; hi=lo+int(2*len(compat)/5); selected=compat[lo:hi]
 if len(selected)==0 or np.any(np.diff(selected[:,0])<0): raise ValueError('Empty/nonmonotone selected interval: '+rel)
 snn=ns['SNN']()
 with contextlib.redirect_stdout(io.StringIO()): snn.spiking(selected)
 points=np.column_stack([snn.T,snn.X,snn.Y]).astype(np.int64)
 np.save(dest/(sid+'_snn.npy'),points)
 windows=[]; sampled=[]
 if len(points):
  ts=points[:,0]; start=int(ts[0]); wid=0
  while start+500000<int(ts[-1]):
   end=start+500000; left=int(np.searchsorted(ts,start)); right=int(np.searchsorted(ts,end,side='right'))-1
   w=points[left:right]; valid=len(w)>5000
   row={'sample_id':sid,'filename':rel,'label':CLASSES.index(rel.split('/')[0]),'mark':mark,'window_id':wid,'start_us':start,'end_us':end,'left':left,'right':right,'snn_points':len(w),'valid':False,'reason':'insufficient_snn_points','point_row':None,
        'raw_left':lo+int(np.searchsorted(selected[:,0],start)),'raw_right':lo+int(np.searchsorted(selected[:,0],end,side='right'))}
   if valid:
    seed=int.from_bytes(hashlib.sha256(f'0:{split}:{rel}:{wid}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed); idx=np.arange(len(w));rng.shuffle(idx);idx=np.sort(idx[:2048]); chosen=w[idx].copy();chosen[:,0]-=chosen[:,0].min()
    if chosen[:,0].max()>0:
     data=chosen.astype(np.float64);data[:,0]/=chosen[:,0].max();data[:,1:]/=150
     sampled.append(data.astype(np.float32));row.update(valid=True,reason=None,point_row=len(sampled)-1)
    else:row['reason']='zero_sampled_time_span'
   windows.append(row);start+=250000;wid+=1
 arr=np.stack(sampled) if sampled else np.empty((0,2048,3),np.float32)
 np.save(dest/(sid+'_points.npy'),arr)
 info={'sample_id':sid,'filename':rel,'label':CLASSES.index(rel.split('/')[0]),'mark':mark,'event_file':f'{split}/{sid}_events.npy','snn_file':f'{split}/{sid}_snn.npy','raw_events':len(ev),'raw_timestamp_decreases':[{'index':int(i),'before':int(ev[i,0]),'after':int(ev[i+1,0])} for i in raw_decreases],'author_decoded_events':len(compat),'selected_event_indices':[lo,hi],'selected_time_us':[int(selected[0,0]),int(selected[-1,0])] if len(selected) else None,'snn_points':len(points),'windows':len(windows),'valid_windows':len(arr),'source_sha256':hashlib.sha256((root/rel).read_bytes()).hexdigest()}
 return info,windows,str(dest/(sid+'_points.npy'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--train-list',required=True);ap.add_argument('--test-list',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 dump(out/'status.json',{'status':'running','scope':'smoke' if args.smoke else 'full'})
 lists={s:[x.strip() for x in Path(getattr(args,s+'_list')).read_text().splitlines() if x.strip()] for s in ['train','test']}
 assert not set(lists['train'])&set(lists['test'])
 for rows in lists.values():
  assert len(rows)==len(set(rows))
  assert all((Path(args.root)/r).is_file() and r.split('/')[0] in CLASSES for r in rows)
 config=vars(args)|{'classes':CLASSES,'window_us':500000,'step_us':250000,'points':2048,'seed':0,'upstream_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'point_input':'author SNN reconstructed pixels; polarity absent','raw_columns':['t','x','y','p'],'spatial_transform':'author x inversion; SNN vertical flip then rows[50:200],cols[100:250]','sampling':'deterministic per-window RNG, differs from unseeded author RNG','boundary':'end < last; last <=end index excluded','snn':'original class unmodified including 180-bound inhibition and coordinate ordering','image_cache_created':False}
 dump(out/'config.json',config)
 audit={'unlisted':sorted(str(p.relative_to(args.root)) for p in Path(args.root).rglob('*.aedat') if str(p.relative_to(args.root)) not in set(lists['train']+lists['test']))}
 try:
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   for split,rows in lists.items():
    if args.smoke:rows=rows[:1]
    infos=[];total=0;count=0
    with h5py.File(out/(split+'.h5'),'w') as h, (out/(split+'_windows.jsonl')).open('w') as wf,(out/(split+'_files.jsonl')).open('w') as sf:
     ds=h.create_dataset('data',(0,2048,3),maxshape=(None,2048,3),dtype='f4',chunks=(1,2048,3));ls=h.create_dataset('label',(0,),maxshape=(None,),dtype='i2');ms=h.create_dataset('mark',(0,),maxshape=(None,),dtype='i4')
     for info,windows,fp in pool.map(process,[(args.root,str(out),split,r,i) for i,r in enumerate(rows)]):
      data=np.load(fp);n=len(data)
      ds.resize(total+n,axis=0);ls.resize(total+n,axis=0);ms.resize(total+n,axis=0)
      ds[total:]=data;ls[total:]=info['label'];ms[total:]=info['mark']
      for w in windows:
       if w['point_row'] is not None:w['point_row']+=total
       wf.write(json.dumps(w)+'\n')
      sf.write(json.dumps(info)+'\n');infos.append(info);total+=n;count+=len(windows);Path(fp).unlink()
      print(split,len(infos),len(rows),info['filename'],'valid',n,flush=True)
     h.attrs['complete']=True
    audit[split]={'recordings':len(infos),'windows':count,'valid_windows':total,'zero_valid_recordings':[i['filename'] for i in infos if not i['valid_windows']]}
  dump(out/'audit.json',audit);dump(out/'status.json',{'status':'complete','scope':'smoke' if args.smoke else 'full'})
 except BaseException as e:
  dump(out/'status.json',{'status':'failed','error':repr(e)});raise
if __name__=='__main__':main()
