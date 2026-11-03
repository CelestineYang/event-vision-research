"""Aligned full-polarity event projections for DVSAction point windows."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'thu_preprocessing'))
from gcstr_representations import _normalize,_histogram2d,_resize_official
VIEWS=('cstr3','xt_p','ty_p')

def crop(events):
 # Cached x is already mirrored by the author DAVIS decoder.
 x=events[:,1]-100;y=259-events[:,2]-50
 keep=(x>=0)&(x<150)&(y>=0)&(y<150)
 return np.column_stack((events[keep,0],x[keep],y[keep],events[keep,3]))

def render(events,start,end):
 if end<=start:raise ValueError('Invalid window duration')
 if len(events) and (np.any(events[:,0]<start) or np.any(events[:,0]>end) or np.any(np.diff(events[:,0])<0)):raise ValueError('Invalid temporal slice')
 e=crop(events);t=e[:,0].astype(np.float64);x=e[:,1].astype(np.int64);y=e[:,2].astype(np.int64);p=e[:,3]
 if not np.isin(p,[0,1]).all():raise ValueError('Invalid polarity')
 pos,neg=p==1,p==0
 def mean(mask):
  sums=np.zeros((150,150));counts=np.full((150,150),.01)
  np.add.at(sums,(x[mask],y[mask]),(t[mask]-start)/(end-start));np.add.at(counts,(x[mask],y[mask]),1.)
  return _normalize(sums/counts)
 density=_normalize(_histogram2d(x,y,(150,150),((0,149),(0,149))))
 result={'cstr3':_resize_official(np.stack([mean(neg),density,mean(pos)]),224)}
 for name,a,b,shape,ranges in [('xt_p',x,t,(150,224),((0,149),(start,end))),('ty_p',t,y,(224,150),((start,end),(0,149)))]:
  h=_normalize(np.stack([_histogram2d(a[m],b[m],shape,ranges) for m in (pos,neg)]))
  result[name]=_resize_official(np.concatenate([h,np.zeros((1,*shape))]),224)
 return result,len(e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--store',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert json.loads((a.store/'status.json').read_text())=={'status':'complete','scope':'full'}
 a.output.mkdir(parents=True,exist_ok=False)
 def save(n,obj):(a.output/n).write_text(json.dumps(obj,indent=2)+'\n')
 save('status.json',{'status':'running'})
 hashes={n:hashlib.sha256((a.store/n).read_bytes()).hexdigest() for n in ['config.json','train_windows.jsonl','test_windows.jsonl','train_files.jsonl','test_files.jsonl']}
 save('config.json',{'store':str(a.store),'source_hashes':hashes,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'projection_helpers_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'thu_preprocessing/gcstr_representations.py').read_bytes()).hexdigest(),'views':VIEWS,'shape':[3,224,224],'dtype':'float32','scope':'all valid point windows','events':'full raw polarity events within selected segment and window; no point sampling or SNN reconstruction','spatial':'x_cached-100, 259-y_cached-50; retain coordinates in [0,150); cached x already flipped','time':'fixed window start/end, same as project THU window renderer','normalization':'projection [0,1]; ImageNet normalization not applied','augmentation':False,'empty_crop':'all-zero projections, recorded; no silent window removal'})
 summary={}
 try:
  for split in ['train','test']:
   files={f['mark']:f for f in map(json.loads,(a.store/(split+'_files.jsonl')).read_text().splitlines())}
   for v in VIEWS:(a.output/split/v).mkdir(parents=True)
   previous=None;ev=None;count=0;empty=[]
   with (a.output/(split+'_frames.jsonl')).open('w') as manifest:
    for row in map(json.loads,(a.store/(split+'_windows.jsonl')).read_text().splitlines()):
     if not row['valid']:continue
     info=files[row['mark']]
     if previous!=row['mark']:ev=np.load(a.store/info['event_file'],mmap_mode='r');previous=row['mark']
     chunk=ev[row['raw_left']:row['raw_right']];images,n=render(chunk,row['start_us'],row['end_us']);paths={}
     for v,img in images.items():
      assert img.shape==(3,224,224) and img.dtype==np.float32 and np.isfinite(img).all() and img.min()>=0 and img.max()<=1
      rel=f"{split}/{v}/{row['point_row']}.npy";np.save(a.output/rel,img,allow_pickle=False)
      assert np.array_equal(np.load(a.output/rel,allow_pickle=False),img)
      paths[v]=rel
     record=row|{'event_file':info['event_file'],'raw_window_events':len(chunk),'cropped_events':n,'frames':paths}
     manifest.write(json.dumps(record)+'\n');count+=1
     if not n:empty.append(row['point_row'])
     if count%100==0:print(split,count,flush=True)
   summary[split]={'windows':count,'images':count*3,'empty_crop_point_rows':empty,'all_files_reread_exactly':True}
  save('audit.json',summary);save('status.json',{'status':'complete','scope':'full'});print(json.dumps(summary),flush=True)
 except BaseException as e:save('status.json',{'status':'failed','error':repr(e)});raise
if __name__=='__main__':main()
