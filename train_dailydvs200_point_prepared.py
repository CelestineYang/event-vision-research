"""Independent EventMamba V2 on current 1s DailyDVS-200 prepared windows."""
import argparse,fcntl,json,os,random,time,hashlib
from pathlib import Path
import h5py,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from eventmamba_v2_model import EventMamba
from eventmamba_v2_metrics import smoothed_loss
from train_gcstr_feature_memory import PredictionMetrics,atomic_save,seed_all,rng_state,restore_rng
from ucf101_window_data import sha,write_json

class Points(Dataset):
 def __init__(self,root,split,seed=0,training=False,limit=None):
  self.path=Path(root)/f'{split}.h5';self.seed=seed;self.training=training;self.epoch=0;self.h=None
  with h5py.File(self.path,'r') as h:
   assert h.attrs['complete'] and h['data'].shape[1:]==(2048,3)
   self.labels=h['label'][:];self.marks=h['mark'][:]
  assert self.labels.min()>=0 and self.labels.max()<200
  self.indices=np.arange(len(self.labels)) if limit is None else np.arange(min(limit,len(self.labels)))
 def __len__(self):return len(self.indices)
 def __getstate__(self):
  d=self.__dict__.copy();d['h']=None;return d
 def __getitem__(self,index):
  i=int(self.indices[index])
  if self.h is None:self.h=h5py.File(self.path,'r')
  a=self.h['data'][i].copy()
  if self.training:
   seed=int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{i}'.encode()).digest()[:8],'little');r=np.random.default_rng(seed)
   # Spatial augmentation on sampled points; time stays in the fixed 1-second frame.
   xy=a[:,1:].astype(np.float64);angle=r.uniform(-np.pi/12,np.pi/12);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
   xy=(xy-.5)@rot.T+.5;lo=r.uniform(0,.1,2);hi=1-r.uniform(0,.1,2)
   valid=((xy>=lo)&(xy<hi)).all(1)
   if valid.any():
    a=a[valid];a[:,1:]=((xy[valid]-lo)/(hi-lo)).astype('f4')
    ix=np.sort(r.choice(len(a),2048,replace=len(a)<2048));a=a[ix]
  if not np.isfinite(a).all() or (np.diff(a[:,0])<0).any():raise ValueError('Invalid points')
  return torch.from_numpy(a.T.copy()),int(self.labels[i]),str(self.marks[i])

def run(model,ds,device,batch,workers,epoch,opt=None):
 model.train(opt is not None);metric=PredictionMetrics();total=0.;n=0;beg=time.time()
 order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(ds.seed+epoch)).tolist() if opt else list(range(len(ds)))
 batches=[order[i:i+batch] for i in range(0,len(order),batch)]
 if opt and len(batches)>1 and len(batches[-1])==1:batches[-2].extend(batches.pop())
 loader=DataLoader(ds,batch_sampler=batches,num_workers=workers,pin_memory=True,**({'multiprocessing_context':'spawn'} if workers else {}))
 with torch.set_grad_enabled(opt is not None):
  for step,(x,y,ids) in enumerate(loader,1):
   x=x.to(device);y=y.to(device)
   if opt:opt.zero_grad(set_to_none=True)
   logits=model(x);loss=smoothed_loss(logits,y)
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   if opt:
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient')
    opt.step()
   metric.update(logits.detach(),y,ids);total+=loss.item()*len(y);n+=len(y)
   if step==1 or step%100==0 or step==len(loader):print(json.dumps({'phase':'train' if opt else 'eval','epoch':epoch,'step':step,'steps':len(loader),'windows':n,'seconds':time.time()-beg}),flush=True)
 result,pred=metric.finish();assert result['windows']==len(ds)
 result.update(smoothed_loss=total/n,seconds=time.time()-beg);return result,pred

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=150);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.batch_size<2 or a.workers<0 or a.epochs<1:raise ValueError('Invalid settings')
 if a.smoke and 'smoke' not in a.output.name:raise ValueError('Separate smoke output required')
 meta=json.loads((a.data/'config.json').read_text());status=json.loads((a.data/'status.json').read_text())
 assert status.get('state')=='complete' and status.get('scope')=='full' and meta['classes']==200 and meta['window_us']==1000000
 config={'dataset':'DailyDVS-200','classes':200,'epochs':a.epochs,'batch':a.batch_size,'workers':a.workers,'seed':a.seed,'architecture':'EventMamba V2','groups':[1024,512,256],'channels':[64,128,256],'neighbors':24,'point_count':2048,'initialization':'random; no pretrained/fusion checkpoint','optimizer':'AdamW','lr':.001,'weight_decay':.0001,'scheduler':'CosineAnnealingLR eta_min0','smoothing':'target .8, other .2/(C-1)','augmentation':'rotation +/-15 degrees and spatial crop on sampled points; fixed-window time preserved','BN':'trainable statistics and affine','precision':'FP32','selection':'video_top1 then lower NLL; original Test-as-val','independent_test':False,'smoke':a.smoke,'data':{f:sha(a.data/f) for f in ['config.json','train_windows.jsonl','test_windows.jsonl','train.h5','test.h5']},'code':{f:sha(Path(__file__).parent/f) for f in ['train_dailydvs200_point_prepared.py','eventmamba_v2_model.py','eventmamba_v2_layers.py','eventmamba_v2_metrics.py','train_gcstr_feature_memory.py']}}
 a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not a.resume and any(x.name!='run.lock' for x in a.output.iterdir()):raise ValueError('Output exists')
 seed_all(a.seed);device='cuda:0';model=EventMamba(200).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=0.)
 train=Points(a.data,'train',a.seed,True,65 if a.smoke else None);evaluation=Points(a.data,'train' if a.smoke else 'test',a.seed,False,65 if a.smoke else None)
 best=(-1.,float('inf'));bestnll=float('inf');start=1
 if a.resume:
  saved=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False)
  if saved['config']!=config:raise ValueError('Resume identity changed')
  model.load_state_dict(saved['net'],strict=True);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);restore_rng(saved['rng']);start=saved['epoch']+1;best=tuple(saved['best']);bestnll=saved['best_nll'];del saved
 else:write_json(a.output/'config.json',config)
 limit=min(start,a.epochs) if a.smoke else a.epochs
 for epoch in range(start,limit+1):
  train.epoch=epoch;torch.cuda.reset_peak_memory_stats();lr=opt.param_groups[0]['lr']
  tr,_=run(model,train,device,a.batch_size,a.workers,epoch,opt);ev,pred=run(model,evaluation,device,a.batch_size,a.workers,epoch)
  candidate=(ev['video_top1'],ev['video_nll']);improved=(candidate[0],-candidate[1])>(best[0],-best[1]);improvednll=candidate[1]<bestnll
  if improved:best=candidate
  if improvednll:bestnll=candidate[1]
  sched.step();record={'epoch':epoch,'train':tr,'evaluation':ev,'lr':lr,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
  state={'net':model.state_dict(),'epoch':epoch,'acc':ev['video_top1'],'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng_state(),'best':best,'best_nll':bestnll,'config':config,'metrics':record}
  atomic_save(a.output/'last.pt',state)
  if improved:atomic_save(a.output/'best_video_top1.pt',state);write_json(a.output/'predictions_best_top1.json',pred)
  if improvednll:atomic_save(a.output/'best_video_nll.pt',state)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  write_json(a.output/'status.json',{'status':'smoke_complete' if a.smoke else 'complete' if epoch==a.epochs else 'running','epoch':epoch,'epochs':a.epochs,'best_video_top1':best[0],'best_video_nll':bestnll});print(json.dumps(record),flush=True)
if __name__=='__main__':main()
