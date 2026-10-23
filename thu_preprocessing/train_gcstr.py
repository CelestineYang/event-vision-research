"""Three independent CHL ResNet-18 branches sharing full-event window loading."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from prepare_chl import sha, write_json
from render_frames import render
from gcstr_representations import GCSTRConfig
from gcstr_augmentation import augment_events
from gcstr_models import GCSTRBranchClassifier

VIEWS = ('cstr3', 'xt_p', 'ty_p')


class Windows(Dataset):
    def __init__(self, store, split, seed=0):
        self.store, self.split, self.seed, self.epoch = Path(store), split, seed, 0
        if json.loads((self.store/'status.json').read_text())['state'] != 'complete':
            raise ValueError('Preprocessing not complete')
        self.rows = [r for line in (self.store/(split+'_windows.jsonl')).read_text().splitlines()
                     if (r := json.loads(line))['valid']]
        if not self.rows: raise ValueError('Empty window split')
        self.opened = OrderedDict()

    def __len__(self): return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy(); state['opened'] = OrderedDict(); return state

    def __getitem__(self, index):
        import hashlib
        row = self.rows[index]
        name = row['event_file']
        if name not in self.opened:
            self.opened[name] = np.load(self.store/name, mmap_mode='r', allow_pickle=False)
            if len(self.opened)>8: self.opened.popitem(last=False)
        self.opened.move_to_end(name)
        events = self.opened[name][row['left']:row['right']]
        if self.split == 'train':
            seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["sample_id"]}:{row["window_id"]}'.encode()).digest()[:8], 'little')
            t,x,y,p = augment_events(events[:,2], events[:,0], events[:,1], events[:,3],
                                    mode='spatial', rng=np.random.default_rng(seed),
                                    config=GCSTRConfig(sensor_width=346,sensor_height=260))
            events = np.column_stack([x,y,t,p])
        views = render(events,row['start_us'],row['end_us'])
        mean=np.array([.485,.456,.406],dtype='f4')[:,None,None]
        std=np.array([.229,.224,.225],dtype='f4')[:,None,None]
        return {v:torch.from_numpy(np.asarray((views[v]-mean)/std,dtype='f4')) for v in VIEWS}, row['label'], row['sample_id']


class Metrics:
    def __init__(self):
        self.n=0;self.nll=0.;self.top1=0;self.top5=0;self.videos={}

    def update(self, logits, labels, ids):
        lp=logits.detach().double().log_softmax(-1).cpu();labels=labels.cpu()
        self.n+=len(labels);self.nll+=float(-lp.gather(1,labels[:,None]).sum())
        self.top1+=int((lp.argmax(-1)==labels).sum())
        self.top5+=int((lp.topk(min(5,lp.shape[1]),dim=-1).indices==labels[:,None]).any(-1).sum())
        for value,label,name in zip(lp,labels,ids):
            label=int(label)
            if name not in self.videos:self.videos[name]=[value.clone(),label,1]
            else:
                total,old,count=self.videos[name]
                if old!=label:raise ValueError('Video label changed')
                self.videos[name]=[torch.logaddexp(total,value),label,count+1]

    def finish(self):
        predictions=[]
        for name,(total,label,count) in sorted(self.videos.items()):
            lp=total-np.log(count)
            predictions.append(dict(sample_id=name,label=label,prediction=int(lp.argmax()),
                                    nll=float(-lp[label]),top5=bool((lp.topk(min(5,len(lp))).indices==label).any()),windows=count))
        return dict(windows=self.n,videos=len(predictions),window_nll=self.nll/self.n,
                    window_top1=self.top1/self.n,window_top5=self.top5/self.n,
                    video_nll=float(np.mean([r['nll'] for r in predictions])),
                    video_top1=float(np.mean([r['prediction']==r['label'] for r in predictions])),
                    video_top5=float(np.mean([r['top5'] for r in predictions]))),predictions


def epoch(models, loader, device, optimizers=None):
    metrics={v:Metrics() for v in VIEWS}
    for model in models.values():model.train(optimizers is not None)
    with torch.set_grad_enabled(optimizers is not None):
        for images,labels,ids in loader:
            labels=labels.to(device)
            for view in VIEWS:
                if optimizers:optimizers[view].zero_grad(set_to_none=True)
                logits=models[view](images[view].to(device,non_blocking=True))
                loss=torch.nn.functional.cross_entropy(logits,labels)
                if not torch.isfinite(loss):raise FloatingPointError(view)
                if optimizers:loss.backward();optimizers[view].step()
                metrics[view].update(logits,labels,ids)
    return {v:m.finish() for v,m in metrics.items()}


def save(path, value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def load_frozen_branch(path, device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=GCSTRBranchClassifier(saved['view'],num_classes=50,pretrained=False)
    model.load_state_dict(saved['model_state'],strict=True)
    return model.to(device).requires_grad_(False).eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--test-as-val',action='store_true',required=True,
                   help='Explicitly acknowledge original Test selects video-NLL checkpoint')
    p.add_argument('--smoke',action='store_true',help='Two complete videos per split, one epoch, random initialization')
    args=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        p.error('This portable entry is single-process; do not launch with torchrun')
    if args.batch_size<2 or args.epochs<1 or args.workers<0 or args.lr<=0:p.error('Invalid budget')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    status=json.loads((args.store/'status.json').read_text())
    if status.get('scope')!='full' and not args.smoke:raise ValueError('Full training requires full preprocessing')
    data={s:Windows(args.store,s,args.seed) for s in ('train','test')}
    if {r['sample_id'] for r in data['train'].rows}&{r['sample_id'] for r in data['test'].rows}:
        raise ValueError('Cross-split video leakage')
    if args.smoke:
        args.epochs=1
        for ds in data.values():
            selected=sorted({r['sample_id'] for r in ds.rows})[:2]
            ds.rows=[r for r in ds.rows if r['sample_id'] in selected]
    identity={f:sha(args.store/f) for f in ('config.json','samples.json','train_windows.jsonl','test_windows.jsonl')}
    code={f:sha(Path(__file__).parent/f) for f in ('train_gcstr.py','render_frames.py','gcstr_models.py','gcstr_augmentation.py','gcstr_representations.py')}
    config=dict(epochs=args.epochs,batch_size=args.batch_size,workers=args.workers,lr=args.lr,seed=args.seed,
                classes=50,optimizer='Adam',weight_decay=0.,scheduler=None,early_stopping=False,
                augmentation='spatial',pretrained=not args.smoke,smoke=args.smoke,identity=identity,code=code,
                selection_protocol='original_test_as_val',independent_test=False,selection_metric='video_nll',
                video_vote='arithmetic mean of window probabilities',window_selection='valid official Point windows')
    checkpoint=args.output/'resume.pt'
    restored=torch.load(checkpoint,map_location='cpu',weights_only=False) if args.resume else None
    if restored and restored['config']!=config:raise ValueError('Resume config/data/code mismatch')
    if not args.resume:
        args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'config.json',config)
    models={v:GCSTRBranchClassifier(v,50,pretrained=not args.smoke and not args.resume).to(args.device) for v in VIEWS}
    optimizers={v:torch.optim.Adam(m.parameters(),lr=args.lr) for v,m in models.items()}
    best={v:float('inf') for v in VIEWS};start=1
    if restored:
        for v in VIEWS:
            models[v].load_state_dict(restored['models'][v]);optimizers[v].load_state_dict(restored['optimizers'][v])
        best=restored['best'];start=restored['epoch']+1
        torch.set_rng_state(restored['torch_rng']);np.random.set_state(restored['numpy_rng']);random.setstate(restored['python_rng'])
        if torch.cuda.is_available() and restored['cuda_rng']:torch.cuda.set_rng_state_all(restored['cuda_rng'])
    for number in range(start,args.epochs+1):
        begin=time.perf_counter();data['train'].epoch=number
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        def loader(split):
            return DataLoader(data[split],batch_size=args.batch_size,shuffle=split=='train',
                  generator=torch.Generator().manual_seed(args.seed+number),num_workers=args.workers,
                  pin_memory=args.device.startswith('cuda'),collate_fn=None,
                  **({'multiprocessing_context':'spawn'} if args.workers else {}))
        trained=epoch(models,loader('train'),args.device,optimizers)
        tested=epoch(models,loader('test'),args.device)
        for v in VIEWS:
            folder=args.output/v;folder.mkdir(exist_ok=True)
            record=dict(epoch=number,train=trained[v][0],test=tested[v][0],seconds=time.perf_counter()-begin,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
            saved=dict(epoch=number,view=v,model_state=models[v].state_dict(),config=config,metrics=record)
            save(folder/'last.pt',saved)
            if tested[v][0]['video_nll']<best[v]:
                best[v]=tested[v][0]['video_nll'];save(folder/'best_loss.pt',saved)
                write_json(folder/'predictions_best.json',tested[v][1])
            with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        save(checkpoint,dict(epoch=number,config=config,best=best,models={v:m.state_dict() for v,m in models.items()},
              optimizers={v:o.state_dict() for v,o in optimizers.items()},torch_rng=torch.get_rng_state(),
              numpy_rng=np.random.get_state(),python_rng=random.getstate(),
              cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        write_json(args.output/'status.json',dict(state='complete' if number==args.epochs else 'running',epoch=number,best_video_nll=best))
        print(json.dumps(dict(epoch=number,test={v:tested[v][0] for v in VIEWS},seconds=time.perf_counter()-begin)),flush=True)


if __name__=='__main__':main()
