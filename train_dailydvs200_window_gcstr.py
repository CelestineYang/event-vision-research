"""Three independent 200-class gCSTR branches; shared spatial augmentation, no fusion.

Single GPU only. Explicit --smoke is Train-only and runs one small epoch per call.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from gcstr_models import GCSTRBranchClassifier
from dailydvs200_prepared_data import PreparedDailyViews, identity as prepared_identity
from train_gcstr_feature_memory import PredictionMetrics, atomic_save, seed_all, rng_state, restore_rng
from ucf101_event_store import PreparedThreeViewDataset
from ucf101_window_data import VIEWS, checked_audit, sha, write_json


class DailyViews(PreparedThreeViewDataset):
    def __getitem__(self,index):
        if self.augmentation != 'none':
            return super().__getitem__(index)
        row=self.rows[index]
        images={}
        for v in VIEWS:
            image=np.load(self.output/'cache'/v/(row['cache_key']+'.npy'),allow_pickle=False)
            if image.shape!=(3,224,224) or not np.isfinite(image).all():
                raise ValueError('Invalid cached projection')
            image=(image-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
            images[v]=torch.from_numpy(image.astype(np.float32))
        return images,int(row['label_id']),row['video_id']


def epoch_pass(models, data, batch_size, workers, device, seed, optimizers=None):
    training=optimizers is not None
    for model in models.values(): model.train(training)
    metrics={v:PredictionMetrics() for v in VIEWS}
    batches=DataLoader(data,batch_size=batch_size,shuffle=training,drop_last=False,
        num_workers=workers,pin_memory=True,generator=torch.Generator().manual_seed(seed),
        **({'multiprocessing_context':'spawn'} if workers else {}))
    started=time.monotonic()
    with torch.set_grad_enabled(training):
        for step,(images,labels,ids) in enumerate(batches,1):
            labels=labels.to(device)
            for view in VIEWS:
                if training: optimizers[view].zero_grad(set_to_none=True)
                logits=models[view](images[view].to(device,non_blocking=True))
                loss=torch.nn.functional.cross_entropy(logits,labels)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                if training:
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in models[view].parameters() if p.grad is not None):
                        raise FloatingPointError('Nonfinite gradient')
                    optimizers[view].step()
                metrics[view].update(logits,labels,ids)
                del logits,loss
            if step==1 or step%50==0 or step==len(batches):
                print(json.dumps(dict(event='progress',phase='train' if training else 'eval',step=step,
                    steps=len(batches),seconds=time.monotonic()-started)),flush=True)
    results,predictions={},{}
    for view,metric in metrics.items():
        results[view],predictions[view]=metric.finish()
        if results[view]['windows']!=len(data): raise RuntimeError('Window coverage mismatch')
    return dict(branches=results,seconds=time.monotonic()-started,
                mode='online_train' if training else 'fixed_eval'),predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--frames',type=Path,help='Optional complete prepared_v1 frame cache for evaluation')
    p.add_argument('--npz-cache-mib',type=int,default=256,help='Per-worker decompressed recording cache budget')
    p.add_argument('--test-as-val',action='store_true',required=True)
    p.add_argument('--memory-fraction',type=float,default=.35,help='Per-process allocator ceiling, not GPU reservation')
    args=p.parse_args()
    if int(os.environ.get('WORLD_SIZE',1))!=1: p.error('Single GPU only')
    if args.batch_size<2 or args.epochs<1 or args.lr<=0 or args.workers<0 or not 0<args.memory_fraction<=1:
        p.error('Invalid budget')
    if args.smoke and 'smoke' not in args.output.name: p.error('Dedicated smoke output required')
    prepared=(args.data/'config.json').is_file() and (args.data/'train_windows.jsonl').is_file()
    if prepared:
        data_identity=prepared_identity(args.data)
        audit={'manifest_sha256':data_identity['train_windows.jsonl'],'video_splits_sha256':data_identity}
        store_sha=data_identity['config.json']
    else:
        complete=json.loads((args.data/'COMPLETE.json').read_text())
        if complete['status']!='complete' or complete['samples']!=18866:
            raise ValueError('Full DailyDVS preprocessing required')
        experiment,store=args.data/'experiment',args.data/'event_store'
        audit=checked_audit(experiment)
        classes=json.loads((experiment/'class_to_idx.json').read_text())
        if set(classes.values())!=set(range(200)): raise ValueError('Require 200 classes')
        store_sha=sha(store/'summary.json')
    weights=Path(torch.hub.get_dir())/'checkpoints'/Path(ResNet18_Weights.DEFAULT.url).name
    if not weights.is_file(): raise FileNotFoundError(f'ImageNet weights must already be cached: {weights}')
    config=dict(classes=200,epochs=args.epochs,batch_size=args.batch_size,workers=args.workers,lr=args.lr,
        seed=args.seed,optimizer='Adam',weight_decay=0.,scheduler=None,early_stopping=False,
        augmentation='shared window spatial',pretrained='ImageNet ResNet18 DEFAULT',pretrained_sha256=sha(weights),
        precision='FP32',smoke=args.smoke,memory_fraction=args.memory_fraction,
        selection_protocol='original_test_as_val',independent_test=False,selection_metric='video_nll',
        voting='mean window probabilities',manifest_sha256=audit['manifest_sha256'],
        splits_sha256=audit['video_splits_sha256'],store_sha256=store_sha,backend='prepared_v1' if prepared else 'legacy_event_store',npz_cache_mib=args.npz_cache_mib,frames=str(args.frames) if args.frames else None,frames_identity=sha(args.frames/'COMPLETE.json') if args.frames else None,
        code={f:sha(Path(__file__).parent/f) for f in ('train_dailydvs200_window_gcstr.py',
            'train_gcstr_feature_memory.py','ucf101_event_store.py','ucf101_window_data.py',
            'gcstr_models.py','gcstr_representations.py','gcstr_augmentation.py','dailydvs200_prepared_data.py')})
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume and any(f.name!='run.lock' for f in args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite existing output')
    saved=torch.load(args.output/'joint_last.pt',map_location='cpu',weights_only=False) if args.resume else None
    if saved and saved['config']!=config: raise ValueError('Resume identity mismatch')
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    def dataset(split):
        if prepared:return PreparedDailyViews(args.data,split,seed=args.seed,cache_mib=args.npz_cache_mib,frames=args.frames)
        return DailyViews(experiment,split,store,seed=args.seed)
    data={s:dataset(s) for s in (('train',) if args.smoke else ('train','test'))}
    if args.smoke:
        data['train'].rows=data['train'].rows[:2*args.batch_size+1]
        evaluation=dataset('train')
        evaluation.rows=list(data['train'].rows); evaluation.augmentation='none'
    else:
        evaluation=data['test']
        if {r['video_id'] for r in data['train'].rows}&{r['video_id'] for r in evaluation.rows}:
            raise ValueError('Cross-split samples')
    models,optimizers={},{}
    for v in VIEWS:
        seed_all(args.seed)
        models[v]=GCSTRBranchClassifier(v,num_classes=200,pretrained=saved is None).to(device)
        optimizers[v]=torch.optim.Adam(models[v].parameters(),lr=args.lr,weight_decay=0.)
    parameter_ids=[{id(p) for p in m.parameters()} for m in models.values()]
    assert all(not a&b for i,a in enumerate(parameter_ids) for b in parameter_ids[i+1:])
    start,best=1,{v:float('inf') for v in VIEWS}
    if saved:
        for v in VIEWS:
            models[v].load_state_dict(saved['models'][v],strict=True)
            optimizers[v].load_state_dict(saved['optimizers'][v])
        restore_rng(saved['rng']); start=saved['epoch']+1; best=saved['best']
    else: write_json(args.output/'config.json',config)
    del saved
    limit=min(args.epochs,start) if args.smoke else args.epochs
    for number in range(start,limit+1):
        data['train'].set_epoch(number)
        torch.cuda.reset_peak_memory_stats(device)
        before={v:models[v].backbone.fc.weight.detach().cpu().clone() for v in VIEWS} if args.smoke else {}
        trained,_=epoch_pass(models,data['train'],args.batch_size,args.workers,device,args.seed+number,optimizers)
        tested,predictions=epoch_pass(models,evaluation,args.batch_size,args.workers,device,args.seed)
        if args.smoke:
            for v in VIEWS:
                if torch.equal(before[v],models[v].backbone.fc.weight.detach().cpu()):
                    raise RuntimeError('Branch did not update')
        for v in VIEWS:
            folder=args.output/v; folder.mkdir(exist_ok=True)
            item=dict(epoch=number,train=trained['branches'][v],evaluation=tested['branches'][v])
            checkpoint=dict(view=v,epoch=number,model_state=models[v].state_dict(),config=config,metrics=item)
            if tested['branches'][v]['video_nll']<best[v]:
                best[v]=tested['branches'][v]['video_nll']
                atomic_save(folder/'best_video_nll.pt',checkpoint)
                write_json(folder/'predictions_best.json',predictions[v])
            atomic_save(folder/'last.pt',checkpoint)
            with (folder/'metrics.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
        atomic_save(args.output/'joint_last.pt',dict(config=config,epoch=number,best=best,rng=rng_state(),
            models={v:m.state_dict() for v,m in models.items()},optimizers={v:o.state_dict() for v,o in optimizers.items()}))
        if args.smoke:
            check=torch.load(args.output/'joint_last.pt',map_location='cpu',weights_only=False)
            for v in VIEWS:
                for k,tensor in models[v].state_dict().items():
                    torch.testing.assert_close(tensor.cpu(),check['models'][v][k],rtol=0,atol=0)
                if not check['optimizers'][v]['state']: raise RuntimeError('Missing optimizer state')
            del check
        status=dict(status='smoke_complete' if args.smoke else 'complete' if number==args.epochs else 'running',
            epoch=number,best_video_nll=best,train_windows=len(data['train']),eval_windows=len(evaluation),
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
            seconds=trained['seconds']+tested['seconds'],smoke=args.smoke)
        write_json(args.output/'status.json',status); print(json.dumps(status),flush=True)


if __name__=='__main__': main()
