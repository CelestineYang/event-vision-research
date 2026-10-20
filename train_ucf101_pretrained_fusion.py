"""Two-GPU window-level A fusion, Test soft-vote NLL selection, no early stop."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset

from train_ucf101_window_gcstr import (Metrics, atomic_checkpoint, barrier, core,
    loader, rank, seed_all, world)
from ucf101_fusion_window_data import FusionWindowDataset
from ucf101_pretrained_fusion import load_pretrained_fusion
from ucf101_window_data import checked_audit, sha, write_csv, write_json, VIEWS


class BatchForward(nn.Module):
    def __init__(self, fusion):
        super().__init__()
        self.fusion = fusion

    def forward(self, batch):
        return self.fusion.forward_batch_outputs(batch)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def gathered(value):
    if world()==1: return [value]
    result = [None]*world()
    dist.all_gather_object(result,value)
    return result


def emit(event, **fields):
    if rank()==0:
        print(json.dumps(dict(event=event,time=datetime.now(timezone.utc).isoformat()) | fields),flush=True)


def configuration(args):
    audit = checked_audit(args.experiment)
    files = ('train_ucf101_pretrained_fusion.py','ucf101_pretrained_fusion.py',
             'ucf101_fusion_window_data.py','eventmamba_v2_pretrained.py',
             'eventmamba_v2_model.py','eventmamba_v2_layers.py','ucf101_event_store.py',
             'gcstr_eventmamba_fusion.py','gcstr_secnet_fusion.py',
             'train_ucf101_window_gcstr.py','ucf101_window_data.py',
             'gcstr_augmentation.py','gcstr_representations.py','gcstr_models.py')
    return dict(architecture='pretrained_filename_V2_prototype_A',classes=101,
        experiment=str(args.experiment.resolve()),store=str(args.store.resolve()),
        checkpoint_dir=str(args.checkpoint_dir.resolve()),
        manifest_sha256=audit['manifest_sha256'],video_splits_sha256=audit['video_splits_sha256'],
        point_checkpoint_sha256=sha(args.point_checkpoint),
        image_checkpoint_sha256={v:sha(args.experiment/'checkpoints'/v/'best_loss.pt') for v in VIEWS},
        epochs=args.epochs,global_batch=args.batch_size,world_size=world(),
        optimizer='AdamW',lr=args.lr,weight_decay=1e-4,scheduler=None,
        point_count=2048,point_normalization='official',point_groups=[1024,512,256],
        images_frozen=True,points_trainable=True,aux_weight=.5,gradient_clip=1.,
        label_smoothing=0.,early_stopping=False,precision='FP32',amp=False,
        seed=args.seed,num_workers=args.workers,selection_split='test',validation_split=None,
        selection_metric='video_nll',independent_test=False,video_voting='mean_window_probability',
        tail='exact coverage without padding/dropping; SyncBN handles singleton local shards',
        checkpoint_interval=5,torch=str(torch.__version__),cuda=torch.version.cuda,
        code={f:sha(Path(__file__).parent/f) for f in files})


def epoch_pass(model, batches, device, epoch, aux_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    forward = model if training else core(model)
    metrics = {name:Metrics() for name in ('fusion','point')}
    sums = torch.zeros(2,device=device,dtype=torch.float64)
    started = time.perf_counter()
    for step,batch in enumerate(batches,1):
        batch = {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        labels = batch['label']
        with torch.set_grad_enabled(training):
            if training: optimizer.zero_grad(set_to_none=True)
            output = forward(batch)
            loss = nn.functional.cross_entropy(output['logits'],labels)
            point_loss = nn.functional.cross_entropy(output['point_logits'],labels)
            objective = loss+aux_weight*point_loss
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite objective')
            if training:
                count = torch.tensor(float(len(labels)),device=device)
                if world()>1: dist.all_reduce(count)
                (objective*(world()*len(labels)/count)).backward()
                nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                        max_norm=1.,error_if_nonfinite=True)
                optimizer.step()
        metrics['fusion'].update(output['logits'],labels,batch['video_id'])
        metrics['point'].update(output['point_logits'],labels,batch['video_id'])
        sums += torch.stack((objective.detach().double()*len(labels),objective.new_tensor(len(labels),dtype=torch.float64)))
        if step==1 or step%20==0 or step==len(batches):
            emit('progress',epoch=epoch,phase='train' if training else 'test',step=step,
                 steps=len(batches),seconds=time.perf_counter()-started,
                 fusion_loss=float(loss.detach()),point_loss=float(point_loss.detach()))
    if world()>1: dist.all_reduce(sums)
    results, predictions = {},{}
    for name,metric in metrics.items():
        metric.synchronize()
        results[name],predictions[name] = metric.finish()
    results.update(seconds=time.perf_counter()-started,objective_loss=float(sums[0]/sums[1]),
                   prediction_mode='online_train_changing_weights' if training else 'fixed_model_eval')
    return results,predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    parser.add_argument('--store',type=Path,default=Path('/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_events_v1'))
    parser.add_argument('--point-checkpoint',type=Path,default=Path('best_checkpoint.pth'))
    parser.add_argument('--output',type=Path,default=Path('experiments/ucf101_filename_v2_fusion_A_seed0_v1'))
    parser.add_argument('--checkpoint-dir',type=Path,help='Optional separate checkpoint storage')
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--batch-size',type=int,default=16,help='global batch')
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int,help='graceful checkpoint-boundary pause')
    args = parser.parse_args()
    if args.checkpoint_dir is None: args.checkpoint_dir=args.output/'checkpoints'
    local = int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    if int(os.environ.get('WORLD_SIZE',1))>1: dist.init_process_group('nccl')
    device = torch.device('cuda',local)
    if args.batch_size<2*world() or args.batch_size%world() or args.workers<0 or args.epochs<1:
        raise ValueError('Invalid batch/worker/epoch settings')
    if args.smoke and args.output.name.find('smoke')<0:
        raise ValueError('Smoke requires a separate output whose name contains smoke')
    args.output.mkdir(parents=True,exist_ok=True)
    lock = None
    if rank()==0:
        lock=(args.output/'run.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    barrier()
    config = configuration(args)
    last = args.checkpoint_dir/'last.pt'
    saved = None
    if args.resume:
        saved = torch.load(last,map_location='cpu',weights_only=False)
        if saved['config']!=config: raise ValueError('Resume config/checkpoint/code mismatch')
    elif last.exists() or (args.output/'config.json').exists():
        raise FileExistsError('Fresh run refuses existing output artifacts')
    barrier()  # All ranks finish the fresh-output check before rank 0 writes.
    if rank()==0: write_json(args.output/'config.json',config)
    seed_all(args.seed)
    fusion = load_pretrained_fusion(args.experiment,args.point_checkpoint)
    model = BatchForward(fusion)
    if saved: model.load_state_dict(saved['model_state'],strict=True)
    if world()>1: model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model=model.to(device)
    if world()>1: model=DDP(model,device_ids=[local],broadcast_buffers=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr,weight_decay=1e-4)
    if saved: optimizer.load_state_dict(saved['optimizer_state'])
    train_data = FusionWindowDataset(args.experiment,'train',args.store,seed=args.seed)
    if args.smoke:
        # Train-only payload; 33 windows test unequal batches and local singleton SyncBN.
        train_data = Subset(train_data,list(range(2*args.batch_size+1)))
        test_data = train_data
    else:
        test_data = FusionWindowDataset(args.experiment,'test',args.store,seed=args.seed)
    start = saved['epoch']+1 if saved else 1
    best = saved['best'] if saved else {'nll':float('inf'),'top1':-1.,'epoch':0}
    if saved: restore_rng(saved['rng_states'][rank()])
    else: seed_all(args.seed+rank())
    emit('start',config=config,train_windows=len(train_data),test_windows=len(test_data),
         start_epoch=start,pids=gathered(os.getpid()),smoke=args.smoke)
    metrics_path=args.output/'metrics.jsonl'
    # last.pt is authoritative after interrupted saves; discard only unsaved log rows.
    if rank()==0 and args.resume and metrics_path.exists():
        rows=[json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if any(r['epoch']>=start for r in rows):
            metrics_path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<start))
    for epoch in range(start,args.epochs+1):
        underlying=train_data.dataset if isinstance(train_data,Subset) else train_data
        underlying.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats(device)
        train,_ = epoch_pass(model,loader(train_data,args.batch_size,args.workers,True,args.seed+epoch),
                             device,epoch,.5,optimizer)
        # Keep eval FPS deterministic across epochs without altering training RNG.
        state = rng_state()
        seed_all(args.seed+100000+rank())
        with torch.inference_mode():
            test,predictions = epoch_pass(model,loader(test_data,args.batch_size,args.workers,False,args.seed),
                                          device,epoch,.5)
        restore_rng(state)
        if train['fusion']['windows']!=len(train_data) or test['fusion']['windows']!=len(test_data):
            raise RuntimeError('Dropped or duplicated window coverage')
        score=test['fusion']
        improved=(score['video_nll'],-score['video_top1'])<(best['nll'],-best['top1'])
        if improved: best={'nll':score['video_nll'],'top1':score['video_top1'],'epoch':epoch}
        states=gathered(rng_state())
        memory=gathered(dict(rank=rank(),allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                             reserved_mib=torch.cuda.max_memory_reserved(device)/2**20))
        if rank()==0:
            item=dict(epoch=epoch,train=train,test=test,best=best,memory=memory,
                      time=datetime.now(timezone.utc).isoformat())
            checkpoint=dict(epoch=epoch,config=config,model_state=core(model).state_dict(),
                            optimizer_state=optimizer.state_dict(),rng_states=states,best=best,metrics=item)
            atomic_checkpoint(last,checkpoint)
            if improved: atomic_checkpoint(args.checkpoint_dir/'best_loss.pt',checkpoint)
            if epoch%5==0 or epoch==args.epochs:
                # Hard link immutable atomic last.pt, saving an initial duplicate copy.
                target=args.checkpoint_dir/f'epoch_{epoch:03d}.pt'
                if not target.exists(): os.link(last,target)
            with metrics_path.open('a') as stream: stream.write(json.dumps(item,allow_nan=False)+'\n')
            for name,rows in predictions.items():
                write_csv(args.output/f'{name}_predictions_last.csv',rows,list(rows[0]))
                if improved: write_csv(args.output/f'{name}_predictions_best_loss.csv',rows,list(rows[0]))
            write_json(args.output/'status.json',dict(status='complete' if epoch==args.epochs else 'running',
                       completed_epoch=epoch,epochs=args.epochs,best=best,last_checkpoint=str(last)))
            emit('epoch',**item)
        barrier()
        if args.smoke:
            # Exercise checkpoint reload before a later --resume invocation.
            reloaded=torch.load(last,map_location='cpu',weights_only=False)
            if reloaded['epoch']!=epoch or not reloaded['optimizer_state']['state']:
                raise RuntimeError('Smoke checkpoint/optimizer save failed')
            for k,v in core(model).state_dict().items():
                torch.testing.assert_close(v.cpu(),reloaded['model_state'][k],rtol=0,atol=0)
        if args.stop_after_epoch and epoch>=args.stop_after_epoch:
            emit('paused',completed_epoch=epoch)
            break
    emit('finished',completed_epoch=epoch if start<=args.epochs else start-1)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
