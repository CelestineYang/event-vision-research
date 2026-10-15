"""Sequential independent gCSTR pretraining (explicit --smoke or --train).

Latest user protocol: official Train/Test only; Test is used for selection.
This is NOT an independent held-out Test estimate. No early stopping.
"""
import argparse
import fcntl
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from gcstr_models import GCSTRBranchClassifier
from ucf101_window_data import (OUTPUT, VIEWS, WindowDataset, checked_audit,
                                sha, write_csv, write_json)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def world(): return dist.get_world_size() if dist.is_initialized() else 1


def rank(): return dist.get_rank() if dist.is_initialized() else 0


def barrier():
    if dist.is_initialized(): dist.barrier()


def core(model): return model.module if isinstance(model, DDP) else model


def wrap(model, device):
    if world() == 1: return model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    return DDP(model, device_ids=[device.index])


class ExactBatchSampler:
    """Partition global batches without padding/dropping any window.

    Redistribute a singleton final tail from its preceding batch, so all ranks
    execute the same number of nonempty steps without exceeding global batch.
    """
    def __init__(self, size, batch, replicas, replica, seed):
        if size < replicas or batch < 2*replicas:
            raise ValueError('Need >=world windows and global batch >=2*world')
        order = torch.randperm(size,generator=torch.Generator().manual_seed(seed)).tolist()
        chunks = [order[i:i+batch] for i in range(0,size,batch)]
        if len(chunks)>1 and len(chunks[-1])<replicas:
            needed = replicas-len(chunks[-1])
            chunks[-1] = chunks[-2][-needed:]+chunks[-1]
            chunks[-2] = chunks[-2][:-needed]
        self.batches = [chunk[replica::replicas] for chunk in chunks]

    def __iter__(self): return iter(self.batches)

    def __len__(self): return len(self.batches)


def loader(data, batch, workers, shuffle, seed):
    if world()>1:
        common = dict(num_workers=workers,pin_memory=True,worker_init_fn=worker_seed,
                      multiprocessing_context='spawn' if workers else None,
                      generator=torch.Generator().manual_seed(seed+rank()))
        if shuffle:
            return DataLoader(data,batch_sampler=ExactBatchSampler(len(data),batch,world(),rank(),seed),**common)
        # Evaluation needs no synchronized forwards and no sampler padding.
        return DataLoader(Subset(data,list(range(rank(),len(data),world()))),
                          batch_size=max(1,batch//world()),shuffle=False,drop_last=False,**common)
    return DataLoader(data, batch_size=batch, shuffle=shuffle, drop_last=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(seed))


class Metrics:
    """Window CE and equal-video soft vote. Hard-vote ties use lowest class ID."""
    def __init__(self):
        self.n = self.ce = self.top1 = self.top5 = 0
        self.videos = {}

    def update(self, logits, labels, ids):
        logp = logits.detach().double().log_softmax(-1).cpu()
        labels = labels.cpu()
        probs = logp.exp().numpy()
        predicted = logp.argmax(-1)
        self.n += len(labels)
        self.ce += float(-logp[torch.arange(len(labels)), labels].sum())
        self.top1 += int((predicted == labels).sum())
        self.top5 += int((logp.topk(5).indices == labels[:,None]).any(1).sum())
        for i, (video, label) in enumerate(zip(ids, labels.tolist())):
            if video not in self.videos:
                self.videos[video] = [label, 0, np.zeros(101), np.zeros(101, dtype=np.int64)]
            item = self.videos[video]
            if item[0] != label: raise ValueError('Video label mismatch')
            item[1] += 1
            item[2] += probs[i]
            item[3][int(predicted[i])] += 1

    def finish(self):
        if not self.n: raise ValueError('Empty evaluation')
        rows, ce, top1, top5, hard = [], 0., 0, 0, 0
        for video in sorted(self.videos):
            label, n, sums, votes = self.videos[video]
            p = sums / n
            # Stable ranking fixes probability ties by ascending class ID.
            rank = np.argsort(-p, kind='stable')
            nll = -float(np.log(max(float(p[label]), np.finfo(np.float64).tiny)))
            ce += nll
            top1 += int(rank[0] == label)
            top5 += int(label in rank[:5])
            hard += int(votes.argmax() == label)
            rows.append(dict(video_id=video,label_id=label,windows=n,prediction=int(rank[0]),
                             top5=json.dumps(rank[:5].tolist()), hard_prediction=int(votes.argmax()),
                             true_class_probability=float(p[label]), nll=nll))
        v = len(rows)
        return dict(windows=self.n, videos=v, window_loss=self.ce/self.n,
                    window_top1=self.top1/self.n, window_top5=self.top5/self.n,
                    video_nll=ce/v, video_top1=top1/v, video_top5=top5/v,
                    hard_vote_video_top1=hard/v), rows

    def synchronize(self):
        if world()==1: return
        parts = [None]*world()
        dist.all_gather_object(parts,self.__dict__)
        self.__init__()
        for part in parts:
            for key in ('n','ce','top1','top5'):
                setattr(self,key,getattr(self,key)+part[key])
            for video, item in part['videos'].items():
                if video not in self.videos:
                    self.videos[video] = [item[0],item[1],item[2].copy(),item[3].copy()]
                else:
                    target = self.videos[video]
                    if target[0] != item[0]: raise ValueError('Cross-rank label mismatch')
                    for i in (1,2,3): target[i] += item[i]


def epoch_pass(model, batches, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    forward_model = model if training else core(model)
    metrics = Metrics()
    start = time.perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels, ids in batches:
            images, labels = images.to(device,non_blocking=True), labels.to(device,non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = forward_model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            if training:
                # DDP averages rank gradients; correct unequal final shard sizes.
                count = torch.tensor(float(len(labels)),device=device)
                if world()>1: dist.all_reduce(count)
                (loss*(world()*len(labels)/count)).backward()
                optimizer.step()
            metrics.update(logits,labels,ids)
    metrics.synchronize()
    result, rows = metrics.finish()
    result['seconds'] = time.perf_counter()-start
    result['prediction_mode'] = 'online_train_changing_weights' if training else 'fixed_model_eval'
    return result, rows


def atomic_checkpoint(path, state):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def load_checkpoint(path):
    # Own checkpoints contain optimizer and config, not only model tensors.
    return torch.load(path,map_location='cpu',weights_only=False)


def protocol_config(args, audit):
    base = Path(__file__).resolve().parent
    return dict(dataset='UCF101-DVS', num_classes=101, views=list(VIEWS),
                world_size=world(), parallelism='DDP+SyncBatchNorm' if world()>1 else 'single',
                per_rank_batch=args.batch_size//world(), num_workers_per_rank=args.num_workers,
                tail_policy='redistribute tiny tail; no duplicate or dropped windows',
                pretrained='ImageNet1K_V1', epochs=args.epochs, batch_size=args.batch_size,
                optimizer='Adam', lr=args.lr, weight_decay=0., scheduler=None,
                early_stopping=False, label_smoothing=0., augmentation='spatial', amp=False,
                augmentation_config=dict(train_only=True,coordinate_dither_pixels=[-.5,.5],
                                         rotation_degrees=[-15.,15.],crop_per_side=[0.,.1],
                                         versions_per_window_per_epoch=1,temporal=False,polarity_flip=False,
                                         rng='seed+epoch+video_id+window_id; independent of rank/view',
                                         empty_crop_policy='existing augment_events retains pre-crop events'),
                precision='float32', seed=args.seed, num_workers=args.num_workers,
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                selection_split='test', independent_test=False,
                selection_metric='video_nll', validation_split=None,
                tie_rules='NLL then higher video Top1 then earlier epoch; Top1 then lower NLL then earlier epoch',
                hard_vote_tie_rule='lowest class ID',
                train_coverage='every valid window once per epoch, drop_last=False',
                data_config=audit['config'], manifest_sha256=audit['manifest_sha256'],
                code={p:sha(base/p) for p in ('train_ucf101_window_gcstr.py','ucf101_window_data.py',
                                             'gcstr_models.py','gcstr_representations.py','gcstr_augmentation.py')})


def smoke(args, config, device):
    """Explicit future command; train subset only, never formal run checkpoints."""
    folder = args.output / 'smoke'
    if rank()==0: folder.mkdir(exist_ok=True)
    barrier()
    results = {}
    for view in VIEWS:
        seed_all(args.seed)
        data = WindowDataset(args.output,'train',view,seed=args.seed)
        data.set_epoch(1)
        subset = Subset(data,list(range(min(2*args.batch_size+1,len(data)))))
        if len(subset) < 2: raise RuntimeError('Need at least two smoke windows')
        model = wrap(GCSTRBranchClassifier(view,num_classes=101,pretrained=True),device)
        opt = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.)
        before = core(model).backbone.fc.weight.detach().clone()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        result, _ = epoch_pass(model,loader(subset,args.batch_size,args.num_workers,True,args.seed),device,opt)
        if torch.equal(before,core(model).backbone.fc.weight): raise RuntimeError('No classifier update')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite smoke gradients')
        path = folder / f'{view}.pt'
        if rank()==0: atomic_checkpoint(path,dict(model_state=core(model).state_dict(),optimizer_state=opt.state_dict(),
                                    view=view,epoch=1,config=config))
        barrier()
        saved = load_checkpoint(path)
        restored = wrap(GCSTRBranchClassifier(view,num_classes=101,pretrained=False),device)
        core(restored).load_state_dict(saved['model_state'])
        restored_opt = torch.optim.Adam(restored.parameters(),lr=args.lr)
        restored_opt.load_state_dict(saved['optimizer_state'])
        model.eval(); restored.eval()
        image = subset[0][0][None].to(device)
        with torch.inference_mode():
            torch.testing.assert_close(core(model)(image),core(restored)(image),rtol=0,atol=0)
        # Exercise an actual resumed optimizer step as well as checkpoint reload.
        data.set_epoch(2)
        resumed, _ = epoch_pass(restored,loader(subset,args.batch_size,args.num_workers,True,args.seed),device,restored_opt)
        evaluated, _ = epoch_pass(restored,loader(subset,args.batch_size,args.num_workers,False,args.seed),device)
        if any(m['windows']!=len(subset) for m in (result,resumed,evaluated)):
            raise RuntimeError('Smoke distributed window coverage mismatch')
        memory = dict(rank=rank(),peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.)
        memories = [memory]
        if world()>1:
            memories = [None]*world()
            dist.all_gather_object(memories,memory)
        results[view] = dict(train=result,resumed=resumed,evaluation_train_subset=evaluated,gpu_memory=memories)
        if rank()==0: print(json.dumps(dict(event='smoke_branch_passed',view=view,results=results[view])),flush=True)
        del model, restored, opt, restored_opt
    if rank()==0:
        write_json(args.output / 'smoke_passed.json',dict(config=config,results=results))
        print('Smoke passed; formal training was not started.',flush=True)
    barrier()


def train_view(args, config, view, device):
    folder = args.output / 'checkpoints' / view
    last = folder / 'last.pt'
    if last.exists() and not args.resume:
        raise FileExistsError(f'Use --resume to continue {last}')
    seed_all(args.seed)
    saved = load_checkpoint(last) if last.exists() else None
    if saved and saved['config'] != config: raise ValueError('Resume configuration mismatch')
    model = wrap(GCSTRBranchClassifier(view,num_classes=101,pretrained=saved is None),device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.)
    best_loss, best_top1 = None, None
    start_epoch = 1
    if saved:
        core(model).load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        best_loss, best_top1 = saved['best_loss'], saved['best_top1']
        start_epoch = saved['epoch']+1
    train_data = WindowDataset(args.output,'train',view,seed=args.seed)
    test_data = WindowDataset(args.output,'test',view)
    logpath = args.output / 'logs' / f'{view}_metrics.jsonl'
    logpath.parent.mkdir(exist_ok=True)
    if rank()==0 and logpath.exists():
        # A crash after log write but before checkpoint must not duplicate epochs.
        old = [json.loads(line) for line in logpath.read_text().splitlines() if line.strip()]
        old = [r for r in old if r['epoch'] < start_epoch]
        logpath.write_text(''.join(json.dumps(r)+'\n' for r in old))
    for epoch in range(start_epoch,args.epochs+1):
        seed_all(args.seed+epoch)
        train_data.set_epoch(epoch)
        begin = time.perf_counter()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        train, _ = epoch_pass(model,loader(train_data,args.batch_size,args.num_workers,True,args.seed+epoch),device,optimizer)
        test, predictions = epoch_pass(model,loader(test_data,args.batch_size,args.num_workers,False,args.seed),device)
        if train['windows'] != len(train_data) or test['windows'] != len(test_data):
            raise RuntimeError('Incomplete epoch coverage')
        candidate = dict(epoch=epoch,video_nll=test['video_nll'],video_top1=test['video_top1'],metrics=test)
        improved_loss = best_loss is None or (candidate['video_nll'],-candidate['video_top1']) < (best_loss['video_nll'],-best_loss['video_top1'])
        improved_top1 = best_top1 is None or (-candidate['video_top1'],candidate['video_nll']) < (-best_top1['video_top1'],best_top1['video_nll'])
        if improved_loss: best_loss = candidate
        if improved_top1: best_top1 = candidate
        metrics = dict(epoch=epoch,view=view,train=train,test=test,independent_test=False,
                       lr=optimizer.param_groups[0]['lr'],optimizer='Adam',
                       train_window_coverage=1.,test_window_coverage=1.,
                       gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.,
                       gpu_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else 0.,
                       compute_seconds=time.perf_counter()-begin,
                       checkpoints=dict(last=str(last),best_loss=str(folder/'best_loss.pt'),best_top1=str(folder/'best_top1.pt')))
        memory = dict(rank=rank(),device=str(device),allocated_mib=metrics['gpu_peak_allocated_mib'],
                      reserved_mib=metrics['gpu_peak_reserved_mib'])
        memories = [memory]
        if world()>1:
            memories = [None]*world()
            dist.all_gather_object(memories,memory)
        metrics['gpu_memory_by_rank'] = memories
        if rank()!=0:
            barrier()
            continue
        state = dict(epoch=epoch,view=view,config=config,model_state=core(model).state_dict(),
                     optimizer_state=optimizer.state_dict(),metrics=metrics,
                     best_loss=best_loss,best_top1=best_top1)
        if improved_loss: atomic_checkpoint(folder/'best_loss.pt',state)
        if improved_top1: atomic_checkpoint(folder/'best_top1.pt',state)
        atomic_checkpoint(last,state)
        metrics['epoch_seconds_including_checkpoints'] = time.perf_counter()-begin
        metrics['checkpoint_hashes'] = {p.name:sha(p) for p in folder.glob('*.pt')}
        with logpath.open('a') as f: f.write(json.dumps(metrics)+'\n')
        reports = args.output/'reports'
        reports.mkdir(exist_ok=True)
        write_csv(reports/f'{view}_test_video_predictions_last.csv',predictions,list(predictions[0]))
        if improved_loss:
            write_csv(reports/f'{view}_test_video_predictions_best_loss.csv',predictions,list(predictions[0]))
        print(json.dumps(metrics),flush=True)
        barrier()
    del model, optimizer
    if device.type == 'cuda': torch.cuda.empty_cache()
    return dict(best_loss=best_loss,best_top1=best_top1,completed_epoch=args.epochs,
                checkpoints={p.name:dict(path=str(p),sha256=sha(p)) for p in folder.glob('*.pt')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--train',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        if int(os.environ['WORLD_SIZE'])!=2: parser.error('This protocol supports exactly two DDP ranks')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'
        dist.init_process_group('nccl')
    if not 1 <= args.epochs <= 50 or args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0:
        parser.error('Invalid training parameters (epochs must be 1..50)')
    if args.batch_size % world() or args.batch_size < 2*world():
        parser.error('Global batch must divide GPU count and be >=2*GPU count')
    audit = checked_audit(args.output)
    if args.train and audit['config'].get('audit_scope')!='full':
        raise RuntimeError('Subset audit is smoke-only; full data preparation is required for training')
    config = protocol_config(args,audit)
    device = torch.device(args.device)
    # Exclusive process lock prevents accidental concurrent writers.
    with ((args.output / '.training.lock').open('a') if rank()==0 else nullcontext()) as lock:
        if rank()==0: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        barrier()
        if args.smoke:
            smoke(args,config,device)
            return
        marker = args.output/'smoke_passed.json'
        if not marker.exists() or json.loads(marker.read_text())['config'] != config:
            raise RuntimeError('Matching explicit Smoke Test required before formal training')
        # Train projections are generated after augmentation; only Test is cached.
        for view in VIEWS:
            for split in ('test',):
                data = WindowDataset(args.output,split,view)
                missing = next((data.cache_path(r) for r in data.rows if not data.cache_path(r).is_file()),None)
                if missing is not None:
                    raise RuntimeError(f'Complete image cache required before training: {missing}')
        config_path = args.output/'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError('Output directory belongs to another configuration')
        if rank()==0: write_json(config_path,config)
        summary = dict(config=config,coverage=audit['coverage'],independent_test=False,branches={})
        for view in VIEWS:
            summary['branches'][view] = train_view(args,config,view,device)
            if rank()==0: write_json(args.output/'reports'/'summary.json',summary)
            barrier()


if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
