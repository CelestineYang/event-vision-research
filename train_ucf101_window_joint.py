"""Three independent gCSTR branches share one event/window input pipeline.

All three models live on each DDP rank. Each batch is loaded/augmented once,
then each branch runs its own forward, CE backward and Adam update. No fusion.
"""
import argparse
import fcntl
import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from gcstr_augmentation import augment_events
from gcstr_models import GCSTRBranchClassifier
from gcstr_representations import GCSTRConfig
from ucf101_window_data import (OUTPUT, VIEWS, WindowDataset, checked_audit,
                                window_views, sha, write_json, write_csv)
from train_ucf101_window_gcstr import (Metrics, atomic_checkpoint, load_checkpoint,
                                     protocol_config, seed_all, core, wrap, world,
                                     rank, barrier, loader)


def training_dataset(args):
    if args.event_store is not None:
        from ucf101_event_store import PreparedThreeViewDataset
        return PreparedThreeViewDataset(args.output,'train',args.event_store,seed=args.seed)
    return ThreeViewDataset(args.output,'train',seed=args.seed)


def check_resume_config(previous,current,allow_transition=False):
    if previous==current: return
    if not allow_transition:
        raise ValueError('Resume protocol mismatch; explicit data-backend transition required')
    if 'data_backend' in previous or current.get('data_backend',{}).get('train')!='full_event_memory_map':
        raise ValueError('Only the one-time legacy NPZ -> full-event mmap transition is permitted')
    semantic=lambda c:{k:v for k,v in c.items() if k not in ('code','data_backend')}
    if semantic(previous)!=semantic(current):
        raise ValueError('Training/data protocol changed beyond the storage backend')
    before,after=previous['code'],current['code']
    if set(after)-set(before)!={'ucf101_event_store.py'} or set(before)-set(after):
        raise ValueError('Unexpected code dependency change')
    for name in before:
        if name!='train_ucf101_window_joint.py' and before[name]!=after[name]:
            raise ValueError(f'Non-backend code changed: {name}')


def smoke_marker(args):
    return args.output/('joint_smoke_mmap_passed.json' if args.event_store is not None else 'joint_smoke_passed.json')


class ThreeViewDataset(WindowDataset):
    """One full-event read and one spatial transform produce all three views."""
    def __init__(self, output, split, *, seed=0):
        super().__init__(output,split,VIEWS[0],seed=seed)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.augmentation == 'none':
            images = {}
            for view in VIEWS:
                path = self.output/'cache'/view/(row['cache_key']+'.npy')
                if not path.is_file(): raise RuntimeError(f'Missing Test image cache: {path}')
                images[view] = np.load(path,allow_pickle=False)
        else:
            if self._video != row['source_path']:
                if sha(row['source_path']) != row['source_sha256']:
                    raise RuntimeError('Source changed after audit')
                with np.load(row['source_path'],allow_pickle=False) as f:
                    self._arrays = {key:f[key] for key in ('t','x','y','p')}
                self._video = row['source_path']
            start,end = int(row['start_us']),int(row['end_us'])
            left,right = np.searchsorted(self._arrays['t'],[start,end],side='left')
            if right-left != int(row['event_count']) or right-left<=10000:
                raise ValueError('Invalid raw window event count')
            config = GCSTRConfig(sensor_width=int(row['sensor_width']),sensor_height=int(row['sensor_height']))
            events = tuple(self._arrays[key][left:right] for key in ('t','x','y','p'))
            events = augment_events(*events,mode='spatial',rng=self.augmentation_rng(row),config=config)
            images = window_views(*events,start,end,config,VIEWS)
        tensors = {}
        for view,image in images.items():
            if image.shape != (3,224,224) or not np.isfinite(image).all():
                raise ValueError(f'Invalid {view} image')
            image = (image-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
            tensors[view] = torch.from_numpy(image)
        return tensors,int(row['label_id']),row['video_id']


def build_bank(args,device,saved=None):
    models,optimizers = {},{}
    for view in VIEWS:
        seed_all(args.seed)  # Match independent-branch initialization protocol.
        models[view] = wrap(GCSTRBranchClassifier(view,num_classes=101,pretrained=saved is None),device)
        optimizers[view] = torch.optim.Adam(models[view].parameters(),lr=args.lr,weight_decay=0.)
        if saved:
            core(models[view]).load_state_dict(saved['branches'][view]['model_state'])
            optimizers[view].load_state_dict(saved['branches'][view]['optimizer_state'])
    parameter_ids = [set(map(id,m.parameters())) for m in models.values()]
    if any(a&b for i,a in enumerate(parameter_ids) for b in parameter_ids[i+1:]):
        raise RuntimeError('Branch parameters unexpectedly shared')
    return models,optimizers


def bank_state(models,optimizers,epoch,config,history):
    return dict(epoch=epoch,config=config,history=history,
                branches={v:dict(view=v,epoch=epoch,config=config,
                                model_state=core(models[v]).state_dict(),
                                optimizer_state=optimizers[v].state_dict()) for v in VIEWS})


def memory_stats(device):
    item = dict(rank=rank(),device=str(device),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else 0.)
    items = [item]
    if world()>1:
        items = [None]*world()
        dist.all_gather_object(items,item)
    return items


def joint_epoch(models,batches,device,optimizers=None):
    training = optimizers is not None
    for model in models.values(): model.train(training)
    metrics = {v:Metrics() for v in VIEWS}
    begin = time.perf_counter()
    steps = 0
    with torch.enable_grad() if training else torch.inference_mode():
        for images,labels,ids in batches:
            labels = labels.to(device,non_blocking=True)
            if training:
                count = torch.tensor(float(len(labels)),device=device)
                if world()>1: dist.all_reduce(count)
                gradient_scale = world()*len(labels)/count
            for view in VIEWS:
                image = images[view].to(device,non_blocking=True)
                if training: optimizers[view].zero_grad(set_to_none=True)
                model = models[view] if training else core(models[view])
                logits = model(image)
                loss = torch.nn.functional.cross_entropy(logits,labels)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite {view} loss')
                if training:
                    (loss*gradient_scale).backward()
                    finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                    if not finite or not torch.stack(finite).all():
                        raise RuntimeError(f'Nonfinite {view} gradient')
                    optimizers[view].step()
                metrics[view].update(logits,labels,ids)
                del logits,loss,image
            steps += 1
            if rank()==0 and (steps==1 or steps%100==0):
                print(json.dumps(dict(event='joint_progress',training=training,step=steps,
                                      local_steps=len(batches),seconds=time.perf_counter()-begin)),flush=True)
    results,predictions = {},{}
    for view in VIEWS:
        metrics[view].synchronize()
        results[view],predictions[view] = metrics[view].finish()
        results[view]['prediction_mode'] = 'online_train_changing_weights' if training else 'fixed_model_eval'
    return dict(branches=results,seconds=time.perf_counter()-begin),predictions


def check_coverage(result,size):
    if any(item['windows']!=size for item in result['branches'].values()):
        raise RuntimeError('Incomplete or duplicated joint window coverage')


def smoke(args,config,device):
    data = training_dataset(args)
    saved=load_checkpoint(args.output/'checkpoints/joint_last.pt') if args.resume else None
    if saved: check_resume_config(saved['config'],config,args.allow_data_backend_transition)
    initial_epoch=saved['epoch']+1 if saved else 1
    data.set_epoch(initial_epoch)
    subset = Subset(data,list(range(min(2*args.batch_size+1,len(data)))))
    models,optimizers = build_bank(args,device,saved)
    del saved
    before = {v:core(m).backbone.fc.weight.detach().clone() for v,m in models.items()}
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    initial,_ = joint_epoch(models,loader(subset,args.batch_size,args.num_workers,True,args.seed+1),device,optimizers)
    check_coverage(initial,len(subset))
    for v in VIEWS:
        if torch.equal(before[v],core(models[v]).backbone.fc.weight): raise RuntimeError(f'{v} did not update')
    images,_,_ = subset[0]
    for m in models.values(): m.eval()
    with torch.inference_mode():
        expected = {v:core(models[v])(images[v][None].to(device)).cpu() for v in VIEWS}
    checkpoint = args.output/('smoke_joint_mmap' if args.event_store is not None else 'smoke_joint')/'last.pt'
    if rank()==0:
        saved_state = bank_state(models,optimizers,initial_epoch,config,{})
        atomic_checkpoint(checkpoint,saved_state)
        for v in VIEWS: atomic_checkpoint(checkpoint.parent/f'{v}.pt',saved_state['branches'][v])
        del saved_state
    barrier()
    del models,optimizers,before
    saved = load_checkpoint(checkpoint)
    models,optimizers = build_bank(args,device,saved)
    for v in VIEWS:
        if not optimizers[v].state: raise RuntimeError(f'{v}: Adam state not restored')
        models[v].eval()
        with torch.inference_mode():
            torch.testing.assert_close(expected[v],core(models[v])(images[v][None].to(device)).cpu(),rtol=0,atol=0)
    del saved
    data.set_epoch(initial_epoch+1)
    resumed,_ = joint_epoch(models,loader(subset,args.batch_size,args.num_workers,True,args.seed+2),device,optimizers)
    hashes={}
    for view in VIEWS:
        digest=hashlib.sha256()
        for key,value in core(models[view]).state_dict().items():
            digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
        hashes[view]=digest.hexdigest()
    rank_hashes=[hashes]
    if world()>1:
        rank_hashes=[None]*world();dist.all_gather_object(rank_hashes,hashes)
    if any(h!=hashes for h in rank_hashes):
        raise RuntimeError(f'Rank model states differ: {rank_hashes}')
    # Evaluate the same deterministic epoch-2 augmented Train subset in eval mode.
    # Official Test payload is not accessed by this execution check.
    evaluated,predictions = joint_epoch(models,loader(subset,args.batch_size,args.num_workers,False,args.seed),device)
    check_coverage(resumed,len(subset)); check_coverage(evaluated,len(subset))
    # Uneven eval shards must agree with direct, single-rank video aggregation.
    if rank()==0:
        from torch.utils.data import DataLoader
        reference = {v:Metrics() for v in VIEWS}
        with torch.inference_mode():
            # Match each actual eval shard's batch shapes. Different convolution
            # batch shapes may choose different FP32/TF32 kernels, which is not
            # a distributed voting error and is visible with trained logits.
            for replica in range(world()):
                shard=Subset(subset,list(range(replica,len(subset),world())))
                for batch,labels,ids in DataLoader(shard,batch_size=max(1,args.batch_size//world()),num_workers=0):
                    for v in VIEWS:
                        reference[v].update(core(models[v])(batch[v].to(device)),labels,ids)
        for v in VIEWS:
            expected_metrics,_ = reference[v].finish()
            for key,value in expected_metrics.items():
                if not np.isclose(value,evaluated['branches'][v][key],rtol=1e-5,atol=1e-5):
                    raise RuntimeError(f'Distributed vote mismatch: {v}/{key}, reference={value}, distributed={evaluated["branches"][v][key]}')
    barrier()
    result = dict(config=config,initial=initial,resumed=resumed,
                  evaluation_train_subset=evaluated,gpu_memory=memory_stats(device),rank_model_hashes=rank_hashes,
                  checks=['three_independent_parameter_sets','finite_gradients','all_classifiers_updated',
                          'checkpoint_logits_identical','three_adam_states_restored','full_window_coverage',
                          'distributed_voting_matches_single_rank'])
    if rank()==0:
        write_json(smoke_marker(args),dict(result,resume_source_epoch=initial_epoch-1))
        print(json.dumps(dict(event='joint_smoke_passed',results=result)),flush=True)
    barrier()


def train(args,config,device,audit):
    last = args.output/'checkpoints'/'joint_last.pt'
    if last.exists() and not args.resume: raise FileExistsError('Use --resume for existing joint checkpoint')
    saved = load_checkpoint(last) if last.exists() else None
    if saved: check_resume_config(saved['config'],config,args.allow_data_backend_transition)
    models,optimizers = build_bank(args,device,saved)
    history = saved['history'] if saved else {v:dict(best_loss=None,best_top1=None) for v in VIEWS}
    start = saved['epoch']+1 if saved else 1
    del saved
    train_data = training_dataset(args)
    test_data = ThreeViewDataset(args.output,'test',seed=args.seed)
    logs,reports = args.output/'logs',args.output/'reports'
    if rank()==0:
        logs.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
        for v in VIEWS:
            path = logs/f'{v}_metrics.jsonl'
            if path.exists():
                rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<start))
    barrier()
    for epoch in range(start,args.epochs+1):
        seed_all(args.seed+epoch); train_data.set_epoch(epoch)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        trained,_ = joint_epoch(models,loader(train_data,args.batch_size,args.num_workers,True,args.seed+epoch),device,optimizers)
        tested,predictions = joint_epoch(models,loader(test_data,args.batch_size,args.num_workers,False,args.seed),device)
        check_coverage(trained,len(train_data)); check_coverage(tested,len(test_data))
        memory = memory_stats(device)
        if rank()==0:
            changed = {}
            for v in VIEWS:
                candidate = dict(epoch=epoch,metrics=tested['branches'][v])
                changed[v] = []
                for name in ('best_loss','best_top1'):
                    def key(item):
                        m = item['metrics']
                        return (m['video_nll'],-m['video_top1']) if name=='best_loss' else (-m['video_top1'],m['video_nll'])
                    previous = history[v][name]
                    if previous is None or key(candidate)<key(previous):
                        history[v][name] = candidate
                        changed[v].append(name)
            state = bank_state(models,optimizers,epoch,config,history)
            metrics = {}
            for v in VIEWS:
                folder = args.output/'checkpoints'/v
                metric = dict(epoch=epoch,view=v,train=trained['branches'][v],test=tested['branches'][v],
                              independent_test=False,lr=args.lr,optimizer='Adam',
                              train_window_coverage=1.,test_window_coverage=1.,gpu_memory_by_rank=memory,
                              shared_train_seconds=trained['seconds'],shared_test_seconds=tested['seconds'],
                              checkpoints={k:str(folder/(k+'.pt')) for k in ('last','best_loss','best_top1')})
                branch = dict(state['branches'][v],metrics=metric,**history[v])
                atomic_checkpoint(folder/'last.pt',branch)
                for name in changed[v]: atomic_checkpoint(folder/(name+'.pt'),branch)
                metric['checkpoint_hashes'] = {p.name:sha(p) for p in folder.glob('*.pt')}
                metrics[v] = metric
                write_csv(reports/f'{v}_test_video_predictions_last.csv',predictions[v],list(predictions[v][0]))
                if 'best_loss' in changed[v]:
                    write_csv(reports/f'{v}_test_video_predictions_best_loss.csv',predictions[v],list(predictions[v][0]))
            # One atomic checkpoint is authoritative for resuming all branches at the same epoch.
            atomic_checkpoint(last,state)
            write_json(args.output/'active_config.json',config)
            for v,metric in metrics.items():
                metric['joint_epoch_seconds_including_checkpoints'] = time.perf_counter()-begin
                with (logs/f'{v}_metrics.jsonl').open('a') as f: f.write(json.dumps(metric)+'\n')
            write_json(reports/'summary.json',dict(config=config,completed_epoch=epoch,branches=history,
                       coverage=audit['coverage'],joint_checkpoint=str(last),joint_checkpoint_sha256=sha(last),
                       checkpoints={v:metrics[v]['checkpoints'] for v in VIEWS},
                       checkpoint_hashes={v:metrics[v]['checkpoint_hashes'] for v in VIEWS}))
            print(json.dumps(dict(event='joint_epoch_complete',epoch=epoch,metrics=metrics)),flush=True)
        barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true'); mode.add_argument('--train',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--event-store',type=Path,help='Verified full-event mmap cache for Train; Test keeps image cache')
    parser.add_argument('--allow-data-backend-transition',action='store_true',
                        help='Allow only legacy NPZ -> verified mmap resume with otherwise identical protocol')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        if int(os.environ['WORLD_SIZE'])!=2: parser.error('Expected two DDP ranks')
        local_rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'; dist.init_process_group('nccl')
    if not 1<=args.epochs<=50 or args.batch_size<2*world() or args.batch_size%world() or args.num_workers<0 or args.lr<=0:
        parser.error('Invalid epochs/global batch/workers/lr')
    audit = checked_audit(args.output)
    if args.train and audit['config'].get('audit_scope')!='full':
        raise RuntimeError('Subset audit is smoke-only, not sufficient for formal training')
    config = protocol_config(args,audit)
    config['execution'] = 'three_independent_branches_shared_event_batch'
    config['code']['train_ucf101_window_joint.py'] = sha(Path(__file__))
    if args.allow_data_backend_transition and (not args.resume or args.event_store is None):
        parser.error('Data backend transition requires --resume and --event-store')
    if args.event_store is not None:
        from ucf101_event_store import EventStore
        store=EventStore(args.event_store,args.output)
        config['data_backend']=dict(train='full_event_memory_map',event_store=str(args.event_store.resolve()),
                                    store_summary_sha256=sha(args.event_store/'summary.json'),
                                    test='unchanged_pre_rendered_images')
        config['code']['ucf101_event_store.py']=sha(Path(__file__).parent/'ucf101_event_store.py')
        del store
    device = torch.device(args.device)
    with ((args.output/'.training.lock').open('a') if rank()==0 else nullcontext()) as lock:
        if rank()==0: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        barrier()
        if args.smoke:
            smoke(args,config,device)
            return
        marker = smoke_marker(args)
        if not marker.is_file() or json.loads(marker.read_text())['config']!=config:
            raise RuntimeError('Matching joint Smoke Test required')
        for v in VIEWS:
            data = WindowDataset(args.output,'test',v)
            for row in data.rows:
                if not data.cache_path(row).is_file(): raise RuntimeError('Complete Test cache required')
        path = args.output/('active_config.json' if (args.output/'active_config.json').exists() else 'config.json')
        if path.exists(): check_resume_config(json.loads(path.read_text()),config,args.allow_data_backend_transition)
        elif rank()==0: write_json(args.output/'config.json',config)
        # Preserve config.json as epoch1 provenance; active config published only
        # after a resumed epoch has successfully saved its new checkpoint.
        train(args,config,device,audit)


if __name__=='__main__':
    try: main()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
