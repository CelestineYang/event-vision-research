"""Train-only batch benchmark. No experiment checkpoints or Test observations."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from gcstr_models import GCSTRBranchClassifier
from train_dailydvs200_window_gcstr import DailyViews, epoch_pass
import train_dailydvs200_window_gcstr as training
from train_gcstr_feature_memory import seed_all
from ucf101_window_data import VIEWS, write_json


class TimedLoader(training.DataLoader):
    wait_seconds=0.

    def __iter__(self):
        started=time.monotonic()
        iterator=super().__iter__()
        type(self).wait_seconds+=time.monotonic()-started
        while True:
            started=time.monotonic()
            try:
                batch=next(iterator)
            except StopIteration:
                return
            type(self).wait_seconds+=time.monotonic()-started
            yield batch


def gpu_snapshot():
    return subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                           '--format=csv'],capture_output=True,text=True,timeout=10).stdout.strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=768)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args()
    training.DataLoader=TimedLoader
    if args.output.exists(): raise FileExistsError(args.output)
    if args.samples % args.batch_size: raise ValueError('Use exact common coverage divisible by each batch')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.35,device)
    data=DailyViews(args.data/'experiment','train',args.data/'event_store',seed=0)
    indices=np.random.default_rng(0).choice(len(data),args.samples,replace=False)
    data.rows=[data.rows[int(i)] for i in indices];data.set_epoch(1)
    identity=hashlib.sha256(json.dumps([(r['video_id'],r['window_id']) for r in data.rows]).encode()).hexdigest()
    models={}
    for v in VIEWS:
        seed_all(0);models[v]=GCSTRBranchClassifier(v,num_classes=200,pretrained=True).to(device)
    initial={v:{k:t.detach().cpu().clone() for k,t in m.state_dict().items()} for v,m in models.items()}
    def reset():
        for v in VIEWS: models[v].load_state_dict(initial[v],strict=True)
        seed_all(0)
        return {v:torch.optim.Adam(models[v].parameters(),lr=1e-4,weight_decay=0.) for v in VIEWS}
    warm=copy.copy(data);warm.rows=data.rows[:args.batch_size]
    warm_optimizers=reset()
    epoch_pass(models,warm,args.batch_size,args.workers,device,0,warm_optimizers)
    del warm_optimizers,warm
    for model in models.values(): model.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
    runs=[]
    for repeat in range(2):
        optimizers=reset()
        torch.cuda.reset_peak_memory_stats(device)
        before=gpu_snapshot();torch.cuda.synchronize();start=time.monotonic()
        TimedLoader.wait_seconds=0.
        result,_=epoch_pass(models,data,args.batch_size,args.workers,device,0,optimizers)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        item=dict(repeat=repeat+1,seconds=elapsed,windows_per_second=args.samples/elapsed,
            data_wait_seconds=TimedLoader.wait_seconds,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
            gpu_before=before,gpu_after=gpu_snapshot())
        runs.append(item);print(json.dumps(dict(event='benchmark',batch_size=args.batch_size,**item)),flush=True)
        del optimizers,result
        for model in models.values(): model.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,dict(batch_size=args.batch_size,samples=args.samples,workers=args.workers,
        train_window_identity=identity,augmentation_epoch=1,precision='FP32',lr=1e-4,
        branches=list(VIEWS),runs=runs,note='End-to-end includes data workers, projection and metrics; no accuracy inference or Test read'))


if __name__=='__main__':main()
