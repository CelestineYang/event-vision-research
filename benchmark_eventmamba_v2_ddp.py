"""Train-only throughput probe from a trusted checkpoint; never saves weights."""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from eventmamba_v2_model import EventMamba
from eventmamba_v2_data import WindowDataset
from eventmamba_v2_metrics import smoothed_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--steps', type=int, default=120)
    args = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torch.manual_seed(0)
    model = EventMamba(200).cuda()
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.load_state_dict(state['model'], strict=True)
    if world > 1:
        model = DDP(model, device_ids=[local])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.load_state_dict(state['optimizer'])
    data = WindowDataset(args.cache)
    if not data.signature['training']:
        raise ValueError('Only Train cache may be used')
    data = Subset(data, range((args.steps + 10) * 32))
    sampler = DistributedSampler(data, world, rank, shuffle=True, drop_last=True) if world > 1 else None
    loader = DataLoader(data, batch_size=32 // world, shuffle=sampler is None,
                        sampler=sampler, num_workers=2, drop_last=True,
                        pin_memory=True, multiprocessing_context='spawn')
    model.train()
    start = None
    for step, batch in enumerate(loader):
        if step == 10:
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier()
            start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['events'].cuda(non_blocking=True))
        loss = smoothed_loss(logits, batch['label'].cuda(non_blocking=True))
        loss.backward()
        if not torch.stack([torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None]).all():
            raise FloatingPointError('Nonfinite benchmark gradient')
        optimizer.step()
    torch.cuda.synchronize()
    seconds = torch.tensor(time.monotonic() - start, device='cuda')
    if world > 1:
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(json.dumps({'world_size': world, 'checkpoint_epoch': state['epoch'],
                          'global_batch': 32, 'measured_steps': args.steps,
                          'seconds': seconds.item(), 'seconds_per_step': seconds.item() / args.steps,
                          'windows_per_second': args.steps * 32 / seconds.item(),
                          'peak_cuda_mib': torch.cuda.max_memory_allocated() / 2**20}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
