"""Single/DDP image-only query control. Global batch size; validation only, no test."""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from create_gcstr_splits import DEFAULT_ROOT
from gcstr_image_query import ImageQueryClassifier, ImageQueryDataset
from gcstr_secnet_fusion import VIEWS


class ExactTrainBatches:
    """Shard each global batch without padding or dropping samples.

For the official train split and world=2, final global batch=10 (5/rank).
Also handles unequal nonempty local counts via loss weighting in epoch_pass.
"""
    def __init__(self, size, batch_size, rank, world, seed):
        remainder = size % batch_size
        if batch_size < world or size < world or (remainder and remainder < world):
            raise ValueError('Every global batch must have at least world_size samples')
        self.size, self.batch_size, self.rank, self.world, self.seed = size, batch_size, rank, world, seed

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed)).tolist()
        for start in range(0, self.size, self.batch_size):
            yield order[start:start + self.batch_size][self.rank::self.world]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def emit(rank, **record):
    if rank == 0:
        print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def reduce_sum(tensor, world):
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def epoch_pass(model, raw, loader, device, epoch, optimizer, limit, log_every, rank, world):
    training = optimizer is not None
    model.train(training)
    phase = 'train' if training else 'val'
    total = len(loader) if limit is None else min(len(loader), limit)
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    started = time.monotonic()
    emit(rank, event='phase_start', epoch=epoch, phase=phase, local_batches=total)
    # Validation bypasses DDP forward: ranks have unequal sample/batch counts.
    # All BN is frozen; no forward collectives are required. Reduce only at end.
    with torch.enable_grad() if training else torch.no_grad():
        for step, batch in enumerate(itertools.islice(loader, total), 1):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            y = batch['label']
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch) if training else raw(batch)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
            finite = torch.isfinite(loss).to(torch.float32)
            if training:
                count = reduce_sum(torch.tensor(float(len(y)), device=device), world)
                if reduce_sum(finite, world).item() != world:
                    raise RuntimeError('Non-finite training loss')
                # DDP averages gradients across ranks: recover global sample mean.
                (loss * world / count).backward()
                torch.nn.utils.clip_grad_norm_((p for p in raw.parameters() if p.requires_grad),
                                              1.0, error_if_nonfinite=True)
                optimizer.step()
            elif not finite.item():
                raise RuntimeError('Non-finite validation loss')
            sums[0] += loss.detach()
            sums[1] += (logits.argmax(1) == y).sum()
            sums[2] += (logits.topk(5, dim=1).indices == y[:, None]).any(1).sum()
            sums[3] += len(y)
            if step == 1 or step % log_every == 0 or step == total:
                # Training ranks have equal step counts; these are global metrics.
                progress = reduce_sum(sums.clone(), world) if training else sums
                elapsed = time.monotonic() - started
                emit(rank, event='progress', epoch=epoch, phase=phase, batch=step,
                     local_batches=total, scope='global' if training else 'rank0_only',
                     loss=(progress[0] / progress[3]).item(), top1=(progress[1] / progress[3]).item(),
                     seconds=round(elapsed, 1), eta_seconds=round(elapsed / step * (total-step), 1))
    sums = reduce_sum(sums, world)
    if sums[3].item() == 0:
        raise RuntimeError('No samples processed')
    return dict(loss=(sums[0]/sums[3]).item(), top1=(sums[1]/sums[3]).item(),
                top5=(sums[2]/sums[3]).item(), samples=int(sums[3].item()),
                seconds=round(time.monotonic()-started, 1))


def atomic_save(path, data):
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('gcstr_splits_seed0.csv'))
    p.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16, help='GLOBAL batch size, not per GPU')
    p.add_argument('--num-workers', type=int, default=1, help='workers PER rank')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10, help='early stop on validation LOSS, as previous run')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--augmentation', choices=('none', 'spatial'), default='spatial')
    p.add_argument('--augmentation-count', type=int, default=5)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--max-train-batches', type=int)
    p.add_argument('--max-val-batches', type=int, help='limit per rank; diagnostics only')
    args = p.parse_args()
    for key in ('epochs', 'batch_size', 'patience', 'augmentation_count', 'log_every'):
        if getattr(args, key) < 1:
            p.error(f'{key} must be positive')
    if args.num_workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        p.error('Invalid workers/optimizer settings')
    if any(x is not None and x < 1 for x in (args.max_train_batches, args.max_val_batches)):
        p.error('Batch limits must be positive')
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if args.batch_size % world:
        p.error('Global batch size must be divisible by world_size')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=10))
        probe = reduce_sum(torch.tensor(float(rank + 1), device=device), world)
        assert probe.item() == world * (world + 1) / 2
    emit(rank, event='communication_ok', world_size=world, global_batch=args.batch_size,
         per_gpu_batch=args.batch_size // world)
    try:
        if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise RuntimeError('Output directory not empty; choose a new directory')
        if world > 1:
            dist.barrier()
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        seed_all(args.seed)
        signature = {k: v for k, v in vars(args).items() if k not in
                     ('epochs', 'output_dir', 'resume', 'cache_dir', 'num_workers', 'log_every', 'manifest', 'dataset_root')}
        signature.update(architecture='image_query_ddp_v1', world_size=world,
                         manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                         dataset_root=str(args.dataset_root.resolve()))
        raw = ImageQueryClassifier()
        checkpoint = None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if checkpoint['signature'] != signature:
                raise ValueError('Resume signature differs (including GPU count or batch size)')
            raw.load_state_dict(checkpoint['model_state'], strict=True)
        else:
            raw.load_image_checkpoints({v: f'gcstr_checkpoints/full_{v}_spatial_seed0/{v}_seed0_best.pt' for v in VIEWS})
        raw.to(device)
        model = DDP(raw, device_ids=[local], broadcast_buffers=False) if world > 1 else raw
        optimizer = torch.optim.AdamW((x for x in raw.parameters() if x.requires_grad),
                                     lr=args.learning_rate, weight_decay=args.weight_decay)
        start, best_loss, best_top1, bad = 1, float('inf'), -1., 0
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start = checkpoint['epoch'] + 1
            best_loss, best_top1, bad = (checkpoint[k] for k in ('best_loss', 'best_top1', 'bad_epochs'))
            del checkpoint
        common = dict(manifest_path=args.manifest, dataset_root=args.dataset_root, cache_dir=args.cache_dir, seed=args.seed)
        train = ImageQueryDataset(split='train', augmentation=args.augmentation,
                                  augmentation_count=args.augmentation_count, augmentation_seed=args.seed, **common)
        val = ImageQueryDataset(split='val', **common)
        emit(rank, event='start', args=vars(args), start_epoch=start, train_samples=len(train), val_samples=len(val),
             parameters=sum(x.numel() for x in raw.parameters()),
             trainable_parameters=sum(x.numel() for x in raw.parameters() if x.requires_grad))
        if bad >= args.patience:
            emit(rank, event='already_early_stopped')
            return
        for epoch in range(start, args.epochs + 1):
            seed_all(args.seed + epoch)
            train.set_epoch(epoch)
            worker_args = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
            train_loader = DataLoader(train, batch_sampler=ExactTrainBatches(len(train), args.batch_size, rank, world, args.seed+epoch),
                                      generator=torch.Generator().manual_seed(args.seed+epoch+rank), **worker_args)
            tm = epoch_pass(model, raw, train_loader, device, epoch, optimizer,
                            args.max_train_batches, args.log_every, rank, world)
            seed_all(args.seed + 1000000)
            # Disjoint, exhaustive validation partition: 2955 stays 2955, not 2956.
            val_loader = DataLoader(Subset(val, list(range(rank, len(val), world))),
                                    batch_size=args.batch_size//world, shuffle=False,
                                    generator=torch.Generator().manual_seed(args.seed+rank), **worker_args)
            vm = epoch_pass(model, raw, val_loader, device, epoch, None,
                            args.max_val_batches, args.log_every, rank, world)
            if args.max_train_batches is None:
                assert tm['samples'] == len(train)
            if args.max_val_batches is None:
                assert vm['samples'] == len(val)
            better_loss, better_top1 = vm['loss'] < best_loss, vm['top1'] > best_top1
            best_loss, best_top1 = min(best_loss, vm['loss']), max(best_top1, vm['top1'])
            bad = 0 if better_loss else bad + 1
            if rank == 0:
                state = dict(epoch=epoch, model_state=raw.state_dict(), optimizer_state=optimizer.state_dict(),
                             best_loss=best_loss, best_top1=best_top1, bad_epochs=bad, signature=signature,
                             config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, train=tm, val=vm)
                if better_loss:
                    atomic_save(args.output_dir/'best_loss.pt', state)
                if better_top1:
                    atomic_save(args.output_dir/'best_top1.pt', state)
                atomic_save(args.output_dir/'last.pt', state)
                record = dict(epoch=epoch, train=tm, val=vm, best_val_loss=best_loss, best_val_top1=best_top1, bad_epochs=bad)
                emit(rank, **record)
                with (args.output_dir/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record)+'\n')
            if world > 1:
                dist.barrier()
            if bad >= args.patience:
                emit(rank, event='early_stop', epoch=epoch)
                break
        emit(rank, event='finished')
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
