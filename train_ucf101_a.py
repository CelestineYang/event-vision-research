"""UCF101-DVS: official Train/Test, fixed epochs, optional two-GPU DDP.

Image pretraining and A use final-epoch weights. Test never selects checkpoints.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from gcstr_models import GCSTRBranchClassifier, PaSNetProbabilityEnsemble
from gcstr_secnet_fusion import VIEWS
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from train_secnet_only import seed_all, atomic_save
from ucf101_a_data import UCF_ROOT, UCF_CONFIG, UCF101ADataset, build_manifest


def rank0_call(function, world_size):
    if world_size == 1:
        return function()
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = {'ok': True, 'value': function()}
        except Exception as exc:
            payload[0] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(payload, src=0)
    if not payload[0]['ok']:
        raise RuntimeError(payload[0]['error'])
    return payload[0]['value']


def common_signature(args):
    here = Path(__file__).resolve().parent
    names = ['ucf101_a_data.py', 'train_ucf101_a.py', 'gcstr_models.py',
             'gcstr_eventmamba_fusion.py', 'gcstr_secnet_fusion.py', 'eventmamba_encoder.py',
             'gcstr_fusion_cached_dataset.py', 'gcstr_fusion_dataset.py', 'gcstr_augmentation.py',
             'gcstr_representations.py', 'train_secnet_only.py']
    return {'protocol': 'ucf101_official_train_test_fixed_epochs_v3',
            'validation': False, 'early_stopping': False, 'checkpoint_selection': 'final_epoch',
            'dataset': 'UCF101-DVS', 'num_classes': 101, 'config': asdict(UCF_CONFIG),
            'manifest_hash': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'root': str(args.dataset_root.resolve()), 'seed': args.seed, 'smoke': args.smoke,
            'source_hashes': {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in names}}


def dataset(args, split, *, augment=None):
    if args.smoke and split != 'train':
        raise ValueError('Smoke may read only Train recordings')
    augment = split == 'train' if augment is None else augment
    data = UCF101ADataset(args.manifest, split, dataset_root=args.dataset_root,
                         point_count=8192, seed=args.seed,
                         augmentation='spatial' if augment else 'none',
                         augmentation_count=5, augmentation_seed=args.seed)
    if args.smoke:
        data.rows = data.rows[:max(32, 2 * args.image_batch, 2 * args.fusion_batch)]
    return data


def loader(data, args, epoch, batch_size, training):
    data.set_epoch(epoch)
    world = args.world_size if training else 1
    rank = args.rank if training else 0
    if batch_size % world or batch_size // world < 2:
        raise ValueError('Global batch must divide world size, with >=2 examples per rank')
    # Actual official Train (9464) divides evenly across two ranks: no duplication,
    # no discarded samples. Refuse padding if a different rank count is requested.
    if training and len(data) % world:
        raise ValueError('Train size must divide ranks exactly; padding is not permitted')
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=False) if training and world > 1 else None
    if sampler is not None:
        sampler.set_epoch(epoch)
    seed = args.seed + epoch + rank * 10000 if training else args.seed + 1000000
    seed_all(seed)
    return DataLoader(data, batch_size=batch_size // world,
                      shuffle=training and sampler is None, sampler=sampler, drop_last=False,
                      num_workers=args.workers, pin_memory=True, persistent_workers=False,
                      multiprocessing_context='spawn' if args.workers else None,
                      generator=torch.Generator().manual_seed(seed))


class FusionOutputs(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, batch):
        return self.core.forward_batch_outputs(batch)


def core_model(model):
    model = model.module if isinstance(model, DDP) else model
    return model.core if isinstance(model, FusionOutputs) else model


def wrap_model(model, args, device, fusion):
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if fusion:
            model = FusionOutputs(model)
        model = DDP(model, device_ids=[device.index])
    return model


def reduced_sums(sums, device, world_size):
    values = torch.tensor(sums, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values)
    return values.cpu().tolist()


def run_epoch(models, optimizers, data_loader, device, *, training, fusion,
              epoch, emit, log_every, world_size=1, limit=None):
    # Per branch: fusion CE, top1/5, samples, point CE, point top1/5, objective.
    sums = {v: [0.] * 8 for v in models}
    steps = min(len(data_loader), limit) if limit else len(data_loader)
    for model in models.values():
        model.train(training)
    started = time.monotonic()

    def metrics():
        result = {}
        for v, values in sums.items():
            x = reduced_sums(values, device, world_size)
            if not x[3]:
                raise ValueError('No samples processed')
            item = {'loss': x[0] / x[3], 'top1': x[1] / x[3], 'top5': x[2] / x[3],
                    'samples': int(x[3])}
            if fusion:
                item.update(point_loss=x[4] / x[3], point_top1=x[5] / x[3],
                            point_top5=x[6] / x[3], objective_loss=x[7] / x[3])
            result[v] = item
        return result

    with torch.set_grad_enabled(training):
        for step, batch in enumerate(data_loader, 1):
            if step > steps:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            labels = batch['label']
            for view, model in models.items():
                if training:
                    optimizers[view].zero_grad(set_to_none=True)
                if fusion:
                    output = model(batch) if isinstance(model, DDP) else model.forward_batch_outputs(batch)
                    logits, point_logits = output['logits'], output['point_logits']
                    point_loss = torch.nn.functional.cross_entropy(point_logits, labels)
                else:
                    logits = model(batch[view]) if isinstance(model, DDP) else model.forward_batch(batch)
                    point_logits, point_loss = None, logits.new_zeros(())
                loss = torch.nn.functional.cross_entropy(logits, labels)
                objective = loss + 0.5 * point_loss if fusion else loss
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite objective')
                if training:
                    objective.backward()
                    if fusion:
                        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                                       max_norm=1., error_if_nonfinite=True)
                    elif not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                                          if p.grad is not None]).all():
                        raise FloatingPointError('Nonfinite image gradient')
                    optimizers[view].step()
                count = len(labels)
                sums[view][0] += loss.item() * count
                sums[view][1] += (logits.argmax(1) == labels).sum().item()
                sums[view][2] += (logits.topk(5, 1).indices == labels[:, None]).any(1).sum().item()
                sums[view][3] += count
                sums[view][7] += objective.item() * count
                if fusion:
                    sums[view][4] += point_loss.item() * count
                    sums[view][5] += (point_logits.argmax(1) == labels).sum().item()
                    sums[view][6] += (point_logits.topk(5, 1).indices == labels[:, None]).any(1).sum().item()
            if step == 1 or step % log_every == 0 or step == steps:
                emit({'event': 'progress', 'phase': 'train' if training else 'final_eval',
                      'epoch': epoch, 'step': step, 'steps': steps,
                      'metrics': metrics(), 'seconds': time.monotonic() - started})
    result = metrics()
    for item in result.values():
        item['seconds'] = time.monotonic() - started
    return result


def image_paths(args):
    return {v: args.output_root / 'images' / f'{v}_final.pt' for v in VIEWS}


def check_image_checkpoint(path, view, common):
    item = torch.load(path, map_location='cpu', weights_only=False)
    if item.get('view') != view or item.get('signature', {}).get('common') != common:
        raise ValueError(f'Wrong final UCF image checkpoint: {path}')
    if item['epoch'] != item['signature']['epochs'] or item['signature']['stage'] != 'images':
        raise ValueError('Image checkpoint must be from the final configured epoch')
    return item


def check_final_state(bank, common):
    if not bank['complete'] or bank['epoch'] != bank['signature']['epochs'] or bank['signature']['common'] != common:
        raise ValueError('Training must finish all configured epochs before Test')


def train_stage(args, directory, common, emit, device):
    fusion = args.stage == 'fusion'
    epochs = 2 if args.smoke else args.epochs
    batch_size = args.fusion_batch if fusion else args.image_batch
    signature = {'common': common, 'stage': args.stage, 'epochs': epochs,
                 'world_size': args.world_size, 'global_batch': batch_size,
                 'normalization': 'SyncBatchNorm' if args.world_size > 1 else 'BatchNorm',
                 'train_tail': 'keep_all_no_padding', 'early_stopping': False,
                 'optimizer': 'AdamW' if fusion else 'Adam', 'lr': 1e-4,
                 'weight_decay': 1e-4 if fusion else 0.0}
    if fusion:
        paths = image_paths(args)
        for v, path in paths.items():
            check_image_checkpoint(path, v, common)
        signature.update(point_count=8192, aux_weight=0.5, images_frozen=True,
                         encoder='EventMamba_V1_groups1024_512_256_K24',
                         image_hashes={v: hashlib.sha256(p.read_bytes()).hexdigest() for v, p in paths.items()})
    else:
        signature.update(init='ImageNet1K_V1', views=list(VIEWS))
    last = directory / 'last.pt'
    state = torch.load(last, map_location='cpu', weights_only=False) if args.resume and last.exists() else None
    if state and state['signature'] != signature:
        raise ValueError('Resume protocol/source/world-size mismatch; old Val/early-stop runs are incompatible')
    if state and state['epoch'] >= epochs:
        emit({'event': 'already_finished', 'epoch': state['epoch']})
        return
    seed_all(args.seed)
    if fusion:
        models = {'A': GCSTREventMambaFusion(num_classes=101, freeze_images=True, point_aux=True)}
        if state is None:
            models['A'].load_image_checkpoints(paths)
    else:
        models = {v: GCSTRBranchClassifier(v, num_classes=101, pretrained=state is None) for v in VIEWS}
    if state:
        for v in models:
            models[v].load_state_dict(state['models'][v], strict=True)
    models = {v: wrap_model(m.to(device), args, device, fusion) for v, m in models.items()}
    if fusion:
        optimizers = {v: torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),
                                          lr=1e-4, weight_decay=1e-4) for v, m in models.items()}
    else:
        optimizers = {v: torch.optim.Adam(m.parameters(), lr=1e-4) for v, m in models.items()}
    if state:
        for v in models:
            optimizers[v].load_state_dict(state['optimizers'][v])
    start = state['epoch'] + 1 if state else 1
    train_data = dataset(args, 'train')
    emit({'event': 'start', 'pid': os.getpid(), 'signature': signature,
          'start_epoch': start, 'train_samples': len(train_data),
          'global_batch': batch_size, 'per_rank_batch': batch_size // args.world_size})
    for epoch in range(start, epochs + 1):
        train = run_epoch(models, optimizers, loader(train_data, args, epoch, batch_size, True), device,
                          training=True, fusion=fusion, epoch=epoch, emit=emit,
                          log_every=args.log_every, world_size=args.world_size,
                          limit=1 if args.smoke else None)
        complete = epoch == epochs

        def save_epoch():
            bank = {'epoch': epoch, 'signature': signature,
                    'models': {v: core_model(m).state_dict() for v, m in models.items()},
                    'optimizers': {v: o.state_dict() for v, o in optimizers.items()},
                    'complete': complete, 'train': train}
            atomic_save(bank, last)
            if complete:
                for v in models:
                    item = {'epoch': epoch, 'signature': signature,
                            'model_state': core_model(models[v]).state_dict(), 'train': train[v]}
                    if not fusion:
                        item['view'] = v
                    atomic_save(item, directory / ('final.pt' if fusion else f'{v}_final.pt'))
            emit({'event': 'epoch', 'epoch': epoch, 'train': train, 'complete': complete,
                  'peak_cuda_mib': torch.cuda.max_memory_allocated() / 2**20})
        rank0_call(save_epoch, args.world_size)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            emit({'event': 'smoke_pause', 'epoch': epoch})
            return
    emit({'event': 'finished'})


def final_evaluation(args, directory, common, emit, device):
    # Main refuses multi-rank evaluation. Every Test recording appears exactly once.
    result_path = directory / ('smoke_eval.json' if args.smoke else 'test_final.json')
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved.get('common') != common or saved.get('selection') != 'final_epoch_only':
            raise ValueError('Existing evaluation belongs to another protocol')
        emit({'event': 'evaluation_already_saved', 'path': str(result_path)})
        return
    image_test = args.stage == 'image_test'
    train_dir = args.output_root / ('images' if image_test else 'fusion')
    bank = torch.load(train_dir / 'last.pt', map_location='cpu', weights_only=False)
    check_final_state(bank, common)
    if image_test:
        branches = {}
        epochs = {}
        for v, path in image_paths(args).items():
            item = check_image_checkpoint(path, v, common)
            if item['signature'] != bank['signature']:
                raise ValueError('Image final-state mismatch')
            epochs[v] = item['epoch']
            branches[v] = GCSTRBranchClassifier(v, num_classes=101, pretrained=False).to(device)
            branches[v].load_state_dict(item['model_state'])
        models = {**branches, 'ensemble': PaSNetProbabilityEnsemble(branches).to(device)}
        selected_epoch = epochs
    else:
        item = torch.load(train_dir / 'final.pt', map_location='cpu', weights_only=False)
        if item['signature'] != bank['signature'] or item['epoch'] != bank['epoch']:
            raise ValueError('A final-state mismatch')
        model = GCSTREventMambaFusion(num_classes=101, freeze_images=True, point_aux=True).to(device)
        model.load_state_dict(item['model_state'], strict=True)
        models, selected_epoch = {'A': model}, item['epoch']
    eval_data = dataset(args, 'train' if args.smoke else 'test', augment=False)
    batch_size = args.image_batch if image_test else args.fusion_batch
    metrics = run_epoch(models, None, loader(eval_data, args, 0, batch_size, False), device,
                        training=False, fusion=not image_test, epoch=0, emit=emit, log_every=args.log_every)
    result = {'event': 'final_evaluation', 'split': 'train_smoke' if args.smoke else 'test',
              'samples': len(eval_data), 'selection': 'final_epoch_only',
              'checkpoint_epoch': selected_epoch, 'metrics': metrics, 'common': common}
    temporary = result_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(result_path)
    emit(result)


def create_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('images', 'image_test', 'fusion', 'test'), required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('ucf101_a_official_split1.csv'))
    p.add_argument('--dataset-root', type=Path, default=UCF_ROOT)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--image-batch', type=int, default=64, help='GLOBAL image batch')
    p.add_argument('--fusion-batch', type=int, default=16, help='GLOBAL A batch')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stop-after-epoch', type=int, help='Smoke-only manual checkpoint pause')
    return p


def main():
    parser = create_parser()
    args = parser.parse_args()
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.epochs < 1 or args.workers < 0 or args.log_every < 1:
        parser.error('Invalid epoch/worker/log settings')
    if any(b % args.world_size or b // args.world_size < 2 for b in (args.image_batch, args.fusion_batch)):
        parser.error('Global batches must divide world size, with >=2 samples per rank')
    if args.stop_after_epoch is not None and (not args.smoke or args.stop_after_epoch < 1):
        parser.error('stop-after-epoch is smoke-only')
    if args.stage in ('image_test', 'test') and args.world_size != 1:
        parser.error('Final Test must be single-process')
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if args.world_size > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    report = rank0_call(lambda: build_manifest(args.dataset_root, args.manifest, seed=args.seed), args.world_size)
    directory = args.output_root / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        lock = (directory / 'run.lock').open('a')
    def claim_output():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ('images', 'fusion') and not args.resume and (
                (directory / 'last.pt').exists() or (directory / 'metrics.jsonl').exists()):
            raise FileExistsError('Existing stage: use --resume or new output root')
    rank0_call(claim_output, args.world_size)
    def emit(record):
        if args.rank != 0:
            return
        record = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': args.stage, **record}
        line = json.dumps(record)
        print(line, flush=True)
        with (directory / 'metrics.jsonl').open('a') as handle:
            handle.write(line + '\n')
    emit({'event': 'split_report', **report})
    common = common_signature(args)
    if args.stage in ('images', 'fusion'):
        train_stage(args, directory, common, emit, device)
    else:
        final_evaluation(args, directory, common, emit, device)
    if args.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
