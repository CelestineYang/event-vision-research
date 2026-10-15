"""Isolated EventMamba V2: fixed 350 epochs, final-only official Test evaluation.

Single CUDA process or torchrun DDP (global batch=32). Never imports V1 encoder.
Smoke uses Train recordings for both training and evaluation.
"""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from create_gcstr_splits import DEFAULT_ROOT
from secnet_only_dataset import official_rows
from train_secnet_only import seed_all, atomic_save
from eventmamba_v2_model import EventMamba
from eventmamba_v2_data import WindowDataset, build_cache, cache_signature
from eventmamba_v2_metrics import smoothed_loss, SampleVotes


def read_gpu_status():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                        "--format=csv,noheader,nounits"], timeout=5, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(), "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    generator.set_state(state["loader"])


def train_epoch(model, loader, optimizer, device, epoch, world, emit, log_every):
    model.train()
    total = np.zeros(5, dtype=np.float64)  # smoothed loss, CE, top1, top5, count
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        events = batch["events"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(events)
        objective = smoothed_loss(logits, labels)
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite training loss")
        objective.backward()
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
        if not finite:
            raise FloatingPointError("Nonfinite training gradients")
        optimizer.step()
        count = len(labels)
        ce = torch.nn.functional.cross_entropy(logits.detach(), labels)
        top = logits.detach().topk(5, 1).indices
        total += [objective.item() * count, ce.item() * count,
                  (logits.detach().argmax(1) == labels).sum().item(), (top == labels[:, None]).any(1).sum().item(), count]
        if step == 1 or step % log_every == 0 or step == len(loader):
            sums = torch.tensor(total, device=device)
            if world > 1:
                dist.all_reduce(sums)
            values = sums.cpu().numpy()
            emit({"event": "progress", "phase": "train", "epoch": epoch, "step": step, "steps": len(loader),
                  "loss": float(values[0] / values[4]), "ce_loss": float(values[1] / values[4]),
                  "top1": float(values[2] / values[4]), "top5": float(values[3] / values[4]),
                  "windows": int(values[4]), "seconds": time.monotonic() - started})
    sums = torch.tensor(total, device=device)
    if world > 1:
        dist.all_reduce(sums)
    total = sums.cpu().numpy()
    if total[4] == 0:
        raise ValueError("No full training batch")
    return {"loss": float(total[0] / total[4]), "ce_loss": float(total[1] / total[4]),
            "top1": float(total[2] / total[4]), "top5": float(total[3] / total[4]),
            "windows": int(total[4]), "seconds": time.monotonic() - started}


@torch.no_grad()
def evaluate(model, data, device, batch_size, workers, emit, log_every, prediction_path):
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=False,
                        pin_memory=True, multiprocessing_context="spawn" if workers else None)
    votes = SampleVotes(data.sample_labels)
    totals = np.zeros(5, dtype=np.float64)
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        labels = batch["label"].to(device)
        logits = model(batch["events"].to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        ce = torch.nn.functional.cross_entropy(logits, labels)
        loss = smoothed_loss(logits, labels)
        top = logits.topk(5, 1).indices
        count = len(labels)
        totals += [ce.item() * count, loss.item() * count, (logits.argmax(1) == labels).sum().item(),
                   (top == labels[:, None]).any(1).sum().item(), count]
        votes.update(batch["sample_index"].numpy(), batch["window_index"].numpy(),
                     logits.softmax(1).cpu().numpy(), batch["label"].numpy())
        if step == 1 or step % log_every == 0 or step == len(loader):
            emit({"event": "progress", "phase": "final_eval", "step": step, "steps": len(loader),
                  "windows": int(totals[4]), "window_top1": float(totals[2] / totals[4]),
                  "seconds": time.monotonic() - started})
    sample_metrics, rankings = votes.result()
    # Keep sufficient aggregate evidence to audit voting; never used to select.
    with prediction_path.with_suffix(".tmp.npz").open("wb") as handle:
        np.savez_compressed(handle, sample_ids=np.asarray(data.sample_ids), labels=data.sample_labels,
                            votes=votes.votes, counts=votes.counts, first=votes.first,
                            probabilities=votes.probs / votes.counts[:, None], ranks=rankings[:, :5])
    os.replace(prediction_path.with_suffix(".tmp.npz"), prediction_path)
    return {"window": {"windows": int(totals[4]), "loss": float(totals[0] / totals[4]),
                       "smoothed_loss": float(totals[1] / totals[4]),
                       "top1": float(totals[2] / totals[4]), "top5": float(totals[3] / totals[4])},
            "sample": sample_metrics, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=32, help="GLOBAL batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.batch_size % world or args.batch_size // world < 2 or args.epochs < 1:
        parser.error("Global batch must divide world size; each rank needs >=2 examples")
    if args.num_workers < 0 or args.prepare_workers < 1 or args.log_every < 1:
        parser.error("Invalid worker/log configuration")
    if args.stop_after_epoch is not None and (not args.smoke or args.stop_after_epoch < 1):
        parser.error("stop-after-epoch is smoke-only")
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    if rank == 0:
        lock = (args.output_dir / "run.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and (args.output_dir / "last.pt").exists():
            raise FileExistsError("Existing checkpoint: use --resume or a new run directory")
        if not args.resume and log_path.exists():
            # It is safe to relaunch interrupted preparation, but not trained steps.
            if any(json.loads(line).get("event") in ("start", "epoch") for line in log_path.read_text().splitlines()):
                raise FileExistsError("Existing run: use --resume or a new directory")
    def emit(record):
        if rank == 0:
            record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
            line = json.dumps(record)
            print(line, flush=True)
            with log_path.open("a") as handle:
                handle.write(line + "\n")
    rows = official_rows(args.dataset_root)
    train_rows = rows["train"][:4] if args.smoke else rows["train"]
    emit({"event": "prepare", "pid": os.getpid(), "smoke": args.smoke,
          "official_samples": {s: len(v) for s, v in rows.items()}, "gpu_status": read_gpu_status()})
    train_path = args.cache_dir / "train.h5"
    if rank == 0:
        build_cache(train_path, args.dataset_root, train_rows, training=True, seed=args.seed,
                    workers=args.prepare_workers, emit=emit)
    if world > 1:
        dist.barrier()
    full_train = WindowDataset(train_path)
    expected_cache = cache_signature(args.dataset_root, train_rows, True, args.seed)
    if full_train.signature != expected_cache:
        raise ValueError("Wrong training cache")
    train_data = Subset(full_train, range(min(2 * args.batch_size, len(full_train)))) if args.smoke else full_train
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True) if world > 1 else None
    seed_all(args.seed)
    model = EventMamba(num_classes=200).to(device)
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
    bare = model.module if world > 1 else model
    seed_all(args.seed + rank)
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(train_data, batch_size=args.batch_size // world,
                        shuffle=sampler is None, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True, generator=generator,
                        multiprocessing_context="spawn" if args.num_workers else None)
    epochs = 2 if args.smoke else args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    from mamba_ssm.modules.mamba_simple import Mamba
    import causal_conv1d_cuda
    import selective_scan_cuda
    here = Path(__file__).resolve().parent
    names = ["eventmamba_v2_model.py", "eventmamba_v2_layers.py", "eventmamba_v2_data.py",
             "eventmamba_v2_metrics.py", "train_eventmamba_v2_only.py", "train_secnet_only.py"]
    dependency_files = [inspect.getfile(Mamba), inspect.getfile(causal_conv1d_cuda), inspect.getfile(selective_scan_cuda)]
    signature = {"protocol": "eventmamba_v2_dailydvs200_v1", "upstream_commit": "7f61fd397cf69f10f0f92a182109d36cd7b6fdbb",
        "smoke": args.smoke, "epochs": epochs, "batch_size": args.batch_size, "world_size": world,
        "seed": args.seed, "lr": 1e-3, "weight_decay": 1e-4, "smoothing": 0.2,
        "optimizer": "AdamW", "scheduler": "CosineAnnealingLR_eta_min0", "precision": "float32",
        "knn": "feature", "pooling": "attention_local_and_global", "bimamba": "v2",
        "cache_signature": expected_cache, "torch": torch.__version__,
        "official_manifest_hashes": {s: hashlib.sha256((args.dataset_root / "splits" / f"{s}.csv").read_bytes()).hexdigest() for s in ("train", "test")},
        "evaluation": "final_only; window_CE; sample_hard_votes_positive_top5; mean_probability_CE",
        "source_hashes": {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in names},
        "dependency_hashes": {Path(n).name: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in dependency_files}}
    start = 1
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Resume protocol/code/dependency mismatch")
        bare.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng_by_rank"][rank], generator)
        start = state["epoch"] + 1
    emit({"event": "start", "pid": os.getpid(), "start_epoch": start, "signature": signature,
          "train_windows": len(full_train), "steps_per_epoch": len(loader),
          "parameters": sum(p.numel() for p in bare.parameters()),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_status": read_gpu_status()})
    for epoch in range(start, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]["lr"]
        metrics = train_epoch(model, loader, optimizer, device, epoch, world, emit, args.log_every)
        scheduler.step()
        local_rng = rng_state(generator)
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, local_rng)
        else:
            all_rng[0] = local_rng
        if rank == 0:
            state = {"epoch": epoch, "signature": signature, "model": bare.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "rng_by_rank": all_rng, "train": metrics}
            atomic_save(state, args.output_dir / "last.pt")
            emit({"event": "epoch", "epoch": epoch, "lr": lr, "train": metrics,
                  "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20, "gpu_status": read_gpu_status()})
        if world > 1:
            dist.barrier()
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            emit({"event": "smoke_pause", "epoch": epoch})
            if world > 1:
                dist.destroy_process_group()
            return
    if rank == 0:
        # Only rank 0 evaluates: no padding/duplicating samples across ranks.
        state = torch.load(args.output_dir / "last.pt", map_location="cpu", weights_only=False)
        if state["epoch"] != epochs:
            raise ValueError("Final epoch required")
        atomic_save(state, args.output_dir / "final.pt")
        result_path = args.output_dir / ("smoke_eval.json" if args.smoke else "test_final.json")
        if result_path.exists():
            emit({"event": "evaluation_already_saved", "path": str(result_path)})
        else:
            # Convert back to ordinary BN for single-rank evaluation. State keys
            # are identical; SyncBN eval uses the same running statistics.
            evaluation_model = EventMamba(200).to(device)
            evaluation_model.load_state_dict(state["model"])
            eval_rows = train_rows if args.smoke else rows["test"]
            eval_path = args.cache_dir / ("train_smoke_eval.h5" if args.smoke else "test.h5")
            build_cache(eval_path, args.dataset_root, eval_rows, training=False, seed=args.seed,
                        workers=args.prepare_workers, emit=emit)
            seed_all(args.seed + 100000)
            eval_data = WindowDataset(eval_path)
            metrics = evaluate(evaluation_model, eval_data, device, args.batch_size, args.num_workers,
                               emit, args.log_every, args.output_dir / "final_sample_predictions.npz")
            result = {"event": "final_evaluation", "split": "train_smoke" if args.smoke else "test",
                      "epoch": epochs, "checkpoint": str(args.output_dir / "final.pt"),
                      "evaluation_seed": args.seed + 100000, **metrics}
            temporary = result_path.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(temporary, result_path)
            emit(result)
        emit({"event": "finished", "epoch": epochs})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
