"""150 fixed epochs, official Train only; evaluate final weights on Test once.

Smoke mode uses only official Train samples, including its evaluation pass.
Resume restarts after the last fully saved epoch, with optimizer/scheduler/RNG.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from create_gcstr_splits import DEFAULT_ROOT
from secnet_only import SECNetOnly
from secnet_only_dataset import official_rows, SECNetOnlyDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_epoch(model, loader, device, *, optimizer=None, log_every=20, emit=print, epoch=0):
    model.train(optimizer is not None)
    total = correct1 = correct5 = 0
    loss_sum = 0.0
    started = time.monotonic()
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(loader, 1):
            events, labels = batch["events"].to(device), batch["label"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(events)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if optimizer is not None:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
            count = labels.numel()
            total += count
            loss_sum += loss.item() * count
            correct1 += (logits.argmax(1) == labels).sum().item()
            correct5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
            if step == 1 or step % log_every == 0 or step == len(loader):
                emit({"event": "progress", "phase": "train" if optimizer else "eval",
                      "epoch": epoch, "step": step, "steps": len(loader), "samples": total,
                      "loss": loss_sum / total, "top1": correct1 / total,
                      "seconds": time.monotonic() - started})
    if not total:
        raise ValueError("Empty loader")
    return {"loss": loss_sum / total, "top1": correct1 / total, "top5": correct5 / total,
            "samples": total, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Smoke only: test restart")
    args = parser.parse_args()
    if args.batch_size < 2 or args.num_workers < 0 or args.log_every < 1:
        parser.error("Invalid batch/worker/log settings")
    if args.stop_after_epoch is not None and (not args.smoke or args.stop_after_epoch < 1):
        parser.error("stop-after-epoch is only for smoke testing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_path = args.output_dir / "metrics.jsonl"
    if not args.resume and (log_path.exists() or (args.output_dir / "last.pt").exists()):
        raise FileExistsError("Existing run: specify --resume or a new output directory")
    def emit(record):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        line = json.dumps(record)
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        # PyTorch 2.0.1 oneDNN GELU backward produced nonfinite gradients in
        # CPU smoke tests; native CPU kernels passed. Does not change encoder.
        torch.backends.mkldnn.enabled = False
    rows = official_rows(args.dataset_root)
    for split in rows:
        missing = [r["npz_relpath"] for r in rows[split]
                   if not (args.dataset_root / r["npz_relpath"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} {split} files: {missing[:3]}")
    epochs = 2 if args.smoke else 150
    root = Path(__file__).resolve().parent
    sources = ["secnet_encoder.py", "secnet_only.py", "secnet_only_dataset.py",
               "train_secnet_only.py", "gcstr_fusion_dataset.py", "gcstr_augmentation.py",
               "gcstr_representations.py", "create_gcstr_splits.py"]
    signature = {"protocol": "official_train_final_test_v1", "smoke": args.smoke,
        "epochs": epochs, "batch_size": args.batch_size, "seed": args.seed,
        "point_count": 8192, "augmentation": "spatial", "augmentation_count": 5,
        "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR", "eta_min": 0.0, "precision": "float32",
        "device_type": device.type, "cpu_mkldnn": False if device.type == "cpu" else None,
        "head": "concat_mean_max_author_mlp256_200", "dataset_root": str(args.dataset_root.resolve()),
        "split_hashes": {s: hashlib.sha256((args.dataset_root / "splits" / f"{s}.csv").read_bytes()).hexdigest() for s in rows},
        "source_hashes": {s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in sources}}
    train_rows = rows["train"][:2 * args.batch_size] if args.smoke else rows["train"]
    if len(train_rows) % args.batch_size == 1:
        raise ValueError("BatchNorm head requires batch_size yielding no singleton training batch")
    train_data = SECNetOnlyDataset(train_rows, dataset_root=args.dataset_root, training=True, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", generator=generator)
    model = SECNetOnly().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    start = 1
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Resume configuration/source mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        generator.set_state(state["rng"]["loader"])
        start = state["epoch"] + 1
    emit({"event": "start", "pid": os.getpid(), "start_epoch": start,
          "official_samples": {s: len(r) for s, r in rows.items()}, "signature": signature,
          "parameters": sum(p.numel() for p in model.parameters()),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    for epoch in range(start, epochs + 1):
        train_data.epoch = epoch
        lr = optimizer.param_groups[0]["lr"]
        metrics = run_epoch(model, loader, device, optimizer=optimizer,
                            log_every=args.log_every, emit=emit, epoch=epoch)
        scheduler.step()
        state = {"epoch": epoch, "signature": signature, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "train": metrics, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state(), "loader": generator.get_state(),
                 "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}}
        atomic_save(state, args.output_dir / "last.pt")
        emit({"event": "epoch", "epoch": epoch, "lr": lr, "train": metrics,
              "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0})
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            emit({"event": "smoke_pause", "epoch": epoch})
            return
    # Always evaluate the saved final-epoch state, never a selected best state.
    state = torch.load(args.output_dir / "last.pt", map_location="cpu", weights_only=False)
    if state["epoch"] != epochs:
        raise ValueError("Final epoch checkpoint is required")
    atomic_save(state, args.output_dir / "final.pt")
    model.load_state_dict(state["model"])
    result_path = args.output_dir / ("smoke_eval.json" if args.smoke else "test_final.json")
    if result_path.exists():
        emit({"event": "evaluation_already_saved", "path": str(result_path)})
        return
    seed_all(args.seed + 100000)
    # Smoke evaluation deliberately uses Train, never official Test events.
    eval_rows = train_rows if args.smoke else rows["test"]
    evaluation = SECNetOnlyDataset(eval_rows, dataset_root=args.dataset_root, seed=args.seed)
    eval_loader = DataLoader(evaluation, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    metrics = run_epoch(model, eval_loader, device, log_every=args.log_every, emit=emit, epoch=epochs)
    result = {"event": "final_evaluation", "split": "train_smoke" if args.smoke else "test",
              "epoch": epochs, "checkpoint": str(args.output_dir / "final.pt"),
              "evaluation_seed": args.seed + 100000, **metrics}
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, result_path)
    emit(result)


if __name__ == "__main__":
    main()
