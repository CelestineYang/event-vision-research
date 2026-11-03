"""Paper training hyperparameters + user-selected official preprocessing stride.

Single GPU; all training begins randomly unless --resume is explicitly supplied.
Test is evaluated every epoch as in the public trainer (no independent Val).
"""
import argparse
from contextlib import contextmanager
import fcntl
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from eventmamba_paper_data import DEFAULT_CONFIG, HERE, PaperPoints, load_config, sha, write_json
from eventmamba_v2_metrics import SampleVotes, smoothed_loss
from eventmamba_v2_model import EventMamba


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def evaluation_rng(seed):
    # FPS has a random start even in eval. Fix it without advancing train RNG.
    state = rng_state()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state)


def atomic_save(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def optimizer_for(model, config):
    recipe = config["training"]
    if recipe["optimizer"] != "Adam":
        raise ValueError("This recipe requires Adam, not AdamW")
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["lr"],
                                 weight_decay=recipe["weight_decay"],
                                 betas=tuple(recipe["betas"]), eps=recipe["eps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=recipe["epochs"], eta_min=recipe["eta_min"])
    return optimizer, scheduler


def run_epoch(model, loader, device, *, optimizer=None, sample_labels=None, emit=print):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(5, dtype=np.float64)
    votes = SampleVotes(sample_labels, classes=200) if sample_labels is not None else None
    started = time.monotonic()
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, 1):
            x = batch["events"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = smoothed_loss(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite logits/loss")
            if training:
                loss.backward()
                finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                if not finite:
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
            detached = logits.detach()
            top = detached.topk(5, dim=1).indices
            n = len(y)
            totals += [loss.item() * n, torch.nn.functional.cross_entropy(detached, y).item() * n,
                       (top[:, 0] == y).sum().item(), (top == y[:, None]).any(1).sum().item(), n]
            if votes is not None:
                votes.update(batch["sample_index"].numpy(), batch["window_index"].numpy(),
                             detached.softmax(1).cpu().numpy(), batch["label"].numpy())
            if step == 1 or step % 100 == 0 or step == len(loader):
                emit({"phase": "train" if training else "eval", "step": step, "steps": len(loader),
                      "windows": int(totals[4]), "window_top1": float(totals[2] / totals[4]),
                      "seconds": time.monotonic() - started})
    if totals[4] == 0:
        raise ValueError("No complete training batch / empty evaluation")
    metrics = {"windows": int(totals[4]), "smoothed_loss": float(totals[0] / totals[4]),
               "ce": float(totals[1] / totals[4]), "window_top1": float(totals[2] / totals[4]),
               "window_top5": float(totals[3] / totals[4]), "seconds": time.monotonic() - started}
    if votes is not None:
        metrics["video"], _ = votes.result()
        if int(votes.counts.sum()) != len(loader.dataset):
            raise ValueError("Evaluation omitted or duplicated windows")
    return metrics, votes


def save_predictions(path, data, votes):
    temporary = Path(path).with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, sample_ids=data.sample_ids, labels=data.sample_labels,
                            counts=votes.counts, votes=votes.votes, first=votes.first,
                            probabilities=votes.probs / votes.counts[:, None])
    os.replace(temporary, path)


def runtime_identity():
    from mamba_ssm.modules.mamba_simple import Mamba
    import causal_conv1d_cuda
    import selective_scan_cuda
    dependency_paths = [inspect.getfile(Mamba), inspect.getfile(causal_conv1d_cuda), inspect.getfile(selective_scan_cuda)]
    names = ["train_eventmamba_paper.py", "eventmamba_paper_data.py", "eventmamba_v2_model.py",
             "eventmamba_v2_layers.py", "eventmamba_v2_metrics.py", "eventmamba_v2_upstream/dataprocess/uti.py"]
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
            "sources": {name: sha(HERE / name) for name in names},
            "dependencies": {Path(name).name: sha(name) for name in dependency_paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Smoke only; scheduler remains 350 epochs")
    args = parser.parse_args()
    if args.workers < 0 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Requires nonnegative workers and a single GPU process")
    if args.stop_after_epoch is not None and (not args.smoke or not 1 <= args.stop_after_epoch <= 2):
        parser.error("stop-after-epoch is restricted to the two-epoch smoke test")
    if args.smoke and "smoke" not in args.output.name:
        parser.error("Smoke requires a separate output directory containing 'smoke'")
    config = load_config(args.config)
    receipt = json.loads((args.data / "COMPLETE.json").read_text())
    scope = "smoke" if args.smoke else "full"
    if receipt["config"] != config or receipt["scope"] != scope:
        raise ValueError("Prepared recipe/scope mismatch; old 1-second caches are not accepted")
    for split in ("train", "test"):
        if sha(args.data / f"{split}.h5") != receipt["hashes"][split]:
            raise ValueError("Prepared artifact checksum mismatch")
    training, evaluation = [PaperPoints(args.data / f"{split}.h5", exclude_empty=True) for split in ("train", "test")]
    for ds, is_train in [(training, True), (evaluation, False)]:
        if ds.signature["data"] != config["data"] or ds.signature["training"] != is_train or ds.signature["scope"] != scope:
            raise ValueError("Incorrect cache role or protocol")
        for name, digest in ds.signature["sources"].items():
            if sha(HERE / name) != digest:
                raise ValueError("Preprocessing source changed since cache build")
    if not args.smoke and set(training.sample_ids) & set(evaluation.sample_ids):
        raise ValueError("Train/Test recording overlap")
    for split, data in [("train", training), ("test", evaluation)]:
        coverage = receipt["coverage"][split]
        if coverage["evaluated_recordings"] != len(data.sample_ids) or coverage["windows"] != len(data):
            raise ValueError("Coverage receipt mismatch")
        if {r["sample_id"] for r in coverage["excluded_recordings"]} != set(data.excluded_ids):
            raise ValueError("Excluded recording report mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and any(p.name != "run.lock" for p in args.output.iterdir()):
        raise ValueError("Existing output; use --resume or a new directory")
    identity = {"config": config, "cache": receipt, "runtime": runtime_identity(),
                "workers": args.workers, "smoke": args.smoke}
    torch.set_num_threads(2)
    seed_all(config["seed"])
    device = torch.device("cuda:0")
    model = EventMamba(config["classes"]).to(device)
    optimizer, scheduler = optimizer_for(model, config)
    batch_size = config["training"]["batch_size"]
    train_data = Subset(training, np.linspace(0, len(training) - 1,
                        min(len(training), batch_size * 2), dtype=int).tolist()) if args.smoke else training
    generator = torch.Generator().manual_seed(config["seed"])
    loader_options = dict(batch_size=batch_size, num_workers=args.workers, pin_memory=True,
                          multiprocessing_context="spawn" if args.workers else None)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, generator=generator, **loader_options)
    eval_loader = DataLoader(evaluation, shuffle=False, drop_last=False,
                             generator=torch.Generator().manual_seed(config["training"]["evaluation_seed"]), **loader_options)
    if not len(train_loader):
        raise ValueError("Fewer than 32 training windows")
    start, best = 1, -1.0
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume identity mismatch (recipe, data, code, or dependency changed)")
        model.load_state_dict(state["net"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["loader_rng"])
        restore_rng(state["rng"])
        start, best = state["epoch"] + 1, state["best_video_top1"]
        # Repair a crash between atomic checkpoint and history append.
        path = args.output / "metrics.jsonl"
        history = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["epoch"] < state["epoch"]:
                    history.append(row)
        history.append(state["metrics"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in history))
        os.replace(temporary, path)
        del state
    else:
        write_json(args.output / "config.json", identity)
    print(json.dumps({"event": "start", "pid": os.getpid(), "epoch": start, "scope": scope,
                      "train_windows": len(training), "eval_windows": len(evaluation),
                      "eval_recordings": len(evaluation.sample_ids), "excluded_eval_recordings": evaluation.excluded_ids,
                      "steps_per_epoch": len(train_loader), "batch_size": batch_size,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    epochs = 2 if args.smoke else config["training"]["epochs"]
    emit = lambda value: print(json.dumps(value), flush=True)
    for epoch in range(start, epochs + 1):
        write_json(args.output / "status.json", {"status": "running", "epoch": epoch,
                   "epochs": epochs, "best_video_top1": best})
        torch.cuda.reset_peak_memory_stats()
        lr = optimizer.param_groups[0]["lr"]
        tr, _ = run_epoch(model, train_loader, device, optimizer=optimizer, emit=emit)
        scheduler.step()
        with evaluation_rng(config["training"]["evaluation_seed"]):
            ev, votes = run_epoch(model, eval_loader, device, sample_labels=evaluation.sample_labels, emit=emit)
        improved = ev["video"]["top1"] > best
        if improved:
            best = ev["video"]["top1"]
        record = {"epoch": epoch, "lr": lr, "train": tr, "evaluation": ev,
                  "eval_split": receipt["eval_split"], "best_video_top1": best,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        state = {"net": model.state_dict(), "epoch": epoch, "acc": ev["video"]["top1"],
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": rng_state(), "loader_rng": generator.get_state(), "identity": identity,
                 "best_video_top1": best, "metrics": record}
        if improved:
            atomic_save(args.output / "best_video_top1.pt", state)
            save_predictions(args.output / "predictions_best.npz", evaluation, votes)
        atomic_save(args.output / "last.pt", state)
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if epoch == epochs:
            atomic_save(args.output / "final.pt", state)
            save_predictions(args.output / "predictions_final.npz", evaluation, votes)
        write_json(args.output / "status.json", {"status": ("smoke_complete" if args.smoke else "complete")
                   if epoch == epochs else "running", "epoch": epoch, "epochs": epochs, "best_video_top1": best})
        emit(record)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            write_json(args.output / "status.json", {"status": "smoke_paused", "epoch": epoch})
            break
    training.close()
    evaluation.close()


if __name__ == "__main__":
    main()
