"""Train custom gCSTR + point fusion; never evaluate official test here."""

import argparse
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from create_gcstr_splits import DEFAULT_ROOT
from gcstr_fusion_cached_dataset import CachedFusionDataset
from gcstr_secnet_fusion import GCSTRSECNetFusion, VIEWS


def log(**items):
    print(json.dumps(items, ensure_ascii=False), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_atomic(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_epoch(model, loader, device, epoch, optimizer, limit, log_every, point_aux_weight=0.0):
    training = optimizer is not None
    phase = "train" if training else "val"
    model.train(training)

    total_batches = len(loader)
    if limit is not None:
        total_batches = min(total_batches, limit)

    log(event="phase_start", epoch=epoch, phase=phase,
        batches=total_batches)

    started = time.monotonic()
    loss_sum = correct1 = correct5 = count = 0
    point_loss_sum = point_correct1 = point_correct5 = objective_sum = 0
    batches = itertools.islice(loader, total_batches)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(batches, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            if training:
                optimizer.zero_grad(set_to_none=True)

            if point_aux_weight > 0:
                outputs = model.forward_batch_outputs(batch)
                logits, point_logits = outputs['logits'], outputs['point_logits']
                if point_logits is None:
                    raise RuntimeError('Auxiliary weight requires a point head')
            else:
                logits, point_logits = model.forward_batch(batch), None
            labels = batch["label"]
            loss = torch.nn.functional.cross_entropy(logits, labels)
            point_loss = (torch.nn.functional.cross_entropy(point_logits, labels)
                          if point_logits is not None else loss.new_zeros(()))
            objective = loss + point_aux_weight * point_loss
            if not torch.isfinite(objective):
                raise RuntimeError(f"Non-finite loss: {phase}, step {step}")

            if training:
                objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            size = labels.numel()
            loss_sum += loss.item() * size
            objective_sum += objective.item() * size
            if point_logits is not None:
                point_loss_sum += point_loss.item() * size
                point_correct1 += (point_logits.argmax(1) == labels).sum().item()
                point_correct5 += (point_logits.topk(min(5, point_logits.shape[1]), 1).indices
                                   == labels[:, None]).any(1).sum().item()
            correct1 += (logits.argmax(1) == labels).sum().item()
            correct5 += (
                logits.topk(min(5, logits.shape[1]), dim=1).indices
                == labels[:, None]
            ).any(1).sum().item()
            count += size

            if step == 1 or step % log_every == 0 or step == total_batches:
                elapsed = time.monotonic() - started
                log(
                    event="progress", epoch=epoch, phase=phase,
                    batch=step, batches=total_batches,
                    loss=loss_sum / count,
                    top1=correct1 / count,
                    **({'point_loss': point_loss_sum / count, 'point_top1': point_correct1 / count,
                        'objective_loss': objective_sum / count} if point_logits is not None else {}),
                    elapsed_seconds=round(elapsed, 1),
                    eta_seconds=round(
                        elapsed / step * (total_batches - step), 1
                    ),
                )

    if count == 0:
        raise RuntimeError("No samples processed")

    metrics = {
        "loss": loss_sum / count,
        "top1": correct1 / count,
        "top5": correct5 / count,
        "samples": count,
        "seconds": round(time.monotonic() - started, 1),
    }
    if point_aux_weight > 0:
        metrics.update(point_loss=point_loss_sum / count, point_top1=point_correct1 / count,
                       point_top5=point_correct5 / count, objective_loss=objective_sum / count)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("gcstr_splits_seed0.csv"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Cache is opt-in because augmented full-event caches may be large.
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--point-encoder", choices=("secnet", "eventmamba"), default="secnet")
    parser.add_argument('--fusion', choices=('prototype', 'local'), default='prototype')
    parser.add_argument('--point-aux-weight', type=float, default=0.0)
    parser.add_argument('--selection-metric', choices=('loss', 'top1'), default='loss',
                        help='Metric used by early stopping and best.pt; best_top1.pt is also saved')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-count", type=int, default=8192)
    parser.add_argument("--augmentation", choices=("none", "spatial"),
                        default="spatial")
    parser.add_argument("--augmentation-count", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()
    if not np.isfinite(args.point_aux_weight) or args.point_aux_weight < 0:
        parser.error('point_aux_weight must be finite and nonnegative')
    if args.point_encoder != 'eventmamba' and (args.fusion != 'prototype' or args.point_aux_weight):
        parser.error('Local fusion and point auxiliary supervision currently require eventmamba')

    for name in ("epochs", "batch_size", "patience", "log_every",
                 "augmentation_count"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.num_workers < 0 or args.point_count < 1024:
        parser.error("num_workers must be >= 0 and point_count >= 1024")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid optimizer settings")
    for limit in (args.max_train_batches, args.max_val_batches):
        if limit is not None and limit < 1:
            parser.error("Batch limits must be positive")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; training was not started")
        torch.zeros(1, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None and any(args.output_dir.iterdir()):
        raise RuntimeError("Output directory is not empty; use a new directory")

    signature = {
        "architecture": ("gcstr_secnet_prototype_v1" if args.point_encoder == "secnet"
                         else "gcstr_eventmamba_v1_prototype_v1"),
        "manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "dataset_root": str(args.dataset_root.resolve()),
        "point_count": args.point_count,
        "augmentation": args.augmentation,
        "augmentation_count": args.augmentation_count,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }

    seed_all(args.seed)
    if args.point_aux_weight > 0:
        signature['point_aux'] = {'weight': args.point_aux_weight, 'head': 'last_stage_mean_ln_linear_v1'}
    if args.selection_metric != 'loss':
        signature['selection_metric'] = args.selection_metric
    if args.point_encoder == "eventmamba":
        from eventmamba_encoder import UPSTREAM_COMMIT
        from gcstr_eventmamba_fusion import GCSTREventMambaFusion
        signature["eventmamba"] = {
            "source_commit": UPSTREAM_COMMIT, "variant": "v1",
            "groups": [1024, 512, 256], "neighbors": 24,
            "channels": [64, 128, 256], "coordinates": "t,x,y",
            "mixer": "bimamba_v2",
        }
        if args.fusion == 'local':
            from gcstr_local_fusion import GCSTRLocalEventMambaFusion
            signature['architecture'] = 'gcstr_eventmamba_v1_local_v1'
            signature['local_fusion'] = {'layer': 'layer3', 'dim': 128, 'heads': 4,
                'radius': 1, 'stride': 16, 'residual_scale': 0.1, 'views': 'equal',
                'projection': 'gcstr_zoom_endpoints_resnet_stride_v1'}
            model = GCSTRLocalEventMambaFusion(point_aux=args.point_aux_weight > 0)
        else:
            model = GCSTREventMambaFusion(freeze_images=True, point_aux=args.point_aux_weight > 0)
    else:
        model = GCSTRSECNetFusion(freeze_images=True)
    checkpoint = None

    if args.resume is not None:
        # Load only your own trusted checkpoint files.
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if checkpoint["signature"] != signature:
            raise ValueError("Resume configuration/manifest mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_image_checkpoints({
            view: (
                f"gcstr_checkpoints/full_{view}_spatial_seed0/"
                f"{view}_seed0_best.pt"
            )
            for view in VIEWS
        })

    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch, best_loss, bad_epochs = 1, float("inf"), 0
    best_top1 = float('-inf')
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        bad_epochs = checkpoint["bad_epochs"]
        best_top1 = checkpoint.get('best_top1', float('-inf'))
        if 'best_top1' not in checkpoint:
            log(event='legacy_resume', message='best_top1 tracking starts at this resume; historical peak weights unavailable')
        del checkpoint

    common = {
        "manifest_path": args.manifest,
        "dataset_root": args.dataset_root,
        "point_count": args.point_count,
        "cache_dir": args.cache_dir,
        "seed": args.seed,
    }
    train_data = CachedFusionDataset(
        split="train", augmentation=args.augmentation,
        augmentation_count=args.augmentation_count,
        augmentation_seed=args.seed, **common
    )
    val_data = CachedFusionDataset(split="val", **common)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    log(
        event="start", config=config,
        train_samples=len(train_data), val_samples=len(val_data),
        start_epoch=start_epoch,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )

    if start_epoch > args.epochs:
        log(event="already_finished")
        return
    if bad_epochs >= args.patience:
        log(event="already_early_stopped")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        # Explicit epoch seeding makes epoch-boundary resume reproducible
        # on the same environment, without storing worker RNG state.
        epoch_seed = args.seed + epoch
        seed_all(epoch_seed)
        train_data.set_epoch(epoch)

        loader_args = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": False,
        }
        train_loader = DataLoader(
            train_data, shuffle=True,
            generator=torch.Generator().manual_seed(epoch_seed),
            **loader_args
        )
        train_metrics = run_epoch(
            model, train_loader, device, epoch, optimizer,
            args.max_train_batches, args.log_every,
            args.point_aux_weight,
        )

        # Fixed validation FPS random sequence each epoch.
        seed_all(args.seed + 1000000)
        val_loader = DataLoader(
            val_data, shuffle=False,
            generator=torch.Generator().manual_seed(args.seed),
            **loader_args
        )
        val_metrics = run_epoch(
            model, val_loader, device, epoch, None,
            args.max_val_batches, args.log_every,
            args.point_aux_weight,
        )

        loss_improved = val_metrics['loss'] < best_loss
        top1_improved = val_metrics['top1'] > best_top1
        improved = loss_improved if args.selection_metric == 'loss' else top1_improved
        best_loss = min(best_loss, val_metrics['loss'])
        best_top1 = max(best_top1, val_metrics['top1'])
        if improved:
            bad_epochs = 0
        else:
            bad_epochs += 1

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_top1": best_top1,
            "bad_epochs": bad_epochs,
            "signature": signature,
            "config": config,
            "train": train_metrics,
            "val": val_metrics,
        }
        if improved:
            save_atomic(args.output_dir / "best.pt", state)
        if top1_improved:
            save_atomic(args.output_dir / 'best_top1.pt', state)
        save_atomic(args.output_dir / "last.pt", state)

        record = {
            "epoch": epoch, "train": train_metrics, "val": val_metrics,
            "best_val_loss": best_loss, "bad_epochs": bad_epochs,
            'best_val_top1': best_top1, 'selection_metric': args.selection_metric,
        }
        log(**record)
        with (args.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        if bad_epochs >= args.patience:
            log(event="early_stop", epoch=epoch)
            break

    log(event="finished")


if __name__ == "__main__":
    main()
