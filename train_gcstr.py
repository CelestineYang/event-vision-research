"""Train one ImageNet-pretrained ResNet-18 gCSTR branch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gcstr_dataset import DailyDVS200GCSTRDataset
from gcstr_augmentation import AUGMENTATIONS
from gcstr_models import GCSTRBranchClassifier
from gcstr_representations import GCSTRConfig, VIEW_NAMES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: GCSTRBranchClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = top5_correct = sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model.forward_batch(batch)
            labels = batch["label"]
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = labels.shape[0]
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum())
            top5_correct += int(
                (logits.topk(k=min(5, logits.shape[1]), dim=1).indices == labels[:, None])
                .any(dim=1)
                .sum()
            )
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("no samples were processed")
    return {
        "loss": loss_sum / sample_count,
        "top1": correct / sample_count,
        "top5": top5_correct / sample_count,
        "samples": sample_count,
    }


def save_checkpoint(
    path: Path,
    model: GCSTRBranchClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "view": args.view,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("gcstr_splits_seed0.csv"))
    parser.add_argument("--view", choices=VIEW_NAMES, default="cstr3")
    parser.add_argument("--cache-dir", type=Path, default=Path("gcstr_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("gcstr_checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augmentation", choices=AUGMENTATIONS, default="none")
    parser.add_argument("--augmentation-count", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    config = GCSTRConfig()
    common = dict(
        manifest_path=args.manifest,
        views=(args.view,),
        config=config,
        cache_dir=args.cache_dir,
        imagenet_normalize=not args.no_pretrained,
    )
    train_dataset = DailyDVS200GCSTRDataset(
        split="train", augmentation=args.augmentation,
        augmentation_count=args.augmentation_count,
        augmentation_seed=args.seed, **common,
    )
    val_dataset = DailyDVS200GCSTRDataset(split="val", **common)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = GCSTRBranchClassifier(
        args.view, pretrained=not args.no_pretrained
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(json.dumps({
        "event": "start",
        "view": args.view,
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "args": vars(args),
    }, default=str))

    best_loss = float("inf")
    last_improved_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.max_train_batches
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, None, args.max_val_batches
        )
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}))

        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss = val_metrics["loss"]
            save_checkpoint(
                args.output_dir / f"{args.view}_seed{args.seed}_best.pt",
                model, optimizer, epoch, args, val_metrics,
            )
        save_checkpoint(
            args.output_dir / f"{args.view}_seed{args.seed}_last.pt",
            model, optimizer, epoch, args, val_metrics,
        )
        if improved:
            last_improved_epoch = epoch
        elif epoch - last_improved_epoch >= args.patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch}))
            break


if __name__ == "__main__":
    main()
