"""Train and evaluate the initial DailyDVS-200 representation baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from baseline_models import build_baseline
from daily_dvs200_dataset import DailyDVS200Dataset, create_dataloader


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model.forward_batch(batch)
            loss = criterion(logits, batch["label"])
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            batch_size = batch["label"].shape[0]
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("no batches were processed")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "samples": float(sample_count),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_name: str,
    val_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("dataset_splits.csv"))
    parser.add_argument(
        "--model", choices=("frame", "voxel", "points", "concat"), default="concat"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--fixed-point-sampling", action="store_true")
    parser.add_argument("--cache-in-memory", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_dataset = DailyDVS200Dataset(
        args.manifest,
        split="train",
        training=True,
        seed=args.seed,
        max_samples=args.max_train_samples,
        deterministic_points=args.fixed_point_sampling,
        cache_in_memory=args.cache_in_memory,
    )
    val_dataset = DailyDVS200Dataset(
        args.manifest,
        split="val",
        training=False,
        seed=args.seed,
        max_samples=args.max_val_samples,
        cache_in_memory=args.cache_in_memory,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_baseline(args.model, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_accuracy = -1.0
    print(
        json.dumps(
            {
                "model": args.model,
                "device": str(device),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            }
        )
    )

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.max_train_batches
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, None, args.max_val_batches
        )
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        print(json.dumps(record))
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch + 1,
            args.model,
            val_metrics,
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch + 1,
                args.model,
                val_metrics,
            )


if __name__ == "__main__":
    main()
