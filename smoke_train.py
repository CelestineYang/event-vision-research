"""Run one real end-to-end optimization step for baseline A."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from baseline_models import ThreeBranchConcatClassifier
from daily_dvs200_dataset import DailyDVS200Dataset, create_dataloader


def move_tensors(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("dataset_inventory.csv"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = DailyDVS200Dataset(
        args.manifest,
        training=True,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = ThreeBranchConcatClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch = move_tensors(next(iter(loader)), device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, features = model.forward_batch(batch, return_features=True)
    loss = criterion(logits, batch["label"])
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("device:", device)
    print("parameters:", parameter_count)
    print("batch_size:", logits.shape[0])
    print("logits:", tuple(logits.shape))
    print("branch_features:", {key: tuple(value.shape) for key, value in features.items()})
    print("labels:", batch["label"].tolist())
    print("loss:", float(loss.detach()))
    print("gradient_norm:", float(gradient_norm))
    if device.type == "cuda":
        print("max_cuda_memory_mb:", round(torch.cuda.max_memory_allocated(device) / 1024**2, 2))


if __name__ == "__main__":
    main()
