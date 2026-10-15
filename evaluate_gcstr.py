"""Evaluate one or more trained gCSTR branches with PaSNet fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gcstr_dataset import DailyDVS200GCSTRDataset
from gcstr_models import GCSTRBranchClassifier, PaSNetProbabilityEnsemble
from gcstr_representations import VIEW_NAMES


def parse_checkpoint(value: str) -> tuple[str, Path]:
    try:
        view, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint must use VIEW=PATH") from error
    if view not in VIEW_NAMES:
        raise argparse.ArgumentTypeError(f"unknown view {view!r}")
    return view, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("gcstr_splits_seed0.csv"))
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--cache-dir", type=Path, default=Path("gcstr_cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    branches = {}
    for view, path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu")
        model = GCSTRBranchClassifier(view, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        branches[view] = model
    ensemble = PaSNetProbabilityEnsemble(branches).to(device).eval()
    views = tuple(branches)
    dataset = DailyDVS200GCSTRDataset(
        args.manifest, args.split, views=views, cache_dir=args.cache_dir
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    correct = top5_correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            moved = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            scores = ensemble.forward_batch(moved)
            labels = moved["label"]
            correct += int((scores.argmax(dim=1) == labels).sum())
            top5_correct += int(
                (scores.topk(5, dim=1).indices == labels[:, None]).any(dim=1).sum()
            )
            total += labels.shape[0]
    print(json.dumps({
        "split": args.split,
        "views": views,
        "samples": total,
        "top1": correct / total,
        "top5": top5_correct / total,
    }))


if __name__ == "__main__":
    main()
