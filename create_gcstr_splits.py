"""Create a leakage-free gCSTR manifest from DailyDVS-200 official train/test."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path(
    "/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200"
)
OUTPUT_FIELDS = (
    "sample_id",
    "split",
    "source_split",
    "npz_relpath",
    "event_relpath",
    "label_id",
    "action_id",
    "class_name",
    "participant_id",
    "movement_id",
    "scene_id",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stratified_train_validation_split(
    rows: list[dict[str, str]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows by class with an exact global validation target."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["label_id"])].append(row)
    if not grouped:
        raise ValueError("cannot split an empty manifest")

    target = round(len(rows) * validation_fraction)
    quotas: dict[int, int] = {}
    fractional: list[tuple[float, int]] = []
    for label, group in grouped.items():
        raw = len(group) * validation_fraction
        quota = min(max(math.floor(raw), 1), len(group) - 1)
        quotas[label] = quota
        fractional.append((raw - math.floor(raw), label))

    remaining = target - sum(quotas.values())
    if remaining > 0:
        for _, label in sorted(fractional, key=lambda item: (-item[0], item[1])):
            if remaining == 0:
                break
            if quotas[label] < len(grouped[label]) - 1:
                quotas[label] += 1
                remaining -= 1
    elif remaining < 0:
        for _, label in sorted(fractional, key=lambda item: (item[0], item[1])):
            if remaining == 0:
                break
            if quotas[label] > 1:
                quotas[label] -= 1
                remaining += 1
    if remaining != 0:
        raise RuntimeError("could not allocate the requested stratified validation size")

    rng = random.Random(seed)
    training: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for label in sorted(grouped):
        group = list(grouped[label])
        rng.shuffle(group)
        validation.extend(group[: quotas[label]])
        training.extend(group[quotas[label] :])
    return training, validation


def _output_row(row: dict[str, str], split: str, source_split: str) -> dict[str, str]:
    label = int(row["label_id"])
    event_path = Path(row["event_relpath"])
    npz_relpath = (
        Path("processed") / source_split / f"{label:03d}" / f"{event_path.stem}.npz"
    )
    result = {field: row.get(field, "") for field in OUTPUT_FIELDS}
    result.update(
        split=split,
        source_split=source_split,
        npz_relpath=npz_relpath.as_posix(),
    )
    return result


def build_manifest_rows(
    dataset_root: Path, validation_fraction: float, seed: int
) -> list[dict[str, str]]:
    train_rows = _read_csv(dataset_root / "splits" / "train.csv")
    test_rows = _read_csv(dataset_root / "splits" / "test.csv")
    training, validation = stratified_train_validation_split(
        train_rows, validation_fraction, seed
    )
    output = [
        *(_output_row(row, "train", "train") for row in training),
        *(_output_row(row, "val", "train") for row in validation),
        *(_output_row(row, "test", "test") for row in test_rows),
    ]

    split_paths: dict[str, set[str]] = defaultdict(set)
    for row in output:
        split_paths[row["split"]].add(row["event_relpath"])
        npz_path = dataset_root / row["npz_relpath"]
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_paths[first] & split_paths[second]
        if overlap:
            raise ValueError(f"{first}/{second} overlap contains {len(overlap)} samples")
    return output


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("gcstr_splits_seed0.csv"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_manifest_rows(args.dataset_root, args.validation_fraction, args.seed)
    write_manifest(args.output, rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    print(f"wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()

