"""Create a reproducible provisional subject-disjoint dataset manifest."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def allocate_people(
    people: list[int], train_ratio: float, val_ratio: float, seed: int
) -> dict[int, str]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train and validation ratios must be between zero and one")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than one")
    shuffled = sorted(people)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_count = round(count * train_ratio)
    val_count = round(count * val_ratio)
    train_people = set(shuffled[:train_count])
    val_people = set(shuffled[train_count:train_count + val_count])
    test_people = set(shuffled[train_count + val_count:])
    if not train_people or not val_people or not test_people:
        raise ValueError("the requested ratios produce an empty split")
    return {
        person: (
            "train" if person in train_people else "val" if person in val_people else "test"
        )
        for person in shuffled
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset_splits.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=124)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if "person_id" not in fieldnames or "class_index" not in fieldnames:
        parser.error("input manifest must contain person_id and class_index")
    people = sorted({int(row["person_id"]) for row in rows})
    assignment = allocate_people(people, args.train_ratio, args.val_ratio, args.seed)

    output_fields = fieldnames if "split" in fieldnames else fieldnames + ["split"]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            row["split"] = assignment[int(row["person_id"])]
            writer.writerow(row)
    temporary.replace(args.output)

    split_people: dict[str, set[int]] = defaultdict(set)
    split_samples = Counter()
    class_coverage: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        split = row["split"]
        split_people[split].add(int(row["person_id"]))
        split_samples[split] += 1
        class_coverage[split].add(int(row["class_index"]))
    print("protocol: provisional subject-disjoint")
    print("seed:", args.seed)
    for split in ("train", "val", "test"):
        print(
            f"{split}: people={sorted(split_people[split])} "
            f"samples={split_samples[split]} classes={len(class_coverage[split])}"
        )
    print("output:", args.output)


if __name__ == "__main__":
    main()
