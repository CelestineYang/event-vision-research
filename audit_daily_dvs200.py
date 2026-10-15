"""Build a structural inventory and stream-safe audit of DailyDVS-200."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from daily_dvs200_loader import inspect_event_file


DEFAULT_ROOT = Path(
    "/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200"
)
FILENAME_PATTERN = re.compile(
    r"^C(?P<class_index>\d+)P(?P<person_id>\d+)M(?P<motion_id>\d+)"
    r"S(?P<session_id>\d+)_(?P<recorded_at>\d{8}_\d{2}_\d{2}_\d{2})\.aedat4$"
)
ACTION_PATTERN = re.compile(r"^action_(?P<action_id>\d+)$")

STRUCTURAL_FIELDS = [
    "path", "action_name", "action_id", "class_index", "class_consistent",
    "person_id", "motion_id", "session_id", "recorded_at", "size_bytes",
]
AUDIT_FIELDS = [
    "decode_status", "error", "width", "height", "packet_count", "event_count",
    "t_start", "t_end", "duration_us", "x_min", "x_max", "y_min", "y_max",
    "on_count", "off_count", "timestamps_monotonic", "invalid_coordinate_count",
]
ALL_FIELDS = STRUCTURAL_FIELDS + AUDIT_FIELDS


def parse_sample_path(path: Path, root: Path) -> dict[str, object]:
    """Parse labels and recording identifiers without opening the AEDAT4 file."""
    relative_path = path.relative_to(root)
    action_match = ACTION_PATTERN.fullmatch(path.parent.name)
    filename_match = FILENAME_PATTERN.fullmatch(path.name)
    if action_match is None:
        raise ValueError(f"invalid action directory: {path.parent.name}")
    if filename_match is None:
        raise ValueError(f"invalid AEDAT4 filename: {path.name}")

    action_id = int(action_match.group("action_id"))
    values = filename_match.groupdict()
    class_index = int(values["class_index"])
    return {
        "path": relative_path.as_posix(),
        "action_name": path.parent.name,
        "action_id": action_id,
        "class_index": class_index,
        "class_consistent": class_index + 1 == action_id,
        "person_id": int(values["person_id"]),
        "motion_id": int(values["motion_id"]),
        "session_id": int(values["session_id"]),
        "recorded_at": values["recorded_at"],
        "size_bytes": path.stat().st_size,
    }


def discover_files(root: Path) -> list[Path]:
    return sorted(root.glob("action_*/*.aedat4"))


def select_files(
    paths: Iterable[Path], per_class: int | None, max_files: int | None, seed: int
) -> list[Path]:
    """Select a deterministic, class-balanced audit subset."""
    paths = list(paths)
    if per_class is not None:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            grouped[path.parent.name].append(path)
        rng = random.Random(seed)
        selected = []
        for action_name in sorted(grouped):
            candidates = grouped[action_name]
            selected.extend(rng.sample(candidates, min(per_class, len(candidates))))
        paths = sorted(selected)
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def read_completed_paths(output: Path) -> set[str]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    with output.open(newline="", encoding="utf-8") as handle:
        return {row["path"] for row in csv.DictReader(handle)}


def write_inventory(root: Path, output: Path, paths: Iterable[Path]) -> None:
    """Write the inexpensive structural manifest atomically."""
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURAL_FIELDS)
        writer.writeheader()
        for path in paths:
            writer.writerow(parse_sample_path(path, root))
    temporary_output.replace(output)


def audit_files(root: Path, output: Path, paths: Iterable[Path], resume: bool) -> None:
    """Decode selected files and flush one row after every completed file."""
    completed = read_completed_paths(output) if resume else set()
    mode = "a" if resume and output.exists() and output.stat().st_size > 0 else "w"
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALL_FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, path in enumerate(paths, start=1):
            row = parse_sample_path(path, root)
            if row["path"] in completed:
                continue
            try:
                row.update(inspect_event_file(path))
                row["decode_status"] = "ok"
                row["error"] = ""
            except Exception as exc:
                row["decode_status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(row)
            handle.flush()
            print(f"[{index}] {row['path']}: {row['decode_status']}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--inventory", type=Path, default=Path("dataset_inventory.csv"))
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"dataset root does not exist: {args.root}")
    if args.per_class is not None and args.per_class < 1:
        parser.error("--per-class must be at least 1")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be at least 1")

    paths = discover_files(args.root)
    if not paths:
        parser.error(f"no AEDAT4 files found below {args.root}")
    write_inventory(args.root, args.inventory, paths)
    print(f"inventory: {len(paths)} files -> {args.inventory}")

    if args.audit_output is not None:
        selected = select_files(paths, args.per_class, args.max_files, args.seed)
        print(f"audit selection: {len(selected)} files")
        audit_files(args.root, args.audit_output, selected, args.resume)


if __name__ == "__main__":
    main()
