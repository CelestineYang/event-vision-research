#!/usr/bin/env python3
"""Prepare UCF101-DVS from the five ZIP archives and official split files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


NAME_RE = re.compile(
    # Official UCF101 lists use .avi; the DVS archives use the matching .mat.
    r"^v_(?P<class_name>.+)_g(?P<group>\d+)_c(?P<clip>\d+)\.(?:mat|avi)$",
    re.IGNORECASE,
)

FIELDS = [
    "sample_id", "split", "source_archive", "event_relpath", "npz_relpath",
    "label_id", "official_label_id", "class_name", "group", "clip",
    "event_count", "start_time_usec", "end_time_usec", "duration_usec",
    "sensor_width", "sensor_height", "source_size_bytes", "npz_size_bytes",
]


def key_for(path_text: str) -> tuple[str, int, int]:
    path = PurePosixPath(path_text.replace("\\", "/"))
    if len(path.parts) != 2:
        raise ValueError(f"bad path: {path_text}")
    match = NAME_RE.fullmatch(path.parts[1])
    if not match:
        raise ValueError(f"bad UCF101 sample name: {path_text}")
    return (
        path.parts[0].casefold(),
        int(match.group("group")),
        int(match.group("clip")),
    )


def parse_name(path_text: str) -> tuple[str, int, int]:
    path = PurePosixPath(path_text)
    match = NAME_RE.fullmatch(path.parts[1])
    return path.parts[0], int(match.group("group")), int(match.group("clip"))


def official_splits(split_zip: Path, split_id: int):
    with ZipFile(split_zip) as z:
        class_rows = z.read("ucfTrainTestlist/classInd.txt").decode().splitlines()
        class_map = {}
        for line in class_rows:
            number, name = line.split(maxsplit=1)
            class_map[name.casefold()] = (int(number), name)

        def read(name: str, has_label: bool):
            lines = z.read(name).decode().splitlines()
            result = {}
            for line in lines:
                if not line.strip():
                    continue
                path_text = line.rsplit(maxsplit=1)[0] if has_label else line.strip()
                result[key_for(path_text)] = path_text
            return result

        train = read(f"ucfTrainTestlist/trainlist{split_id:02d}.txt", True)
        test = read(f"ucfTrainTestlist/testlist{split_id:02d}.txt", False)
    return class_map, train, test


def scan_archives(archives: list[Path]):
    samples = {}
    for archive in archives:
        with ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".mat"):
                    continue
                relpath = info.filename.replace("\\", "/")
                key = key_for(relpath)
                if key in samples:
                    raise RuntimeError(f"duplicate sample key: {key}")
                samples[key] = {
                    "archive": archive.name,
                    "member": relpath,
                    "size": info.file_size,
                }
    return samples


def convert_one(task):
    import numpy as np
    from scipy.io import loadmat

    archive_path, member, output_path = task
    output_path = Path(output_path)
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as d:
            count = len(d["t"])
            start = int(d["t"][0])
            end = int(d["t"][-1])
        return {"status": "skipped", "event_count": count,
                "start": start, "end": end, "size": output_path.stat().st_size}

    with ZipFile(archive_path) as z:
        raw = z.read(member)
    data = loadmat(BytesIO(raw), squeeze_me=False, struct_as_record=False)
    required = {"ts", "x", "y", "pol"}
    if not required.issubset(data):
        raise ValueError(f"missing fields in {member}: {sorted(data)}")

    t = np.asarray(data["ts"]).reshape(-1).astype(np.int64, copy=False)
    x = np.asarray(data["x"]).reshape(-1).astype(np.int16, copy=False)
    y = np.asarray(data["y"]).reshape(-1).astype(np.int16, copy=False)
    p = np.asarray(data["pol"]).reshape(-1).astype(np.uint8, copy=False)
    if not (len(t) == len(x) == len(y) == len(p)):
        raise ValueError(f"invalid event arrays in {member}")
    if len(t) == 0:
        return {"status": "empty", "event_count": 0,
                "member": member, "size": 0}
    if np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        t, x, y, p = t[order], x[order], y[order], p[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, t=t, x=x, y=y, p=p)
    os.replace(tmp, output_path)
    return {"status": "converted", "event_count": len(t),
            "start": int(t[0]), "end": int(t[-1]),
            "size": output_path.stat().st_size}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    root = args.root.resolve()

    archives = sorted(root.glob("*UCF101-DVS-Part*.zip"))
    if len(archives) != 5:
        raise SystemExit(f"expected 5 UCF ZIPs, found {len(archives)}")
    split_zip = root / "UCF101TrainTestSplits-RecognitionTask.zip"
    if not split_zip.is_file():
        raise SystemExit(f"missing split ZIP: {split_zip}")

    class_map, official_train, official_test = official_splits(split_zip, args.split)
    samples = scan_archives(archives)
    train_keys = set(official_train) & set(samples)
    test_keys = set(official_test) & set(samples)
    missing = (set(official_train) | set(official_test)) - set(samples)
    extra = set(samples) - set(official_train) - set(official_test)
    if train_keys & test_keys:
        raise SystemExit("official train/test overlap after normalization")

    selected = [("train", k) for k in sorted(train_keys)] + [("test", k) for k in sorted(test_keys)]
    if args.limit is not None:
        selected = selected[:args.limit]

    processed = root / "UCF101-DVS" / "processed"
    splits = root / "UCF101-DVS" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    original = splits / "original"
    original.mkdir(exist_ok=True)
    (original / f"trainlist{args.split:02d}.txt").write_text(
        "\n".join(official_train.values()) + "\n", encoding="utf-8"
    )
    (original / f"testlist{args.split:02d}.txt").write_text(
        "\n".join(official_test.values()) + "\n", encoding="utf-8"
    )

    tasks = []
    meta = []
    for split, key in selected:
        item = samples[key]
        class_name, group, clip = parse_name(item["member"])
        official_id, official_name = class_map[class_name.casefold()]
        label_id = official_id - 1
        stem = PurePosixPath(item["member"]).stem
        npz_rel = Path("processed") / split / f"{label_id:03d}" / f"{stem}.npz"
        output = root / "UCF101-DVS" / npz_rel
        archive_path = root / item["archive"]
        tasks.append((str(archive_path), item["member"], str(output)))
        meta.append((split, key, item, npz_rel.as_posix(), label_id, official_id,
                     official_name, group, clip))

    start_time = time.time()
    results = [None] * len(tasks)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, task): i for i, task in enumerate(tasks)}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                failures.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
            if done == 1 or done % 100 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] failed={len(failures)}", flush=True)

    report = {
        "official_split": args.split,
        "official_train": len(official_train),
        "official_test": len(official_test),
        "converted_requested": len(tasks),
        "available_train": len(train_keys),
        "available_test": len(test_keys),
        "missing_official": len(missing),
        "extra_unlisted": len(extra),
        "empty_samples": sum(r is not None and r.get("status") == "empty" for r in results),
        "failures": failures,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    (splits / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"conversion failed: {len(failures)}")

    rows = []
    empty_rows = []
    for (split, key, item, npz_rel, label_id, official_id, official_name, group, clip), result in zip(meta, results):
        if result["status"] == "empty":
            empty_rows.append({
                "split": split,
                "event_relpath": item["member"],
                "source_archive": item["archive"],
                "label_id": label_id,
                "official_label_id": official_id,
                "class_name": official_name,
                "group": group,
                "clip": clip,
            })
            continue
        rows.append({
            "sample_id": f"{split}_{len(rows):05d}",
            "split": split,
            "source_archive": item["archive"],
            "event_relpath": item["member"],
            "npz_relpath": npz_rel,
            "label_id": label_id,
            "official_label_id": official_id,
            "class_name": official_name,
            "group": group,
            "clip": clip,
            "event_count": result["event_count"],
            "start_time_usec": result["start"],
            "end_time_usec": result["end"],
            "duration_usec": result["end"] - result["start"],
            "sensor_width": 240,
            "sensor_height": 180,
            "source_size_bytes": item["size"],
            "npz_size_bytes": result["size"],
        })

    write_csv(splits / "manifest.csv", rows)
    write_csv(splits / "train.csv", [r for r in rows if r["split"] == "train"])
    write_csv(splits / "test.csv", [r for r in rows if r["split"] == "test"])
    write_csv(splits / "class_map.csv", [
        {"label_id": n - 1, "official_label_id": n, "class_name": name}
        for name, (n, _) in sorted(class_map.items(), key=lambda item: item[1][0])
    ], ["label_id", "official_label_id", "class_name"])
    write_csv(splits / "missing_official.csv", [
        {"key": str(k), "official_path": official_train.get(k, official_test.get(k, ""))}
        for k in sorted(missing)
    ], ["key", "official_path"])
    write_csv(splits / "excluded_unlisted.csv", [
        {"key": str(k), "source_archive": samples[k]["archive"], "event_relpath": samples[k]["member"]}
        for k in sorted(extra)
    ], ["key", "source_archive", "event_relpath"])
    write_csv(splits / "empty_samples.csv", empty_rows, [
        "split", "source_archive", "event_relpath", "label_id",
        "official_label_id", "class_name", "group", "clip",
    ])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
