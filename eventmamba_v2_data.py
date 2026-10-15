"""DailyDVS-200 window adapter. Protocol versioned; Test prepared only at the end.

Official reference: eventmamba_v2_upstream/dataprocess/generate_daily.py.
Repairs/choices are documented in EVENTMAMBA_V2_ONLY_RUNBOOK.md.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from create_gcstr_splits import DEFAULT_ROOT
from secnet_only_dataset import official_rows, stable_seed
from eventmamba_v2_upstream.dataprocess.uti import random_crop

WINDOW_US = 1500000
STRIDE_US = 500000
POINTS = 2048
WIDTH, HEIGHT = 320, 240
PROTOCOL = "dailydvs200_v2_windows_v1"


def window_bounds(t):
    """Full half-open windows on a 0.5s grid; short recording gets one window.

    Unlike upstream's mismatched/exclusive end indices, these indices implement
    [start,start+1.5s) exactly. An event at the boundary belongs to later windows.
    """
    if len(t) == 0 or np.any(t[1:] < t[:-1]):
        raise ValueError("Events must be nonempty and chronological")
    relative = t.astype(np.int64) - int(t[0])
    duration = int(relative[-1])
    if duration < WINDOW_US:
        return [(0, len(t))]
    starts = np.arange(0, duration - WINDOW_US + 1, STRIDE_US, dtype=np.int64)
    left = np.searchsorted(relative, starts, side="left")
    right = np.searchsorted(relative, starts + WINDOW_US, side="left")
    return [(int(a), int(b)) for a, b in zip(left, right) if b > a]


def normalize_sample(events, seed, *, milliseconds=False):
    if not len(events):
        raise ValueError("Cannot sample an empty window")
    rng = np.random.RandomState(seed % 2**32)
    if len(events) < POINTS:
        indices = rng.choice(len(events), POINTS, replace=True)
    else:
        indices = np.arange(len(events))
        rng.shuffle(indices)
        indices = indices[:POINTS]
    indices.sort()
    points = events[indices].astype(np.float64)
    if milliseconds:
        points[:, 0] = np.floor(points[:, 0] / 1000)
    points[:, 0] -= points[:, 0].min()
    points[:, 0] /= max(float(points[:, 0].max()), 1e-6)
    points[:, 1] /= WIDTH
    points[:, 2] /= HEIGHT
    if not np.isfinite(points).all() or np.any(points[1:, 0] < points[:-1, 0]):
        raise ValueError("Invalid normalized points")
    return points.astype(np.float32)


def prepare_sample(task):
    root, row, training, seed = task
    torch.set_num_threads(1)
    with np.load(Path(root) / row["npz_relpath"], allow_pickle=False) as data:
        t, x, y = (data[k] for k in ("t", "x", "y"))
    if not len(t) or len(t) != len(x) or len(t) != len(y):
        raise ValueError(f"Invalid events: {row['sample_id']}")
    if np.any(x < 0) or np.any(x >= WIDTH) or np.any(y < 0) or np.any(y >= HEIGHT):
        raise ValueError("Sensor coordinates outside 320x240")
    events = np.stack((t.astype(np.int64) - int(t[0]), x, y), -1)
    bounds = window_bounds(events[:, 0])
    arrays, metadata = [], []
    stats = {"candidate_windows": len(bounds), "sparse_base_skipped": 0,
             "empty_crop_skipped": 0, "fallback_samples": 0}
    for wi, (a, b) in enumerate(bounds):
        window = events[a:b]
        base_seed = stable_seed(f"{seed}:{row['sample_id']}:{wi}")
        if len(window) > 10000:
            arrays.append(normalize_sample(window, base_seed, milliseconds=True))
            metadata.append((wi, 0))
        else:
            stats["sparse_base_skipped"] += 1
        if not training:
            continue
        random.seed(base_seed)
        np.random.seed(base_seed % 2**32)
        torch.manual_seed(base_seed % (2**63 - 1))
        # Use official random_crop, with a guard for degenerate time ranges.
        crop = random_crop(window.copy(), WIDTH, HEIGHT) if window[-1, 0] - window[0, 0] > 1 else window.copy()
        reverse = window[::-1].copy()
        reverse[:, 0] = window[0, 0] + window[-1, 0] - reverse[:, 0]
        vertical, horizontal = window.copy(), window.copy()
        vertical[:, 2] = HEIGHT - 1 - vertical[:, 2]
        horizontal[:, 1] = WIDTH - 1 - horizontal[:, 1]
        for variant, augmented in enumerate((crop, reverse, vertical, horizontal), 1):
            if not len(augmented):
                stats["empty_crop_skipped"] += 1
                continue
            arrays.append(normalize_sample(augmented, base_seed + variant))
            metadata.append((wi, variant))
    if not arrays:
        # Do not silently omit an official sample from the evaluation denominator.
        if bounds:
            wi, (a, b) = max(enumerate(bounds), key=lambda item: item[1][1] - item[1][0])
        else:
            wi, a, b = 0, 0, len(events)
        arrays.append(normalize_sample(events[a:b], stable_seed(f"{seed}:{row['sample_id']}:fallback"), milliseconds=True))
        metadata.append((wi, 0))
        stats["fallback_samples"] = 1
    return np.stack(arrays), np.asarray(metadata, dtype=np.int32), stats


def cache_signature(root, rows, training, seed):
    here = Path(__file__).resolve().parent
    files = ("eventmamba_v2_data.py", "secnet_only_dataset.py", "create_gcstr_splits.py",
             "eventmamba_v2_upstream/dataprocess/uti.py")
    return {"protocol": PROTOCOL, "root": str(Path(root).resolve()), "training": training,
            "seed": seed, "point_count": POINTS, "window_us": WINDOW_US, "stride_us": STRIDE_US,
            "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "source_hashes": {p: hashlib.sha256((here / p).read_bytes()).hexdigest() for p in files}}


def build_cache(path, root, rows, *, training, seed=0, workers=4, emit=print):
    """One bounded-size HDF5 of fixed sampled windows, like official offline data.

    Append checkpoints are per 32 recordings. Interrupted cache builds resume
    from the last committed recording count, dropping only uncommitted entries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(root, rows, training, seed)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            with h5py.File(path, "r") as f:
                if json.loads(f.attrs["signature"]) != signature or not f.attrs.get("complete", False):
                    raise ValueError("Cache signature/completion mismatch")
                return len(f["points"])
        partial = path.with_suffix(".partial.h5")
        with h5py.File(partial, "a") as f:
            if "signature" not in f.attrs:
                f.attrs["signature"] = json.dumps(signature, sort_keys=True)
                f.attrs["commit"] = json.dumps({"samples": 0, "windows": 0, "stats": {}})
                f.create_dataset("points", shape=(0, POINTS, 3), maxshape=(None, POINTS, 3), chunks=(32, POINTS, 3), dtype="f4")
                for key in ("labels", "sample_index", "window_index", "variant"):
                    f.create_dataset(key, shape=(0,), maxshape=(None,), chunks=(4096,), dtype="i4")
                f.create_dataset("sample_ids", data=np.asarray([r["sample_id"] for r in rows], dtype=h5py.string_dtype()))
                f.create_dataset("sample_labels", data=[int(r["label_id"]) for r in rows])
            if json.loads(f.attrs["signature"]) != signature:
                raise ValueError("Partial cache signature mismatch")
            committed = json.loads(f.attrs["commit"])
            done, count, stats = committed["samples"], committed["windows"], committed["stats"]
            for key in ("points", "labels", "sample_index", "window_index", "variant"):
                f[key].resize(count, axis=0)
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                for first in range(done, len(rows), 32):
                    batch_rows = rows[first:first + 32]
                    tasks = [(str(root), r, training, seed) for r in batch_rows]
                    for offset, (points, meta, item_stats) in enumerate(pool.map(prepare_sample, tasks)):
                        size = len(points)
                        values = {"points": points, "labels": np.full(size, int(batch_rows[offset]["label_id"])),
                                  "sample_index": np.full(size, first + offset), "window_index": meta[:, 0], "variant": meta[:, 1]}
                        for key, value in values.items():
                            f[key].resize(count + size, axis=0)
                            f[key][count:count + size] = value
                        count += size
                        for key, value in item_stats.items():
                            stats[key] = stats.get(key, 0) + value
                    done = first + len(batch_rows)
                    f.flush()
                    f.attrs["commit"] = json.dumps({"samples": done, "windows": count, "stats": stats})
                    f.flush()
                    emit({"event": "cache_progress", "split": "train" if training else "test",
                          "samples": done, "total_samples": len(rows), "windows": count,
                          "point_data_gib": count * POINTS * 3 * 4 / 2**30, "stats": stats})
            f.attrs["complete"] = True
            f.flush()
        os.replace(partial, path)
        return count


class WindowDataset(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.handle = None
        with h5py.File(path, "r") as f:
            if not f.attrs.get("complete", False):
                raise ValueError("Incomplete window cache")
            self.size = len(f["points"])
            self.sample_ids = f["sample_ids"].asstr()[:].tolist()
            self.sample_labels = f["sample_labels"][:].astype(np.int64)
            self.signature = json.loads(f.attrs["signature"])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        f = self.handle
        return {"events": torch.from_numpy(f["points"][index].T.copy()),
                "label": int(f["labels"][index]), "sample_index": int(f["sample_index"][index]),
                "window_index": index}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = official_rows(args.dataset_root)["train"]
    if args.smoke:
        rows = rows[:4]
    build_cache(args.cache_dir / "train.h5", args.dataset_root, rows, training=True,
                workers=args.workers, emit=lambda r: print(json.dumps(r), flush=True))
