"""Offline full-event preprocessing for the explicitly documented paper recipe.

DailyDVS-200 is an adaptation, not the paper's 12-class Daily DVS benchmark.
See EVENTMAMBA_PAPER_REPRODUCTION.md for missing upstream functions and repairs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
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

from eventmamba_v2_upstream.dataprocess.uti import random_crop

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/eventmamba_paper/dailydvs200.json"
DEFAULT_ROOT = Path(os.environ.get("DAILYDVS200_ROOT", "LOCAL_PATH/DailyDvs-200"))
DEFAULT_AUDIT = Path(os.environ.get("DAILYDVS200_AUDIT_ROOT", "LOCAL_PATH/prepared_v1"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config != json.loads(DEFAULT_CONFIG.read_text()):
        raise ValueError("This entry enforces the reviewed recipe; use a separately versioned protocol for changes")
    return config


def read_rows(root, audit_root):
    """Join official NPZ manifests to already audited content hashes, not old points."""
    root, audit_root = Path(root), Path(audit_root)
    result = {}
    for split in ("train", "test"):
        with (root / "splits" / f"{split}_npz.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        with (root / "splits" / f"{split}.csv").open() as handle:
            originals = list(csv.DictReader(handle))
        fields = ("sample_id", "event_relpath", "label_id", "participant_id")
        if [[r[k] for k in fields] for r in rows] != [[r[k] for k in fields] for r in originals]:
            raise ValueError("NPZ manifest differs from original split")
        audit = {r["sample_id"]: r for r in map(json.loads, (audit_root / f"{split}_sources.jsonl").read_text().splitlines())}
        if {int(r["label_id"]) for r in rows} != set(range(200)):
            raise ValueError("Expected all 200 classes")
        for key in ("sample_id", "event_relpath", "npz_relpath"):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate {key}")
        for row in rows:
            prior = audit[row["sample_id"]]
            path = (root / row["npz_relpath"]).resolve()
            if row["split"] != split or path != Path(prior["path"]).resolve():
                raise ValueError("Source/split mismatch")
            if (int(row["sensor_width"]), int(row["sensor_height"])) != (320, 240):
                raise ValueError("Wrong sensor")
            row["sha256"] = prior["sha256"]
        result[split] = rows
    for key in ("sample_id", "npz_relpath", "event_relpath", "participant_id"):
        if {r[key] for r in result["train"]} & {r[key] for r in result["test"]}:
            raise ValueError(f"Train/Test overlap in {key}")
    return result


def window_bounds(t, spec):
    """Full [start, end) windows. No fabricated short-video fallback."""
    if not len(t) or np.any(t[1:] < t[:-1]):
        raise ValueError("Expected nonempty chronological events")
    relative = t.astype(np.int64) - int(t[0])
    starts = np.arange(0, int(relative[-1]) - spec["window_us"],
                       spec["stride_us"], dtype=np.int64)
    return [(int(s), int(a), int(b)) for s, a, b in zip(
        starts, np.searchsorted(relative, starts, side="left"),
        np.searchsorted(relative, starts + spec["window_us"], side="left"))]


@contextmanager
def crop_rng(seed):
    # Official crop consumes Python and CPU Torch RNG. No global RNG side effects.
    py, pt = random.getstate(), torch.get_rng_state()
    try:
        random.seed(seed)
        torch.random.default_generator.manual_seed(seed % (2**63 - 1))
        yield
    finally:
        random.setstate(py)
        torch.set_rng_state(pt)


def normalize_sample(events, seed, spec, *, original=False):
    if not len(events):
        raise ValueError("Empty window cannot be sampled")
    rng = np.random.RandomState(seed % 2**32)
    count = spec["points"]
    if len(events) < count:
        indices = rng.choice(len(events), count, replace=True)
    else:
        indices = np.arange(len(events))
        rng.shuffle(indices)
        indices = indices[:count]
    indices.sort()
    selected = events[indices]
    # Subtract integers before converting large event timestamps to float.
    time = selected[:, 0].astype(np.int64)
    if original:
        time = time // 1000
    time = time - time.min()
    result = selected.astype(np.float64)
    denominator = float(time.max()) + (0.0 if original else 1e-6)
    result[:, 0] = time / max(denominator, 1e-6)
    result[:, 1] /= spec["width"]
    result[:, 2] /= spec["height"]
    result = result.astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.diff(result[:, 0]) < 0) or result.min() < 0 or result.max() > 1:
        raise ValueError("Invalid normalized points")
    return result


def augmented_windows(window, seed, spec):
    with crop_rng(seed):
        crop = random_crop(window.copy(), spec["width"], spec["height"]) if np.ptp(window[:, 0]) > 1 else window.copy()
    yield 1, crop
    reverse = window[::-1].copy()
    reverse[:, 0] = window[0, 0] + window[-1, 0] - reverse[:, 0]
    yield 2, reverse
    vertical = window.copy()
    vertical[:, 2] = spec["height"] - 1 - vertical[:, 2]
    yield 3, vertical
    horizontal = window.copy()
    horizontal[:, 1] = spec["width"] - 1 - horizontal[:, 1]
    yield 4, horizontal


def prepare_sample(task):
    root, row, training, seed, spec = task
    torch.set_num_threads(1)
    if training and row["split"] != "train":
        raise ValueError("Test recording passed as training input")
    path = Path(root) / row["npz_relpath"]
    if sha(path) != row["sha256"]:
        raise ValueError(f"Source changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        t, x, y = (data[k] for k in ("t", "x", "y"))
    if not len(t) or not len(t) == len(x) == len(y) or len(t) != int(row["event_count"]):
        raise ValueError("Source length mismatch")
    if int(t[0]) != int(row["start_time_usec"]) or int(t[-1]) != int(row["end_time_usec"]):
        raise ValueError("Timestamp metadata mismatch")
    if np.any(x >= spec["width"]) or np.any(y >= spec["height"]) or np.any(x < 0) or np.any(y < 0):
        raise ValueError("Invalid sensor coordinates")
    events = np.stack((t.astype(np.int64) - int(t[0]), x, y), axis=1)
    bounds = window_bounds(events[:, 0], spec)
    arrays, metadata = [], []
    stats = {"candidate_windows": len(bounds), "sparse_originals_skipped": 0,
             "empty_windows_skipped": 0, "empty_crops_skipped": 0, "zero_window_recordings": 0}
    for wi, (start, left, right) in enumerate(bounds):
        window = events[left:right]
        if not len(window):
            stats["empty_windows_skipped"] += 1
            continue
        key = f"{seed}:{row['sample_id']}:{wi}"
        base_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        variants = []
        if len(window) > spec["original_min_events_exclusive"]:
            variants.append((0, window))
        else:
            stats["sparse_originals_skipped"] += 1
        if training:
            variants.extend(augmented_windows(window, base_seed, spec))
        for variant, population in variants:
            if not len(population):
                stats["empty_crops_skipped"] += 1
                continue
            arrays.append(normalize_sample(population, base_seed + variant, spec, original=variant == 0))
            metadata.append((wi, variant, start, left, right, len(population)))
    if not arrays:
        stats["zero_window_recordings"] = 1
    points = np.stack(arrays) if arrays else np.empty((0, spec["points"], 3), np.float32)
    return points, np.asarray(metadata, dtype=np.int64).reshape(-1, 6), stats


def cache_signature(root, rows, training, config, scope):
    names = ["eventmamba_paper_data.py", "eventmamba_v2_upstream/dataprocess/uti.py"]
    return {"protocol": config["protocol"], "data": config["data"], "scope": scope,
            "root": str(Path(root).resolve()), "training": training, "seed": config["seed"],
            "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "sources": {name: sha(HERE / name) for name in names},
            "numpy": np.__version__, "torch": torch.__version__}


def build_cache(path, root, rows, *, training, config, scope="full", workers=2, emit=print):
    """Bounded preparation with per-recording commits and SHA-verified sources."""
    if not rows or workers < 1 or scope not in ("full", "smoke"):
        raise ValueError("Invalid cache request")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(root, rows, training, config, scope)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            with h5py.File(path, "r") as h:
                if json.loads(h.attrs["signature"]) != signature or not h.attrs.get("complete"):
                    raise ValueError("Existing cache protocol mismatch")
                return len(h["points"])
        partial = path.with_suffix(".partial.h5")
        with h5py.File(partial, "a") as h:
            columns = ("labels", "sample_index", "window_index", "variant", "start_us", "left", "right", "population")
            if "signature" not in h.attrs:
                h.attrs["signature"] = json.dumps(signature, sort_keys=True)
                h.attrs["commit"] = json.dumps({"recordings": 0, "windows": 0, "stats": {}})
                h.create_dataset("points", (0, 2048, 3), maxshape=(None, 2048, 3), chunks=(1, 2048, 3), dtype="f4")
                for name in columns:
                    h.create_dataset(name, (0,), maxshape=(None,), chunks=(4096,), dtype="i8")
                h.create_dataset("sample_ids", data=np.asarray([r["sample_id"] for r in rows], dtype=h5py.string_dtype()))
                h.create_dataset("sample_labels", data=[int(r["label_id"]) for r in rows])
                h.create_dataset("sample_window_counts", data=np.zeros(len(rows), dtype=np.int64))
            if json.loads(h.attrs["signature"]) != signature:
                raise ValueError("Partial cache protocol mismatch")
            committed = json.loads(h.attrs["commit"])
            done, count, stats = committed["recordings"], committed["windows"], committed["stats"]
            for name in ("points", *columns):
                h[name].resize(count, axis=0)
            # Small batches limit queued/decompressed recordings and returned arrays.
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                for first in range(done, len(rows), workers):
                    tasks = [(str(root), row, training, config["seed"], config["data"]) for row in rows[first:first + workers]]
                    for offset, (points, meta, item_stats) in enumerate(pool.map(prepare_sample, tasks)):
                        index = first + offset
                        size = len(points)
                        values = {"points": points, "labels": np.full(size, int(rows[index]["label_id"])),
                                  "sample_index": np.full(size, index)}
                        values.update({name: meta[:, i] for i, name in enumerate(columns[2:])})
                        for name, value in values.items():
                            h[name].resize(count + size, axis=0)
                            h[name][count:count + size] = value
                        h["sample_window_counts"][index] = size
                        count += size
                        for key, value in item_stats.items():
                            stats[key] = stats.get(key, 0) + value
                        h.flush()
                        h.attrs["commit"] = json.dumps({"recordings": index + 1, "windows": count, "stats": stats})
                        h.flush()
                    emit({"event": "cache", "training": training, "recordings": first + len(tasks),
                          "total_recordings": len(rows), "windows": count, "stats": stats})
            h.attrs["complete"] = True
            h.flush()
        os.replace(partial, path)
    return count


class PaperPoints(Dataset):
    def __init__(self, path, *, exclude_empty=False):
        self.path, self.handle = str(path), None
        with h5py.File(path, "r") as h:
            if not h.attrs.get("complete"):
                raise ValueError("Incomplete cache")
            self.signature = json.loads(h.attrs["signature"])
            self.size = len(h["points"])
            all_ids = h["sample_ids"].asstr()[:].tolist()
            all_labels = h["sample_labels"][:]
            all_counts = h["sample_window_counts"][:]
            missing = np.flatnonzero(all_counts == 0)
            if not exclude_empty and len(missing):
                raise ValueError(f"{len(missing)} recordings have no eligible window; explicitly select exclusion and report coverage")
            active = np.flatnonzero(all_counts > 0)
            self.original_recordings = len(all_ids)
            self.excluded_ids = [all_ids[i] for i in missing]
            self.sample_ids = [all_ids[i] for i in active]
            self.sample_labels = all_labels[active]
            self.sample_counts = all_counts[active]
            self.index_map = np.full(len(all_ids), -1, dtype=np.int64)
            self.index_map[active] = np.arange(len(active))

    def __len__(self):
        return self.size

    def __getstate__(self):
        return dict(self.__dict__, handle=None)

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        h = self.handle
        points = h["points"][index]
        if not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) < 0):
            raise ValueError("Invalid cached points")
        sample_index = int(self.index_map[h["sample_index"][index]])
        if sample_index < 0 or int(h["labels"][index]) != self.sample_labels[sample_index]:
            raise ValueError("Cached sample/label mapping mismatch")
        return {"events": torch.from_numpy(points.T.copy()), "label": int(h["labels"][index]),
                "sample_index": sample_index, "window_index": int(index)}

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_rows(args.root, args.audit_root)
    if args.smoke and "smoke" not in args.output.name:
        parser.error("Smoke requires a separate directory containing 'smoke'")
    selected = {split: rows["train"][:8] if args.smoke else rows[split] for split in rows}
    scope = "smoke" if args.smoke else "full"
    # Test in smoke is an unaugmented Train subset, never original Test data.
    emit = lambda value: print(json.dumps(value), flush=True)
    for split in ("train", "test"):
        path = args.output / f"{split}.h5"
        build_cache(path, args.root, selected[split], training=split == "train",
                    config=config, scope=scope, workers=args.workers, emit=emit)
    datasets = {split: PaperPoints(args.output / f"{split}.h5", exclude_empty=True) for split in selected}
    coverage = {}
    for split, data in datasets.items():
        excluded = set(data.excluded_ids)
        coverage[split] = {"original_recordings": data.original_recordings,
                          "evaluated_recordings": len(data.sample_ids), "windows": len(data),
                          "excluded_recordings": [
            {"sample_id": row["sample_id"], "label": int(row["label_id"]),
             "duration_us": int(row["duration_usec"]), "events": int(row["event_count"]),
             "reason": "no_full_window_end_strictly_before_last_event" if int(row["duration_usec"]) <= config["data"]["window_us"]
                       else "no_eligible_nonempty_window_or_original_event_threshold"}
            for row in selected[split] if row["sample_id"] in excluded]}
    write_json(args.output / "coverage.json", coverage)
    # Explicit user-selected filtering; votes use precisely the retained denominator.
    if any(not len(data) for data in datasets.values()):
        raise ValueError("Filtering removed every window")
    write_json(args.output / "COMPLETE.json", {"scope": scope, "config": config,
               "eval_split": "train_smoke" if args.smoke else "test",
               "coverage": coverage,
               "hashes": {split: sha(args.output / f"{split}.h5") for split in selected}})


if __name__ == "__main__":
    main()
