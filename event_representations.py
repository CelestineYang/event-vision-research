"""Build aligned event-frame, voxel-grid, and point-set representations."""

from __future__ import annotations

from typing import Any

import numpy as np


OFF_CHANNEL = 0
ON_CHANNEL = 1


def _validate_events(events: np.ndarray, height: int, width: int) -> None:
    required_fields = {"t", "x", "y", "on"}
    fields = set(events.dtype.fields or {})
    missing = required_fields - fields
    if missing:
        raise ValueError(f"event array is missing fields: {sorted(missing)}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if len(events) == 0:
        return
    if np.any(events["x"] >= width) or np.any(events["y"] >= height):
        raise ValueError("event coordinates fall outside the sensor dimensions")
    if len(events) > 1 and np.any(events["t"][1:] < events["t"][:-1]):
        raise ValueError("event timestamps must be monotonic non-decreasing")


def _normalized_time(events: np.ndarray) -> tuple[np.ndarray, int]:
    """Subtract uint64 timestamps before conversion to floating point."""
    if len(events) == 0:
        return np.empty(0, dtype=np.float64), 0
    relative_t = events["t"] - events["t"][0]
    duration_us = int(relative_t[-1])
    if duration_us == 0:
        return np.zeros(len(events), dtype=np.float64), 0
    return relative_t.astype(np.float64) / duration_us, duration_us


def _time_bins(normalized_t: np.ndarray, num_bins: int) -> np.ndarray:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    bins = np.floor(normalized_t * num_bins).astype(np.int64)
    return np.minimum(bins, num_bins - 1)


def _build_voxel(
    events: np.ndarray,
    bins: np.ndarray,
    height: int,
    width: int,
    num_bins: int,
) -> np.ndarray:
    """Count events into [polarity, time, y, x] with OFF before ON."""
    if len(events) == 0:
        return np.zeros((2, num_bins, height, width), dtype=np.int32)
    polarity = events["on"].astype(np.int64)
    flat_indices = (
        ((polarity * num_bins + bins) * height + events["y"].astype(np.int64))
        * width
        + events["x"].astype(np.int64)
    )
    counts = np.bincount(
        flat_indices, minlength=2 * num_bins * height * width
    ).reshape(2, num_bins, height, width)
    if counts.max(initial=0) > np.iinfo(np.int32).max:
        raise OverflowError("a voxel count exceeds int32 capacity")
    return counts.astype(np.int32, copy=False)


def _sample_points(
    events: np.ndarray,
    normalized_t: np.ndarray,
    bins: np.ndarray,
    height: int,
    width: int,
    num_bins: int,
    points_per_bin: int,
    deterministic: bool,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_per_bin <= 0:
        raise ValueError("points_per_bin must be positive")
    capacity = num_bins * points_per_bin
    points = np.zeros((capacity, 4), dtype=np.float32)
    point_mask = np.zeros(capacity, dtype=np.bool_)
    point_bins = np.full(capacity, -1, dtype=np.int64)
    if len(events) == 0:
        return points, point_mask, point_bins

    rng = np.random.default_rng(seed)
    selected_parts = []
    selected_bins = []
    for bin_index in range(num_bins):
        candidates = np.flatnonzero(bins == bin_index)
        sample_count = min(points_per_bin, len(candidates))
        if sample_count == 0:
            continue
        if sample_count == len(candidates):
            selected = candidates
        elif deterministic:
            positions = np.linspace(0, len(candidates) - 1, sample_count, dtype=np.int64)
            selected = candidates[positions]
        else:
            selected = np.sort(rng.choice(candidates, sample_count, replace=False))
        selected_parts.append(selected)
        selected_bins.append(np.full(sample_count, bin_index, dtype=np.int64))

    if not selected_parts:
        return points, point_mask, point_bins
    selected = np.concatenate(selected_parts)
    sampled_bins = np.concatenate(selected_bins)
    count = len(selected)
    xs = events["x"][selected].astype(np.float32)
    ys = events["y"][selected].astype(np.float32)
    points[:count, 0] = 2.0 * xs / max(width - 1, 1) - 1.0
    points[:count, 1] = 2.0 * ys / max(height - 1, 1) - 1.0
    points[:count, 2] = normalized_t[selected].astype(np.float32)
    points[:count, 3] = np.where(events["on"][selected], 1.0, -1.0)
    point_mask[:count] = True
    point_bins[:count] = sampled_bins
    return points, point_mask, point_bins


def build_representations(
    events: np.ndarray,
    *,
    height: int = 240,
    width: int = 320,
    num_bins: int = 8,
    points_per_bin: int = 256,
    deterministic: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate three aligned representations from one complete event window.

    Raw frame and voxel outputs contain integer counts. Apply transforms such
    as ``log1p`` in the dataset or model, so count-conservation tests remain
    exact. Point channels are normalized x, normalized y, normalized time,
    and polarity in {-1, +1}.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    _validate_events(events, height, width)
    normalized_t, duration_us = _normalized_time(events)
    bins = _time_bins(normalized_t, num_bins)
    voxel = _build_voxel(events, bins, height, width, num_bins)
    frame = voxel.sum(axis=1, dtype=np.int64)
    if frame.max(initial=0) > np.iinfo(np.int32).max:
        raise OverflowError("a frame count exceeds int32 capacity")
    frame = frame.astype(np.int32, copy=False)
    points, point_mask, point_bins = _sample_points(
        events,
        normalized_t,
        bins,
        height,
        width,
        num_bins,
        points_per_bin,
        deterministic,
        seed,
    )
    return {
        "frame": frame,
        "voxel": voxel,
        "points": points,
        "point_mask": point_mask,
        "point_bin": point_bins,
        "duration_us": duration_us,
        "event_count": len(events),
    }
