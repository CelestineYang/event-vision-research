"""Code-faithful gCSTR image representations for event streams.

The implementation follows the public ``newflw/gcstr`` repository for the
original views and the XT- and TY-plane variants:

* ``cstr3``: negative mean timestamp, event density, positive mean timestamp
* ``cstr2``: negative mean timestamp, zero, positive mean timestamp
* ``xt_p``: positive/negative event-count histograms in the x-t plane + zero
* ``xt_p_b``: binary event count + positive/negative mean y (XT_p^b)
* ``xt_b_p``: positive/negative event count + binary mean y (XT_b^p)
* ``xt_mean_p``: positive/negative mean y + zero (XT^p)
* ``ty_p``: positive/negative event-count histograms in the t-y plane + zero
* ``ty_p_b``: binary event count + positive/negative mean x (YT_p^b)
* ``ty_b_p``: positive/negative event count + binary mean x (YT_b^p)
* ``ty_mean_p``: positive/negative mean x + zero (YT^p)

Arrays intentionally follow the official code's axis convention (x-y, x-t,
t-y) before being resized to a conventional ``[C, H, W]`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import zoom


VIEW_NAMES = (
    "cstr3", "cstr2", "xt_p", "xt_p_b", "xt_b_p", "xt_mean_p",
    "ty_p", "ty_p_b", "ty_b_p", "ty_mean_p",
)


@dataclass(frozen=True)
class GCSTRConfig:
    sensor_width: int = 320
    sensor_height: int = 240
    image_size: int = 224
    time_bins: int = 224

    def __post_init__(self) -> None:
        for name, value in (
            ("sensor_width", self.sensor_width),
            ("sensor_height", self.sensor_height),
            ("image_size", self.image_size),
            ("time_bins", self.time_bins),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


def _normalize(array: np.ndarray) -> np.ndarray:
    """Apply the sample-wise min-max normalization used by the public code."""
    minimum = float(array.min(initial=0.0))
    maximum = float(array.max(initial=0.0))
    span = maximum - minimum
    if span == 0.0:
        span = 1.0
    return (array - minimum) / span


def _validate_events(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    config: GCSTRConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value) for value in (t, x, y, p))
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("t, x, y, and p must be one-dimensional")
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("t, x, y, and p must have equal lengths")
    if len(t) == 0:
        raise ValueError("gCSTR requires at least one event")

    t_array, x_array, y_array, p_array = arrays
    if len(t_array) > 1 and np.any(t_array[1:] < t_array[:-1]):
        raise ValueError("event timestamps must be monotonic non-decreasing")
    if np.any(x_array >= config.sensor_width) or np.any(y_array >= config.sensor_height):
        raise ValueError("event coordinates exceed the configured sensor dimensions")
    if np.any(x_array < 0) or np.any(y_array < 0):
        raise ValueError("event coordinates must be non-negative")

    return (
        t_array,
        x_array.astype(np.int64, copy=False),
        y_array.astype(np.int64, copy=False),
        p_array.astype(np.bool_, copy=False),
    )


def _mean_timestamp_xy(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Match ``get_mean_timestamps`` from the public implementation."""
    if len(t) <= 1:
        return np.full((width, height), 0.5, dtype=np.float64)

    minimum = t[0]
    duration = int(t[-1] - minimum)
    if duration == 0:
        duration = 1
    normalized_t = (t - minimum).astype(np.float64) / duration

    sums = np.zeros((width, height), dtype=np.float64)
    counts = np.full((width, height), 0.01, dtype=np.float64)
    np.add.at(sums, (x, y), normalized_t)
    np.add.at(counts, (x, y), 1.0)
    means = sums / counts
    maximum = float(means.max(initial=0.0))
    if maximum == 0.0:
        return means
    return means / maximum


def _mean_y_xt(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    time_bins: int,
) -> np.ndarray:
    """Match the public code's XT mean-y bins and per-component normalization."""
    if len(t) <= 1:
        return np.full((width, time_bins), 0.5, dtype=np.float64)

    duration = float(t[-1] - t[0])
    if duration == 0.0:
        duration = 1.0
    bin_indices = np.floor(
        (t - t[0]).astype(np.float64) * (time_bins / duration)
    ).astype(np.int64)
    np.clip(bin_indices, 0, time_bins - 1, out=bin_indices)
    flat_indices = x * time_bins + bin_indices
    shape = (width, time_bins)
    sums = np.bincount(
        flat_indices, weights=y.astype(np.float64), minlength=width * time_bins
    ).reshape(shape)
    counts = np.bincount(flat_indices, minlength=width * time_bins).reshape(shape)
    means = sums / (counts + 0.01)
    maximum = float(means.max(initial=0.0))
    return means / maximum if maximum > 0.0 else means


def _mean_x_ty(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    time_bins: int,
) -> np.ndarray:
    """Match the public code's TY mean-x bins and per-component normalization."""
    if len(t) <= 1:
        return np.full((time_bins, height), 0.5, dtype=np.float64)

    duration = float(t[-1] - t[0])
    if duration == 0.0:
        duration = 1.0
    bin_indices = np.floor(
        (t - t[0]).astype(np.float64) * (time_bins / duration)
    ).astype(np.int64)
    np.clip(bin_indices, 0, time_bins - 1, out=bin_indices)
    flat_indices = bin_indices * height + y
    shape = (time_bins, height)
    sums = np.bincount(
        flat_indices, weights=x.astype(np.float64), minlength=time_bins * height
    ).reshape(shape)
    counts = np.bincount(flat_indices, minlength=time_bins * height).reshape(shape)
    means = sums / (counts + 0.01)
    maximum = float(means.max(initial=0.0))
    return means / maximum if maximum > 0.0 else means


def _histogram2d(
    first: np.ndarray,
    second: np.ndarray,
    bins: tuple[int, int],
    ranges: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    if len(first) == 0:
        return np.zeros(bins, dtype=np.float64)
    return np.histogram2d(first, second, bins=bins, range=ranges)[0]


def _resize_official(array: np.ndarray, image_size: int) -> np.ndarray:
    factors = (1.0, image_size / array.shape[1], image_size / array.shape[2])
    resized = zoom(array, factors, order=1)
    if resized.shape != (array.shape[0], image_size, image_size):
        raise RuntimeError(
            f"unexpected resized shape {resized.shape}; expected "
            f"{(array.shape[0], image_size, image_size)}"
        )
    return resized.astype(np.float32, copy=False)


def build_gcstr_views(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    views: tuple[str, ...] = VIEW_NAMES,
    config: GCSTRConfig = GCSTRConfig(),
) -> Mapping[str, np.ndarray]:
    """Generate selected gCSTR views as float32 ``[3, S, S]`` arrays."""
    unknown = set(views) - set(VIEW_NAMES)
    if unknown:
        raise ValueError(f"unknown gCSTR views: {sorted(unknown)}")
    if len(set(views)) != len(views):
        raise ValueError("views must not contain duplicates")

    t, x, y, p = _validate_events(t, x, y, p, config)
    positive = p
    negative = ~p
    width, height = config.sensor_width, config.sensor_height
    minimum_t = float(t[0])
    maximum_t = float(t[-1])
    if maximum_t == minimum_t:
        maximum_t = minimum_t + 1.0

    result: dict[str, np.ndarray] = {}

    if "cstr3" in views or "cstr2" in views:
        positive_mean = _mean_timestamp_xy(
            t[positive], x[positive], y[positive], width, height
        )
        negative_mean = _mean_timestamp_xy(
            t[negative], x[negative], y[negative], width, height
        )
        if "cstr2" in views:
            cstr2 = np.stack(
                (negative_mean, np.zeros_like(negative_mean), positive_mean), axis=0
            )
            result["cstr2"] = _resize_official(cstr2, config.image_size)
        if "cstr3" in views:
            density = _histogram2d(
                x,
                y,
                (width, height),
                ((0.0, float(width - 1)), (0.0, float(height - 1))),
            )
            density = _normalize(density)
            cstr3 = np.stack((negative_mean, density, positive_mean), axis=0)
            result["cstr3"] = _resize_official(cstr3, config.image_size)

    xt_views = {"xt_p", "xt_p_b", "xt_b_p", "xt_mean_p"} & set(views)
    if xt_views:
        ranges = ((0.0, float(width - 1)), (minimum_t, maximum_t))
        zero = np.zeros((1, width, config.time_bins), dtype=np.float64)
        polarized_density = None
        if xt_views & {"xt_p", "xt_b_p"}:
            positive_hist = _histogram2d(
                x[positive], t[positive].astype(np.float64),
                (width, config.time_bins), ranges,
            )
            negative_hist = _histogram2d(
                x[negative], t[negative].astype(np.float64),
                (width, config.time_bins), ranges,
            )
            polarized_density = _normalize(
                np.stack((positive_hist, negative_hist), axis=0)
            )
        if "xt_p" in xt_views:
            xt = np.concatenate((polarized_density, zero), axis=0)
            result["xt_p"] = _resize_official(xt, config.image_size)

        polarized_mean_y = None
        if xt_views & {"xt_p_b", "xt_mean_p"}:
            polarized_mean_y = np.stack((
                _mean_y_xt(
                    t[positive], x[positive], y[positive], width, config.time_bins
                ),
                _mean_y_xt(
                    t[negative], x[negative], y[negative], width, config.time_bins
                ),
            ), axis=0)
        if "xt_p_b" in xt_views:
            binary_hist = _histogram2d(
                x, t.astype(np.float64), (width, config.time_bins), ranges
            )
            # Match the public dimension string: density_bin;mean_y_pol.
            xt = np.concatenate(
                (_normalize(binary_hist)[None], polarized_mean_y), axis=0
            )
            result["xt_p_b"] = _resize_official(xt, config.image_size)
        if "xt_b_p" in xt_views:
            binary_mean_y = _mean_y_xt(t, x, y, width, config.time_bins)
            xt = np.concatenate(
                (polarized_density, binary_mean_y[None]), axis=0
            )
            result["xt_b_p"] = _resize_official(xt, config.image_size)
        if "xt_mean_p" in xt_views:
            xt = np.concatenate((polarized_mean_y, zero), axis=0)
            result["xt_mean_p"] = _resize_official(xt, config.image_size)

    ty_views = {"ty_p", "ty_p_b", "ty_b_p", "ty_mean_p"} & set(views)
    if ty_views:
        ranges = ((minimum_t, maximum_t), (0.0, float(height - 1)))
        zero = np.zeros((1, config.time_bins, height), dtype=np.float64)
        polarized_density = None
        if ty_views & {"ty_p", "ty_b_p"}:
            positive_hist = _histogram2d(
                t[positive].astype(np.float64), y[positive],
                (config.time_bins, height), ranges,
            )
            negative_hist = _histogram2d(
                t[negative].astype(np.float64), y[negative],
                (config.time_bins, height), ranges,
            )
            polarized_density = _normalize(
                np.stack((positive_hist, negative_hist), axis=0)
            )
        if "ty_p" in ty_views:
            ty = np.concatenate((polarized_density, zero), axis=0)
            result["ty_p"] = _resize_official(ty, config.image_size)

        polarized_mean_x = None
        if ty_views & {"ty_p_b", "ty_mean_p"}:
            polarized_mean_x = np.stack((
                _mean_x_ty(
                    t[positive], x[positive], y[positive], height, config.time_bins
                ),
                _mean_x_ty(
                    t[negative], x[negative], y[negative], height, config.time_bins
                ),
            ), axis=0)
        if "ty_p_b" in ty_views:
            binary_hist = _histogram2d(
                t.astype(np.float64), y, (config.time_bins, height), ranges
            )
            # Paper YT_p^b: polarized mean x and binary density.
            ty = np.concatenate(
                (_normalize(binary_hist)[None], polarized_mean_x), axis=0
            )
            result["ty_p_b"] = _resize_official(ty, config.image_size)
        if "ty_b_p" in ty_views:
            binary_mean_x = _mean_x_ty(t, x, y, height, config.time_bins)
            ty = np.concatenate(
                (polarized_density, binary_mean_x[None]), axis=0
            )
            result["ty_b_p"] = _resize_official(ty, config.image_size)
        if "ty_mean_p" in ty_views:
            ty = np.concatenate((polarized_mean_x, zero), axis=0)
            result["ty_mean_p"] = _resize_official(ty, config.image_size)

    return result
