"""Training-only event augmentation based on gCSTR Appendix A."""

from __future__ import annotations

import numpy as np

from gcstr_representations import GCSTRConfig


AUGMENTATIONS = ("none", "spatial", "temporal", "polarity", "spatial_temporal", "all")


def augment_events(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    mode: str,
    rng: np.random.Generator,
    config: GCSTRConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform an event stream while preserving its chronological order."""
    if mode not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation {mode!r}")
    t, x, y, p = (np.asarray(array) for array in (t, x, y, p))
    if mode == "none":
        return t, x, y, p
    if not (len(t) == len(x) == len(y) == len(p)) or len(t) == 0:
        raise ValueError("augmentation requires nonempty, equal-length event arrays")

    if mode in {"spatial", "spatial_temporal", "all"}:
        # The paper applies spatial transforms to uniformly dithered coordinates.
        xf = x.astype(np.float64) + rng.uniform(-0.5, 0.5, len(x))
        yf = y.astype(np.float64) + rng.uniform(-0.5, 0.5, len(y))
        angle = np.deg2rad(rng.uniform(-15.0, 15.0))
        cx, cy = (config.sensor_width - 1) / 2, (config.sensor_height - 1) / 2
        dx, dy = xf - cx, yf - cy
        cosine, sine = np.cos(angle), np.sin(angle)
        xf = dx * cosine - dy * sine + cx
        yf = dx * sine + dy * cosine + cy
        inside = ((xf >= 0) & (xf <= config.sensor_width - 1)
                  & (yf >= 0) & (yf <= config.sensor_height - 1))
        if np.any(inside):
            t, xf, yf, p = t[inside], xf[inside], yf[inside], p[inside]

        left, right, top, bottom = rng.uniform(0.0, 0.1, 4)
        xmin, xmax = left * config.sensor_width, (1 - right) * config.sensor_width
        ymin, ymax = top * config.sensor_height, (1 - bottom) * config.sensor_height
        inside = (xf >= xmin) & (xf < xmax) & (yf >= ymin) & (yf < ymax)
        if np.any(inside):
            t, xf, yf, p = t[inside], xf[inside], yf[inside], p[inside]
        x = np.clip(np.rint((xf - xmin) / (xmax - xmin) * (config.sensor_width - 1)),
                    0, config.sensor_width - 1).astype(np.int64)
        y = np.clip(np.rint((yf - ymin) / (ymax - ymin) * (config.sensor_height - 1)),
                    0, config.sensor_height - 1).astype(np.int64)

    if mode in {"temporal", "spatial_temporal", "all"} and len(t) > 1:
        start, end = float(t[0]), float(t[-1])
        duration = end - start
        if duration > 0:
            lower = start + rng.uniform(0.0, 0.1) * duration
            upper = end - rng.uniform(0.0, 0.1) * duration
            inside = (t >= lower) & (t < upper)
            if np.any(inside):
                t, x, y, p = t[inside], x[inside], y[inside], p[inside]
            cropped_start, cropped_end = float(t[0]), float(t[-1])
            cropped_duration = cropped_end - cropped_start
            if cropped_duration > 0:
                exponent = rng.uniform(0.4, 2.2)
                normalized = (t.astype(np.float64) - cropped_start) / cropped_duration
                t = np.rint(normalized ** exponent * cropped_duration + cropped_start).astype(np.int64)

    if mode in {"polarity", "all"} and rng.random() < 0.5:
        p = np.logical_not(p)
    return t, x, y, p
