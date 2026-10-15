"""Utilities for reading and inspecting DailyDVS-200 AEDAT4 event files.

The three representations used later in the project must all originate from
one event stream. This module keeps event decoding in one place and ignores
IMU and trigger packets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator

import aedat
import numpy as np


EVENT_DTYPE = np.dtype(
    [("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")]
)
DEFAULT_SAMPLE = Path(
    "/remote-home/datasets/Joind-dataset for Action Recognition/"
    "DailyDvs-200/action_001/C0P10M0S2_20231118_09_44_16.aedat4"
)


def iter_event_packets(path: str | Path) -> Iterator[np.ndarray]:
    """Yield non-empty event packets, ignoring IMU and trigger streams."""
    decoder = aedat.Decoder(str(path))
    for packet in decoder:
        events = packet.get("events")
        if events is not None and len(events) > 0:
            yield events


def load_events(path: str | Path) -> np.ndarray:
    """Load all events into memory; use inspect_event_file for bulk audits."""
    event_packets = list(iter_event_packets(path))
    if not event_packets:
        return np.empty(0, dtype=EVENT_DTYPE)
    return np.concatenate(event_packets)


def event_stream_metadata(path: str | Path) -> dict[str, int | None]:
    """Return the event sensor dimensions stored in the AEDAT4 metadata."""
    decoder = aedat.Decoder(str(path))
    streams = [
        stream
        for stream in decoder.id_to_stream().values()
        if stream.get("type") == "events"
    ]
    if not streams:
        return {"width": None, "height": None}
    dimensions = {(stream.get("width"), stream.get("height")) for stream in streams}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent event stream dimensions: {sorted(dimensions)}")
    width, height = dimensions.pop()
    return {"width": width, "height": height}


def inspect_event_file(path: str | Path) -> dict[str, int | bool | None]:
    """Compute event statistics packet by packet with bounded memory usage."""
    metadata = event_stream_metadata(path)
    width, height = metadata["width"], metadata["height"]
    event_count = packet_count = on_count = invalid_coordinate_count = 0
    t_start = t_end = previous_t = None
    x_min = x_max = y_min = y_max = None
    monotonic = True

    for events in iter_event_packets(path):
        packet_count += 1
        event_count += len(events)
        timestamps, xs, ys = events["t"], events["x"], events["y"]
        packet_first_t, packet_last_t = int(timestamps[0]), int(timestamps[-1])
        if t_start is None:
            t_start = packet_first_t
        if previous_t is not None and packet_first_t < previous_t:
            monotonic = False
        if len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
            monotonic = False
        previous_t, t_end = packet_last_t, packet_last_t

        packet_x_min, packet_x_max = int(xs.min()), int(xs.max())
        packet_y_min, packet_y_max = int(ys.min()), int(ys.max())
        x_min = packet_x_min if x_min is None else min(x_min, packet_x_min)
        x_max = packet_x_max if x_max is None else max(x_max, packet_x_max)
        y_min = packet_y_min if y_min is None else min(y_min, packet_y_min)
        y_max = packet_y_max if y_max is None else max(y_max, packet_y_max)
        on_count += int(np.count_nonzero(events["on"]))
        if width is not None:
            invalid_coordinate_count += int(np.count_nonzero(xs >= width))
        if height is not None:
            invalid_coordinate_count += int(np.count_nonzero(ys >= height))

    return {
        "width": width,
        "height": height,
        "packet_count": packet_count,
        "event_count": event_count,
        "t_start": t_start,
        "t_end": t_end,
        "duration_us": None if t_start is None else t_end - t_start,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "on_count": on_count,
        "off_count": event_count - on_count,
        "timestamps_monotonic": monotonic,
        "invalid_coordinate_count": invalid_coordinate_count,
    }


def write_preview(events: np.ndarray, output: str | Path, limit: int = 1000) -> int:
    """Write up to limit events to a human-readable CSV file."""
    preview_count = min(limit, len(events))
    with Path(output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "polarity"])
        for event in events[:preview_count]:
            writer.writerow(
                [int(event["t"]), int(event["x"]), int(event["y"]), bool(event["on"])]
            )
    return preview_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--preview", type=Path, default=Path("events_preview.csv"))
    parser.add_argument("--preview-events", type=int, default=1000)
    args = parser.parse_args()

    events = load_events(args.path)
    print("===== basic information =====")
    print("file:", args.path)
    print("event count:", len(events))
    print("dtype:", events.dtype)
    print("stream metadata:", event_stream_metadata(args.path))
    print("first 10 events:")
    print(events[:10])
    preview_count = write_preview(events, args.preview, args.preview_events)
    print(f"wrote {preview_count} events to {args.preview}")


if __name__ == "__main__":
    main()
