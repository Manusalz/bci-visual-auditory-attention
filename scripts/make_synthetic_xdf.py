"""Create a small synthetic XDF recording for demos and tests.

The file contains no real participant data. It simulates:

- an EEG stream named `openvibeSignal`, 8 channels, 250 Hz;
- a marker stream named `BCI_Markers`;
- a visual block with left/right cues, targets and space responses;
- an auditory block with targets, standards and high hit rate.

The writer implements the subset of XDF needed by pyxdf for numeric and
string streams. It is intentionally small and transparent.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


EEG_CHANNELS = ["FZ", "CZ", "P3", "PZ", "P4", "PO7", "PO8", "OZ"]


def _varlen_int(value: int) -> bytes:
    if value < 256:
        return bytes([1, value])
    if value < 2**32:
        return bytes([4]) + struct.pack("<I", value)
    return bytes([8]) + struct.pack("<Q", value)


def _chunk(tag: int, payload: bytes, stream_id: int | None = None) -> bytes:
    body = struct.pack("<H", tag)
    if stream_id is not None:
        body += struct.pack("<I", stream_id)
    body += payload
    return _varlen_int(len(body)) + body


def _stream_header(
    stream_id: int,
    name: str,
    typ: str,
    channel_count: int,
    nominal_srate: float,
    channel_format: str,
    labels: list[str],
) -> bytes:
    channels = "".join(
        f"<channel><label>{escape(label)}</label><type>{escape(typ)}</type></channel>"
        for label in labels
    )
    xml = f"""<?xml version="1.0"?>
    <info>
      <name>{escape(name)}</name>
      <type>{escape(typ)}</type>
      <channel_count>{channel_count}</channel_count>
      <nominal_srate>{nominal_srate}</nominal_srate>
      <channel_format>{channel_format}</channel_format>
      <source_id>synthetic_public_example_{stream_id}</source_id>
      <desc><channels>{channels}</channels></desc>
    </info>"""
    return _chunk(2, xml.encode("utf-8"), stream_id=stream_id)


def _stream_footer(sample_count: int, first_ts: float, last_ts: float) -> bytes:
    xml = f"""<info>
      <sample_count>{sample_count}</sample_count>
      <first_timestamp>{first_ts:.6f}</first_timestamp>
      <last_timestamp>{last_ts:.6f}</last_timestamp>
    </info>"""
    return xml.encode("utf-8")


def _samples_numeric(timestamps: np.ndarray, values: np.ndarray) -> bytes:
    payload = _varlen_int(len(timestamps))
    values = np.asarray(values, dtype="<f4")
    for ts, row in zip(timestamps, values):
        payload += b"\x01" + struct.pack("<d", float(ts))
        payload += row.astype("<f4", copy=False).tobytes(order="C")
    return payload


def _samples_string(timestamps: list[float], values: list[str]) -> bytes:
    payload = _varlen_int(len(timestamps))
    for ts, value in zip(timestamps, values):
        raw = value.encode("utf-8")
        payload += b"\x01" + struct.pack("<d", float(ts))
        payload += _varlen_int(len(raw)) + raw
    return payload


def build_synthetic_events() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(ts: float, marker: str, kind: str = "", correct: object = "") -> None:
        rows.append(
            {
                "time_s": round(ts, 3),
                "marker": marker,
                "kind": kind,
                "responded_correctly": correct,
            }
        )

    add(2.0, "baseline/open/start")
    add(12.0, "baseline/open/end")
    add(13.0, "baseline/closed/start")
    add(23.0, "baseline/closed/end")
    add(25.0, "avsync/countdown/exp2/visual/task_onset")

    rng = np.random.default_rng(20260701)
    t = 30.0
    visual_sides = ["L", "R"] * 10
    rng.shuffle(visual_sides)
    visual_misses = {7}
    visual_false_alarm = 13
    for i, side in enumerate(visual_sides, start=1):
        add(t, f"visual/trial_start/b1_{i}")
        add(t + 0.25, f"visual/cue/{side}", "visual_cue")
        add(t + 1.45, f"visual/target/{side}", "visual_target")
        if i not in visual_misses:
            rt = float(rng.normal(0.54, 0.09))
            rt = min(max(rt, 0.28), 1.15)
            add(t + 1.45 + rt, "visual/response/space", "space_response", True)
        if i == visual_false_alarm:
            add(t + 0.75, "visual/response/space", "space_response", False)
        add(t + 2.15, f"visual/trial_end/b1_{i}")
        t += 2.8

    add(t + 1.0, "audio2/block_start/attL/b1")
    t += 2.0
    audio_events = []
    for i in range(36):
        if i in {3, 9, 15, 21, 27, 33}:
            audio_events.append(("L", "tgt", True))
        elif i in {6, 18, 30}:
            audio_events.append(("R", "tgt", False))
        else:
            audio_events.append((rng.choice(["L", "R"]), "std", False))
    audio_miss_target_index = 21
    for i, (side, subtype, attended_target) in enumerate(audio_events, start=1):
        marker = f"audio2/event/{side}/{subtype}"
        add(t, marker, "audio_event", attended_target)
        if attended_target and i != audio_miss_target_index:
            rt = float(rng.normal(0.48, 0.08))
            rt = min(max(rt, 0.25), 0.95)
            add(t + rt, "audio2/response/space", "space_response", True)
        t += 0.85
    add(t + 1.0, "exp/end")

    return sorted(rows, key=lambda row: float(row["time_s"]))


def build_synthetic_eeg(duration_s: float, events: list[dict[str, object]], sfreq: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260701)
    ts = np.arange(0, duration_s, 1.0 / sfreq)
    data = rng.normal(0.0, 2.0, size=(len(ts), len(EEG_CHANNELS))).astype(np.float32)

    # Background rhythms in microvolt-like units.
    for ch, phase in enumerate(np.linspace(0, math.pi, len(EEG_CHANNELS))):
        data[:, ch] += 5.0 * np.sin(2 * math.pi * 10.0 * ts + phase)
        data[:, ch] += 1.2 * np.sin(2 * math.pi * 6.0 * ts + phase / 2)

    # Visual covert-attention-like lateralized alpha modulation after cues.
    for event in events:
        marker = str(event["marker"])
        if not marker.startswith("visual/cue/"):
            continue
        side = marker.rsplit("/", 1)[-1]
        start = float(event["time_s"]) + 0.35
        stop = float(event["time_s"]) + 1.35
        mask = (ts >= start) & (ts <= stop)
        envelope = np.sin(np.linspace(0, math.pi, int(mask.sum()))) if mask.any() else []
        if side == "L":
            data[mask, EEG_CHANNELS.index("PO8")] -= 3.0 * envelope
            data[mask, EEG_CHANNELS.index("PO7")] += 1.0 * envelope
        else:
            data[mask, EEG_CHANNELS.index("PO7")] -= 3.0 * envelope
            data[mask, EEG_CHANNELS.index("PO8")] += 1.0 * envelope

    # Auditory P300-like centro-parietal positivity after attended target hits.
    for event in events:
        if event.get("kind") != "audio_event" or event.get("responded_correctly") is not True:
            continue
        center = float(event["time_s"]) + 0.38
        width = 0.09
        bump = 7.0 * np.exp(-0.5 * ((ts - center) / width) ** 2)
        for ch_name in ["CZ", "P3", "PZ", "P4"]:
            data[:, EEG_CHANNELS.index(ch_name)] += bump

    return ts, data


def write_xdf(path: Path, events: list[dict[str, object]], ts: np.ndarray, eeg: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    marker_ts = [float(row["time_s"]) for row in events]
    marker_values = [str(row["marker"]) for row in events]
    file_header = b"<info><version>1.0</version></info>"
    with path.open("wb") as f:
        f.write(b"XDF:")
        f.write(_chunk(1, file_header))
        f.write(
            _stream_header(
                1,
                "openvibeSignal",
                "signal",
                len(EEG_CHANNELS),
                250.0,
                "float32",
                EEG_CHANNELS,
            )
        )
        f.write(_stream_header(2, "BCI_Markers", "Markers", 1, 0.0, "string", ["marker"]))
        f.write(_chunk(4, struct.pack("<dd", 0.0, 0.0), stream_id=1))
        f.write(_chunk(4, struct.pack("<dd", 0.0, 0.0), stream_id=2))
        f.write(_chunk(3, _samples_numeric(ts, eeg), stream_id=1))
        f.write(_chunk(3, _samples_string(marker_ts, marker_values), stream_id=2))
        f.write(_chunk(6, _stream_footer(len(ts), float(ts[0]), float(ts[-1])), stream_id=1))
        f.write(_chunk(6, _stream_footer(len(marker_ts), marker_ts[0], marker_ts[-1]), stream_id=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-xdf", default="data/synthetic/P01_synthetic_high_performance.xdf")
    parser.add_argument("--out-events", default="data/synthetic/P01_synthetic_high_performance_events.csv")
    args = parser.parse_args(argv)

    events = build_synthetic_events()
    duration = max(float(row["time_s"]) for row in events) + 3.0
    ts, eeg = build_synthetic_eeg(duration, events)
    write_xdf(Path(args.out_xdf), events, ts, eeg)

    events_path = Path(args.out_events)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "marker", "kind", "responded_correctly"])
        writer.writeheader()
        writer.writerows(events)
    print(f"Wrote {args.out_xdf}")
    print(f"Wrote {args.out_events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
