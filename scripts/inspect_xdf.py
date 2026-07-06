"""Inspect streams inside an XDF recording."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyxdf


def stream_name(stream: dict) -> str:
    return stream["info"]["name"][0]


def stream_type(stream: dict) -> str:
    return stream["info"].get("type", [""])[0]


def channel_count(stream: dict) -> int:
    return int(float(stream["info"].get("channel_count", ["0"])[0]))


def nominal_srate(stream: dict) -> float:
    return float(stream["info"].get("nominal_srate", ["0"])[0])


def stream_duration(stream: dict) -> float:
    stamps = stream.get("time_stamps", [])
    if len(stamps) < 2:
        return 0.0
    return float(stamps[-1] - stamps[0])


def inspect_xdf(path: Path) -> list[dict[str, object]]:
    streams, _header = pyxdf.load_xdf(str(path))
    rows: list[dict[str, object]] = []
    for idx, stream in enumerate(streams):
        stamps = stream.get("time_stamps", [])
        duration = stream_duration(stream)
        samples = len(stamps)
        effective_srate = samples / duration if duration > 0 else 0.0
        rows.append(
            {
                "stream_index": idx,
                "name": stream_name(stream),
                "type": stream_type(stream),
                "channels": channel_count(stream),
                "nominal_srate_hz": nominal_srate(stream),
                "samples": samples,
                "duration_s": round(duration, 6),
                "effective_srate_hz": round(effective_srate, 6),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", required=True, help="Path to XDF recording.")
    parser.add_argument("--out", help="Optional CSV output path.")
    args = parser.parse_args(argv)

    rows = inspect_xdf(Path(args.xdf))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(
            f"{row['stream_index']}: {row['name']} | {row['type']} | "
            f"ch={row['channels']} | nominal={row['nominal_srate_hz']} Hz | "
            f"samples={row['samples']} | duration={row['duration_s']} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
