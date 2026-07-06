"""Plot a time window from one stream inside an XDF recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyxdf


def _stream_name(stream: dict) -> str:
    return stream["info"]["name"][0]


def _channel_labels(stream: dict, n_channels: int) -> list[str]:
    labels = []
    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        for channel in channels:
            labels.append(channel.get("label", [None])[0] or "")
    except (KeyError, IndexError, TypeError):
        pass
    if len(labels) != n_channels or not any(labels):
        labels = [f"ch{i + 1}" for i in range(n_channels)]
    return labels


def _find_stream(streams: list[dict], name: str) -> dict:
    matches = [stream for stream in streams if _stream_name(stream) == name]
    if not matches:
        available = ", ".join(_stream_name(stream) for stream in streams)
        raise SystemExit(f"Stream '{name}' not found. Available streams: {available}")
    return matches[0]


def _parse_channels(value: str, n_channels: int) -> list[int]:
    if value.lower() == "all":
        return list(range(n_channels))
    channels = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        channel_number = int(part)
        if channel_number < 1 or channel_number > n_channels:
            raise SystemExit(f"Channel {channel_number} is outside 1..{n_channels}")
        channels.append(channel_number - 1)
    return channels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", required=True, help="Path to the XDF file.")
    parser.add_argument("--stream", default="openvibeSignal", help="Stream name to plot.")
    parser.add_argument("--start", type=float, default=0.0, help="Window start in seconds from stream start.")
    parser.add_argument("--duration", type=float, default=10.0, help="Window duration in seconds.")
    parser.add_argument(
        "--channels",
        default="1,2,3,4,5,6,7,8",
        help="1-based channel list, for example '1,2,5', or 'all'.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier applied before plotting. Use 1e6 if data are in volts and you want microvolts.",
    )
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument("--save", default="", help="Optional PNG output path. If omitted, opens an interactive plot.")
    args = parser.parse_args()

    xdf_path = Path(args.xdf)
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    stream = _find_stream(streams, args.stream)

    ts = np.asarray(stream["time_stamps"], dtype=float)
    data = np.asarray(stream["time_series"], dtype=float)
    if ts.size == 0 or data.size == 0:
        raise SystemExit(f"Stream '{args.stream}' has no samples.")
    if data.ndim == 1:
        data = data[:, None]

    rel_t = ts - ts[0]
    end = args.start + args.duration
    mask = (rel_t >= args.start) & (rel_t <= end)
    if not np.any(mask):
        available = rel_t[-1]
        raise SystemExit(f"No samples in {args.start:.3f}-{end:.3f}s. Stream duration is {available:.3f}s.")

    channels = _parse_channels(args.channels, data.shape[1])
    labels = _channel_labels(stream, data.shape[1])
    t = rel_t[mask]
    y = data[mask][:, channels] * args.scale

    fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=(12, max(3, 1.5 * len(channels))))
    if len(channels) == 1:
        axes = [axes]
    for ax, channel_index, values in zip(axes, channels, y.T):
        ax.plot(t, values, linewidth=0.8)
        ax.set_ylabel(labels[channel_index])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Seconds from stream start")
    title = args.title or f"{xdf_path.name} | {args.stream} | {args.start:.2f}-{end:.2f}s"
    fig.suptitle(title)
    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
