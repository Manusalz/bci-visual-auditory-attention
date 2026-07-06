"""Minimal BCI_Audio LSL publisher and optional stereo playback demo."""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable

import numpy as np


def raised_cosine_ramp(n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.array([], dtype=np.float32)
    x = np.linspace(0.0, math.pi / 2.0, n_samples, dtype=np.float32)
    return np.sin(x) ** 2


def generate_tone(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float,
    ramp_s: float,
) -> np.ndarray:
    n_samples = max(1, int(round(duration_s * sample_rate)))
    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    wave = amplitude * np.sin(2.0 * np.pi * float(frequency_hz) * t)
    ramp_n = min(int(round(ramp_s * sample_rate)), n_samples // 2)
    if ramp_n > 0:
        ramp = raised_cosine_ramp(ramp_n)
        wave[:ramp_n] *= ramp
        wave[-ramp_n:] *= ramp[::-1]
    return wave.astype(np.float32)


def make_stereo(mono: np.ndarray, side: str) -> np.ndarray:
    stereo = np.zeros((mono.shape[0], 2), dtype=np.float32)
    if side.upper() == "L":
        stereo[:, 0] = mono
    elif side.upper() == "R":
        stereo[:, 1] = mono
    else:
        stereo[:, 0] = mono
        stereo[:, 1] = mono
    return stereo


def iter_chunks(samples: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, samples.shape[0], chunk_size):
        yield samples[start : start + chunk_size]


def publish_lsl(samples: np.ndarray, sample_rate: int, chunk_size: int, stream_name: str) -> None:
    from pylsl import StreamInfo, StreamOutlet, local_clock

    info = StreamInfo(stream_name, "Audio", samples.shape[1], float(sample_rate), "float32", "bci_audio_demo_01")
    channels = info.desc().append_child("channels")
    for label in ("left", "right")[: samples.shape[1]]:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("type", "audio")
        channel.append_child_value("unit", "norm")

    outlet = StreamOutlet(info, chunk_size=chunk_size)
    start_clock = local_clock() + 0.25
    for offset, chunk in enumerate(iter_chunks(samples, chunk_size)):
        chunk_start = offset * chunk_size
        timestamp = start_clock + (chunk_start / float(sample_rate))
        outlet.push_chunk(chunk.tolist(), timestamp)
        elapsed = local_clock() - start_clock
        expected = (chunk_start + chunk.shape[0]) / float(sample_rate)
        if expected > elapsed:
            time.sleep(expected - elapsed)


def play_audio(samples: np.ndarray, sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(samples, sample_rate, blocking=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a stereo tone as BCI_Audio over LSL.")
    parser.add_argument("--frequency-hz", type=float, default=1000.0)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--ramp-s", type=float, default=0.005)
    parser.add_argument("--side", choices=["L", "R", "both"], default="both")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--stream-name", default="BCI_Audio")
    parser.add_argument("--no-lsl", action="store_true")
    parser.add_argument("--play", action="store_true", help="Also play the generated tone locally.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mono = generate_tone(args.frequency_hz, args.duration_s, args.sample_rate, args.amplitude, args.ramp_s)
    samples = make_stereo(mono, args.side)

    if args.play:
        play_audio(samples, args.sample_rate)
    if not args.no_lsl:
        publish_lsl(samples, args.sample_rate, args.chunk_size, args.stream_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
