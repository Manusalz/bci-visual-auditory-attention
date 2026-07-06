"""Extract visual alpha features from an XDF plus an anonymized cue table.

Expected event CSV columns:
- cue_time_s: cue onset in seconds from EEG stream start;
- cue_side: L or R;
- optional metadata columns such as trial_id, block, included, hit, miss.

The public repository does not include raw XDF files. This script documents the
feature-extraction logic used before the public anonymized feature table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import butter, hilbert, sosfiltfilt


CHANNELS = ["PO7", "PO8", "OZ", "PZ"]
WINDOWS = {
    "cue_0p5_1p2": (0.5, 1.2),
}
BASELINE = (-0.8, -0.2)


def stream_name(stream: dict) -> str:
    return stream["info"]["name"][0]


def find_stream(streams: list[dict], name: str) -> dict:
    for stream in streams:
        if stream_name(stream) == name:
            return stream
    available = ", ".join(stream_name(stream) for stream in streams)
    raise SystemExit(f"Stream {name!r} not found. Available: {available}")


def channel_labels(stream: dict) -> list[str]:
    n_channels = int(stream["info"]["channel_count"][0])
    labels: list[str] = []
    try:
        for channel in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(channel.get("label", [""])[0])
    except (KeyError, IndexError, TypeError):
        pass
    if len(labels) != n_channels or not any(labels):
        labels = [f"ch{i + 1}" for i in range(n_channels)]
    return labels


def to_microvolts(x: np.ndarray) -> np.ndarray:
    scale = float(np.nanpercentile(np.abs(x), 95))
    return x * 1e6 if np.isfinite(scale) and scale < 1e-2 else x


def alpha_power(data_uv: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low / (sfreq / 2.0), high / (sfreq / 2.0)], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, data_uv, axis=0)
    return np.abs(hilbert(filtered, axis=0)) ** 2


def db_ratio(post: float, base: float) -> float:
    if not np.isfinite(post) or not np.isfinite(base) or base <= 0 or post <= 0:
        return float("nan")
    return float(10.0 * np.log10(post / base))


def event_features(power: np.ndarray, times_s: np.ndarray, event_s: float, window: tuple[float, float]) -> dict[str, float]:
    rel = times_s - event_s
    base_mask = (rel >= BASELINE[0]) & (rel <= BASELINE[1])
    metric_mask = (rel >= window[0]) & (rel <= window[1])
    values: dict[str, float] = {}
    for idx, ch in enumerate(CHANNELS):
        base = float(np.nanmean(power[base_mask, idx])) if np.any(base_mask) else np.nan
        post = float(np.nanmean(power[metric_mask, idx])) if np.any(metric_mask) else np.nan
        values[f"{ch}_alpha_db"] = db_ratio(post, base)
    return values


def lateralized(values: dict[str, float], cue_side: str) -> dict[str, float | str]:
    contra = "PO8" if cue_side == "L" else "PO7"
    ipsi = "PO7" if cue_side == "L" else "PO8"
    contra_db = float(values[f"{contra}_alpha_db"])
    ipsi_db = float(values[f"{ipsi}_alpha_db"])
    return {
        "contra_channel": contra,
        "ipsi_channel": ipsi,
        "contra_alpha_db": contra_db,
        "ipsi_alpha_db": ipsi_db,
        "contra_minus_ipsi_db": contra_db - ipsi_db,
        "po8_minus_po7_db": float(values["PO8_alpha_db"]) - float(values["PO7_alpha_db"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eeg-stream", default="openvibeSignal")
    parser.add_argument("--alpha-low", type=float, default=8.0)
    parser.add_argument("--alpha-high", type=float, default=12.0)
    args = parser.parse_args(argv)

    streams, _ = pyxdf.load_xdf(args.xdf)
    eeg = find_stream(streams, args.eeg_stream)
    labels = channel_labels(eeg)
    idx = [labels.index(ch) for ch in CHANNELS]
    data_uv = to_microvolts(np.asarray(eeg["time_series"], dtype=float))[:, idx]
    times_s = np.asarray(eeg["time_stamps"], dtype=float)
    times_s = times_s - times_s[0]
    sfreq = float(eeg["info"]["nominal_srate"][0])
    power = alpha_power(data_uv, sfreq, args.alpha_low, args.alpha_high)

    events = pd.read_csv(args.events)
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        cue_side = str(event["cue_side"])
        if cue_side not in {"L", "R"}:
            continue
        for window_name, window in WINDOWS.items():
            values = event_features(power, times_s, float(event["cue_time_s"]), window)
            row = event.to_dict()
            row.update(
                {
                    "alpha_low_hz": args.alpha_low,
                    "alpha_high_hz": args.alpha_high,
                    "window": window_name,
                    **values,
                    **lateralized(values, cue_side),
                }
            )
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
