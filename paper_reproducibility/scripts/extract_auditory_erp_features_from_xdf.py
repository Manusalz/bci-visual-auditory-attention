"""Extract auditory ERP features from an XDF plus an anonymized event table.

Expected event CSV columns:
- event_time_s: auditory event onset in seconds from EEG stream start;
- condition: standard, target_atendido_hit, target_atendido_miss, or target_ignorado;
- optional metadata columns such as participant, module, block, instruction.

The public repository does not include raw XDF files. This script documents the
ERP feature-extraction logic used before the public anonymized feature table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt


CHANNELS = ["FZ", "CZ", "P3", "PZ", "P4"]
CP_CHANNELS = ["CZ", "P3", "PZ", "P4"]
BASELINE = (-0.2, 0.0)
WINDOWS = {
    "p300_250_600": (0.25, 0.60),
}


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


def preprocess_erp(data_uv: np.ndarray, sfreq: float) -> np.ndarray:
    b, a = iirnotch(50.0, Q=30.0, fs=sfreq)
    notched = filtfilt(b, a, data_uv, axis=0)
    sos = butter(4, [0.1 / (sfreq / 2.0), 15.0 / (sfreq / 2.0)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, notched, axis=0)


def epoch_features(data_uv: np.ndarray, times_s: np.ndarray, event_s: float, window: tuple[float, float]) -> dict[str, float]:
    rel = times_s - event_s
    base_mask = (rel >= BASELINE[0]) & (rel <= BASELINE[1])
    win_mask = (rel >= window[0]) & (rel <= window[1])
    epoch_mask = (rel >= -0.2) & (rel <= 0.8)
    if not np.any(base_mask) or not np.any(win_mask):
        return {}
    baseline = np.nanmean(data_uv[base_mask, :], axis=0)
    corrected = data_uv - baseline
    peak_to_peak = float(np.nanmax(corrected[epoch_mask, :]) - np.nanmin(corrected[epoch_mask, :])) if np.any(epoch_mask) else np.nan
    values: dict[str, float] = {"peak_to_peak_uv": peak_to_peak}
    for idx, ch in enumerate(CHANNELS):
        values[f"{ch}_mean_uv"] = float(np.nanmean(corrected[win_mask, idx]))
    cp_idx = [CHANNELS.index(ch) for ch in CP_CHANNELS]
    cp = np.nanmean(corrected[:, cp_idx], axis=1)
    values["centro_parietal_mean_uv"] = float(np.nanmean(cp[win_mask]))
    values["centro_parietal_peak_uv"] = float(np.nanmax(cp[win_mask]))
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eeg-stream", default="openvibeSignal")
    parser.add_argument("--reject-peak-to-peak-uv", type=float, default=250.0)
    args = parser.parse_args(argv)

    streams, _ = pyxdf.load_xdf(args.xdf)
    eeg = find_stream(streams, args.eeg_stream)
    labels = channel_labels(eeg)
    idx = [labels.index(ch) for ch in CHANNELS]
    data_uv = to_microvolts(np.asarray(eeg["time_series"], dtype=float))[:, idx]
    sfreq = float(eeg["info"]["nominal_srate"][0])
    data_uv = preprocess_erp(data_uv, sfreq)
    times_s = np.asarray(eeg["time_stamps"], dtype=float)
    times_s = times_s - times_s[0]

    events = pd.read_csv(args.events)
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        for window_name, window in WINDOWS.items():
            values = epoch_features(data_uv, times_s, float(event["event_time_s"]), window)
            if not values:
                continue
            if values["peak_to_peak_uv"] > args.reject_peak_to_peak_uv:
                continue
            row = event.to_dict()
            row.update({"window": window_name, **values})
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
