"""Plot EEG channels with marker lines from an XDF recording."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyxdf
from matplotlib.patches import Rectangle


DEFAULT_MARKER_REGEX = (
    r"visual/(cue|target|response/space)|"
    r"audio[24]/(block_start|event/.*/tgt|response/space)|"
    r"audio[24]/response/space|"
    r"debug/global/space|"
    r"assr/block_(start|end)|"
    r"baseline/(open|closed)/(start|end)"
)

AUDIO2_DISPLAY = "Audio de dos corrientes"
AUDIO4_DISPLAY = "Audio de cuatro clases"

AudioMarker = dict[str, float | str | None]


def _stream_name(stream: dict) -> str:
    return stream["info"]["name"][0]


def _find_stream(streams: list[dict], name: str) -> dict:
    matches = [stream for stream in streams if _stream_name(stream) == name]
    if not matches:
        available = ", ".join(_stream_name(stream) for stream in streams)
        raise SystemExit(f"Stream '{name}' not found. Available streams: {available}")
    return matches[0]


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
    if not channels:
        raise SystemExit("No channels selected.")
    return channels


def _marker_color(label: str) -> str:
    if "debug/global/space" in label or "response/space" in label:
        return "#d62728"
    if "visual/cue" in label:
        return "#1f77b4"
    if "visual/target" in label or "/tgt" in label:
        return "#2ca02c"
    if "baseline/" in label:
        return "#9467bd"
    if "assr/" in label:
        return "#ff7f0e"
    return "#7f7f7f"


def _audio4_target_for_class(class_name: str | None) -> tuple[str, str] | None:
    mapping = {
        "left_low": ("L", "tgt_low"),
        "left_high": ("L", "tgt_high"),
        "right_low": ("R", "tgt_low"),
        "right_high": ("R", "tgt_high"),
    }
    if class_name is None:
        return None
    return mapping.get(class_name)


def _marker_style(label: str, context: dict[str, str | None]) -> tuple[str, str, str, float]:
    if "debug/global/space" in label:
        return "espacio global", "#d62728", "-", 1.4
    if "response/space" in label:
        return "respuesta espacio", "#a50f15", "-", 1.8
    if label.startswith("visual/cue/"):
        side = label.rsplit("/", 1)[-1]
        context["last_visual_cue"] = side
        return f"cue visual {side}", "#1f77b4", "-", 1.2
    if label.startswith("visual/target/"):
        side = label.rsplit("/", 1)[-1]
        if context.get("last_visual_cue") == side:
            return f"target visual atendido {side}", "#238b45" if side == "L" else "#08519c", "-", 1.9
        return f"target visual no atendido {side}", "#74c476" if side == "L" else "#6baed6", "--", 1.1
    if label.startswith("audio2/block_start/att"):
        parts = label.split("/")
        if len(parts) >= 3:
            context["audio2_attended_side"] = parts[2].replace("att", "")
        return f"{AUDIO2_DISPLAY}: atiende {context.get('audio2_attended_side')}", "#8c6d31", "-", 1.2
    if label.startswith("audio2/event/") and "/tgt" in label:
        parts = label.split("/")
        side = parts[2] if len(parts) > 2 else "?"
        if side == context.get("audio2_attended_side"):
            return f"{AUDIO2_DISPLAY}: target atendido {side}", "#006d2c" if side == "L" else "#08519c", "-", 2.0
        return f"{AUDIO2_DISPLAY}: target no atendido {side}", "#74c476" if side == "L" else "#6baed6", "--", 1.1
    if label.startswith("audio4/block_start/"):
        parts = label.split("/")
        if len(parts) >= 3:
            context["audio4_attended_class"] = parts[2]
        return f"{AUDIO4_DISPLAY}: atiende {context.get('audio4_attended_class')}", "#8c6d31", "-", 1.2
    if label.startswith("audio4/event/") and "/tgt" in label:
        parts = label.split("/")
        side = parts[2] if len(parts) > 2 else "?"
        event_type = parts[3] if len(parts) > 3 else "?"
        target = _audio4_target_for_class(context.get("audio4_attended_class"))
        if target == (side, event_type):
            return f"{AUDIO4_DISPLAY}: target atendido {side}/{event_type}", "#006d2c" if side == "L" else "#08519c", "-", 2.0
        return f"{AUDIO4_DISPLAY}: target no atendido {side}/{event_type}", "#74c476" if side == "L" else "#6baed6", "--", 1.1
    if "baseline/" in label:
        return "baseline", "#9467bd", "-", 1.2
    if "assr/" in label:
        return "ASSR", "#ff7f0e", "-", 1.2
    return "otro", "#7f7f7f", "-", 1.0


def _short_label(label: str) -> str:
    replacements = {
        "debug/global/space": "espacio",
        "visual/response/space": "v-espacio",
        "audio2/response/space": "a2-espacio",
        "audio4/response/space": "a4-espacio",
        "visual/cue/": "cue/",
        "visual/target/": "target/",
        "audio2/event/": "a2/",
        "audio4/event/": "a4/",
        "baseline/": "base/",
    }
    out = label
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out[-36:]


def _phase_from_marker(label: str) -> str | None:
    match = re.match(r"avsync/countdown/exp\d+/([^/]+)/", label)
    if match:
        return f"preparacion {match.group(1)}"
    if label.startswith("baseline/open/"):
        return "baseline ojos abiertos"
    if label.startswith("baseline/closed/"):
        return "baseline ojos cerrados"
    if label.startswith("visual/practice/start"):
        return "visual practica"
    if label.startswith("visual/block_start/"):
        return f"visual b{label.split('/')[-1]}"
    if label.startswith("audio2/practice/start"):
        return f"{AUDIO2_DISPLAY}: practica"
    if label.startswith("audio2/block_start/"):
        parts = label.split("/")
        if len(parts) >= 4:
            attended = parts[2].replace("attL", "atiende L").replace("attR", "atiende R")
            return f"{AUDIO2_DISPLAY}: {attended} b{parts[3]}"
        return AUDIO2_DISPLAY
    if label.startswith("audio4/practice/start"):
        return f"{AUDIO4_DISPLAY}: practica"
    if label.startswith("audio4/block_start/"):
        parts = label.split("/")
        if len(parts) >= 4:
            class_labels = {
                "left_low": "izq grave",
                "left_high": "izq agudo",
                "right_low": "der grave",
                "right_high": "der agudo",
            }
            return f"{AUDIO4_DISPLAY}: {class_labels.get(parts[2], parts[2])} b{parts[3]}"
        return AUDIO4_DISPLAY
    if label.startswith("assr/"):
        return "ASSR"
    if label.startswith("exp/start"):
        return "inicio experimento"
    if label.startswith("exp/end"):
        return "fin experimento"
    return None


def _phase_color(phase: str) -> str:
    if phase.startswith("preparacion"):
        return "#dddddd"
    if phase == "visual practica":
        return "#d9ecf8"
    if phase == "visual b1":
        return "#9ecae1"
    if phase == "visual b2":
        return "#3182bd"
    if phase.startswith("visual"):
        return "#9ecae1"
    if phase == f"{AUDIO2_DISPLAY}: practica":
        return "#fee6ce"
    if f"{AUDIO2_DISPLAY}: atiende L" in phase:
        return "#fdae6b"
    if f"{AUDIO2_DISPLAY}: atiende R" in phase:
        return "#e6550d"
    if phase.startswith(AUDIO2_DISPLAY):
        return "#fdd0a2"
    if phase == f"{AUDIO4_DISPLAY}: practica":
        return "#fdd0a2"
    if f"{AUDIO4_DISPLAY}: izq grave" in phase:
        return "#fdae6b"
    if f"{AUDIO4_DISPLAY}: izq agudo" in phase:
        return "#fd8d3c"
    if f"{AUDIO4_DISPLAY}: der grave" in phase:
        return "#e6550d"
    if f"{AUDIO4_DISPLAY}: der agudo" in phase:
        return "#a63603"
    if phase.startswith(AUDIO4_DISPLAY):
        return "#fdae6b"
    return {
        "baseline ojos abiertos": "#c7e9c0",
        "baseline ojos cerrados": "#74c476",
        "ASSR": "#dadaeb",
        "inicio experimento": "#eeeeee",
        "fin experimento": "#eeeeee",
    }.get(phase, "#eeeeee")


def _phase_segments(
    marker_stream: dict,
    eeg_start_ts: float,
    start: float,
    end: float,
) -> list[tuple[float, float, str]]:
    points: list[tuple[float, str]] = []
    for ts, value in zip(marker_stream["time_stamps"], marker_stream["time_series"]):
        label = str(value[0] if isinstance(value, (list, tuple, np.ndarray)) else value)
        phase = _phase_from_marker(label)
        if phase is None:
            continue
        points.append((float(ts - eeg_start_ts), phase))

    if not points:
        return []

    segments: list[tuple[float, float, str]] = []
    for idx, (phase_start, phase) in enumerate(points):
        phase_end = points[idx + 1][0] if idx + 1 < len(points) else end
        if phase_end < start or phase_start > end:
            continue
        clipped_start = max(start, phase_start)
        clipped_end = min(end, phase_end)
        if clipped_end > clipped_start:
            segments.append((clipped_start, clipped_end, phase))

    if points[0][0] > start:
        segments.insert(0, (start, min(points[0][0], end), "antes de modulo/registro desconocido"))
    return segments


def _write_marker_csv(path: Path, marker_rows: list[AudioMarker]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seconds_from_eeg_start", "kind", "marker"])
        for row in marker_rows:
            writer.writerow([f"{float(row['t']):.6f}", row["kind"], row["label"]])


def _apply_visual_clip(y: np.ndarray, clip_uv: float) -> tuple[np.ndarray, int]:
    if clip_uv <= 0:
        return y, 0
    clipped = np.clip(y, -clip_uv, clip_uv)
    n_clipped = int(np.count_nonzero(clipped != y))
    return clipped, n_clipped


def _apply_robust_ylim(ax, values: np.ndarray, percentile: float, min_span: float = 5.0) -> None:
    if percentile <= 0:
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    percentile = min(max(percentile, 50.0), 100.0)
    lower_tail = (100.0 - percentile) / 2.0
    lo, hi = np.percentile(finite, [lower_tail, 100.0 - lower_tail])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi <= lo:
        center = float(np.median(finite))
        lo = center - min_span / 2.0
        hi = center + min_span / 2.0
    span = max(float(hi - lo), min_span)
    center = float((hi + lo) / 2.0)
    pad = span * 0.15
    ax.set_ylim(center - span / 2.0 - pad, center + span / 2.0 + pad)


def _add_side_panel(fig, marker_rows: list[AudioMarker], phase_rows: list[tuple[float, float, str]]) -> None:
    panel = fig.add_axes([0.75, 0.08, 0.24, 0.84])
    panel.set_axis_off()
    panel.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=panel.transAxes,
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=0.8,
            alpha=0.95,
        )
    )

    y = 0.97
    line_h = 0.034

    def draw_header(text: str) -> None:
        nonlocal y
        panel.text(0.04, y, text, transform=panel.transAxes, fontsize=9, fontweight="bold", va="top")
        y -= line_h * 1.1

    def draw_marker_row(kind: str, count: int, color: str, linestyle: str) -> None:
        nonlocal y
        if y < 0.08:
            return
        panel.plot(
            [0.05, 0.16],
            [y - 0.012, y - 0.012],
            transform=panel.transAxes,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            solid_capstyle="butt",
        )
        panel.text(0.19, y, f"{count}  {kind}", transform=panel.transAxes, fontsize=8, va="top")
        y -= line_h

    def draw_phase_row(phase: str) -> None:
        nonlocal y
        if y < 0.08:
            return
        panel.add_patch(
            Rectangle(
                (0.05, y - 0.026),
                0.11,
                0.022,
                transform=panel.transAxes,
                facecolor=_phase_color(phase),
                edgecolor="none",
            )
        )
        panel.text(0.19, y, phase, transform=panel.transAxes, fontsize=8, va="top")
        y -= line_h

    counts = Counter(str(row["kind"]) for row in marker_rows)
    style_by_kind: dict[str, tuple[str, str]] = {}
    for row in marker_rows:
        style_by_kind.setdefault(str(row["kind"]), (str(row["color"]), str(row["linestyle"])))

    draw_header("Marcas")
    for kind, count in counts.most_common(12):
        color, linestyle = style_by_kind.get(kind, ("#777777", "-"))
        draw_marker_row(kind, count, color, linestyle)
    if len(counts) > 12 and y >= 0.08:
        panel.text(0.05, y, f"... {len(counts) - 12} tipos mas", transform=panel.transAxes, fontsize=8, va="top")
        y -= line_h

    phase_names = []
    for _, _, phase in phase_rows:
        if phase not in phase_names:
            phase_names.append(phase)
    y -= line_h * 0.35
    draw_header("Fases")
    for phase in phase_names[:12]:
        draw_phase_row(phase)
    if len(phase_names) > 12 and y >= 0.08:
        panel.text(0.05, y, f"... {len(phase_names) - 12} fases mas", transform=panel.transAxes, fontsize=8, va="top")


def _enable_click_details(fig, axes, marker_rows: list[AudioMarker], phase_rows: list[tuple[float, float, str]], duration: float) -> None:
    if not marker_rows and not phase_rows:
        return

    annotations = []
    for ax in axes:
        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xycoords="data",
            xytext=(10, -10),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "ec": "#555555", "alpha": 0.98},
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
            fontsize=8,
            va="top",
            ha="left",
            zorder=20,
        )
        annotation.set_visible(False)
        annotations.append(annotation)
    marker_tolerance = max(0.035, duration / 900.0)

    def on_click(event):
        if event.inaxes is None or event.xdata is None:
            return
        if event.inaxes not in axes:
            return

        x = float(event.xdata)
        nearest = None
        nearest_dist = None
        for row in marker_rows:
            marker_rel = float(row["t"])
            dist = abs(marker_rel - x)
            if nearest_dist is None or dist < nearest_dist:
                nearest = row
                nearest_dist = dist

        text = None
        if nearest is not None and nearest_dist is not None and nearest_dist <= marker_tolerance:
            marker_rel = float(nearest["t"])
            text = f"{marker_rel:.3f} s\n{nearest['kind']}\n{nearest['label']}"
        else:
            for phase_start, phase_end, phase in phase_rows:
                if phase_start <= x <= phase_end:
                    text = f"{x:.3f} s\nfase: {phase}\nsegmento: {phase_start:.3f}-{phase_end:.3f} s"
                    break

        if text is None:
            for annotation in annotations:
                annotation.set_visible(False)
        else:
            for annotation in annotations:
                annotation.set_visible(False)
            annotation = annotations[list(axes).index(event.inaxes)]
            y = event.ydata
            if y is None:
                ymin, ymax = event.inaxes.get_ylim()
                y = ymax - 0.08 * (ymax - ymin)
            annotation.xy = (x, y)
            annotation.set_text(text)
            annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", required=True, help="Path to the XDF file.")
    parser.add_argument("--eeg-stream", default="openvibeSignal", help="Continuous signal stream to plot.")
    parser.add_argument("--marker-stream", default="BCI_Markers", help="Marker stream to overlay.")
    parser.add_argument("--start", type=float, default=0.0, help="Window start in seconds from EEG stream start.")
    parser.add_argument("--duration", type=float, default=30.0, help="Window duration in seconds.")
    parser.add_argument("--to-end", action="store_true", help="Plot from --start to the end of the EEG stream.")
    parser.add_argument("--channels", default="1,2,3,4,5,6,7,8", help="1-based channel list or 'all'.")
    parser.add_argument(
        "--marker-regex",
        default=DEFAULT_MARKER_REGEX,
        help="Regex for marker labels to show. Use '.*' to show all markers.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier applied to signal before plotting. Use 1e6 if data are in volts.",
    )
    parser.add_argument(
        "--clip-uv",
        type=float,
        default=0.0,
        help="Visual-only clipping threshold in plotted units. Example: --clip-uv 250 limits EEG to +/-250 uV.",
    )
    parser.add_argument(
        "--robust-ylim-percentile",
        type=float,
        default=0.0,
        help=(
            "Set each EEG y-axis from the central percentile of the plotted signal. "
            "Example: 99 ignores the largest/smallest 0.5%% for display."
        ),
    )
    parser.add_argument("--save", default="", help="Optional PNG output path. If omitted, opens an interactive plot.")
    parser.add_argument(
        "--markers-out",
        default="",
        help="Optional CSV output with markers shown in the plotted window. If --save is used, defaults to '<png>_markers.csv'.",
    )
    parser.add_argument("--max-labels", type=int, default=50, help="Maximum marker labels drawn as text.")
    parser.add_argument(
        "--label-mode",
        choices=("auto", "none", "all"),
        default="auto",
        help="Marker text labels in the marker band. Hover/CSV still show full labels.",
    )
    parser.add_argument("--no-phase-band", action="store_true", help="Do not draw the bottom task phase band.")
    parser.add_argument(
        "--show-audio-response-windows",
        action="store_true",
        help="Shade post-target response windows for attended auditory targets.",
    )
    parser.add_argument("--audio-min-rt", type=float, default=0.20, help="Start of shaded auditory response window.")
    parser.add_argument("--audio-max-rt", type=float, default=1.50, help="End of shaded auditory response window.")
    args = parser.parse_args()

    xdf_path = Path(args.xdf)
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg = _find_stream(streams, args.eeg_stream)
    marker_stream = _find_stream(streams, args.marker_stream)

    eeg_ts = np.asarray(eeg["time_stamps"], dtype=float)
    eeg_data = np.asarray(eeg["time_series"], dtype=float)
    if eeg_data.ndim == 1:
        eeg_data = eeg_data[:, None]
    if eeg_ts.size == 0 or eeg_data.size == 0:
        raise SystemExit(f"EEG stream '{args.eeg_stream}' has no samples.")

    rel_t = eeg_ts - eeg_ts[0]
    stream_end = float(rel_t[-1])
    if args.to_end:
        args.duration = max(0.0, stream_end - args.start)
    end = args.start + args.duration
    mask = (rel_t >= args.start) & (rel_t <= end)
    if not np.any(mask):
        raise SystemExit(f"No EEG samples in {args.start:.3f}-{end:.3f}s.")

    channels = _parse_channels(args.channels, eeg_data.shape[1])
    labels = _channel_labels(eeg, eeg_data.shape[1])
    t = rel_t[mask]
    y = eeg_data[mask][:, channels] * args.scale
    y, n_clipped = _apply_visual_clip(y, float(args.clip_uv))
    if args.clip_uv > 0:
        print(
            f"Visual clipping only: {n_clipped} plotted samples clipped to +/-{args.clip_uv:g}. "
            "The XDF file was not modified."
        )

    pattern = re.compile(args.marker_regex)
    marker_ts = np.asarray(marker_stream["time_stamps"], dtype=float)
    marker_values = marker_stream["time_series"]
    marker_rows: list[AudioMarker] = []
    context: dict[str, str | None] = {
        "last_visual_cue": None,
        "audio2_attended_side": None,
        "audio4_attended_class": None,
    }
    for ts, value in zip(marker_ts, marker_values):
        label = str(value[0] if isinstance(value, (list, tuple, np.ndarray)) else value)
        marker_rel = float(ts - eeg_ts[0])
        kind, color, linestyle, linewidth = _marker_style(label, context)
        if args.start <= marker_rel <= end and pattern.search(label):
            marker_rows.append(
                {
                    "t": marker_rel,
                    "label": label,
                    "kind": kind,
                    "color": color,
                    "linestyle": linestyle,
                    "linewidth": linewidth,
                }
            )
    phase_rows = _phase_segments(marker_stream, float(eeg_ts[0]), args.start, end)

    markers_out = Path(args.markers_out) if args.markers_out else None
    if markers_out is None and args.save:
        save_path = Path(args.save)
        markers_out = save_path.with_name(f"{save_path.stem}_markers.csv")
    if markers_out is not None:
        _write_marker_csv(markers_out, marker_rows)
        print(f"Saved marker table: {markers_out}")

    show_phase_band = not args.no_phase_band
    n_plots = len(channels) + 1 + int(show_phase_band)
    fig, axes = plt.subplots(
        n_plots,
        1,
        sharex=True,
        figsize=(16, max(5, 1.45 * n_plots)),
        gridspec_kw={"height_ratios": [1] * len(channels) + [0.55] + ([0.45] if show_phase_band else [])},
    )
    if n_plots == 1:
        axes = [axes]

    for ax, channel_index, values in zip(axes[:-1], channels, y.T):
        ax.plot(t, values, linewidth=0.8, color="#222222")
        ax.set_ylabel(labels[channel_index])
        ax.grid(True, alpha=0.25)
        _apply_robust_ylim(ax, values, float(args.robust_ylim_percentile))
        for row in marker_rows:
            if args.show_audio_response_windows and "target atendido" in str(row["kind"]) and str(row["label"]).startswith("audio"):
                win_start = float(row["t"]) + float(args.audio_min_rt)
                win_end = min(float(row["t"]) + float(args.audio_max_rt), end)
                if win_end >= args.start and win_start <= end:
                    ax.axvspan(max(args.start, win_start), win_end, color=str(row["color"]), alpha=0.08)
            ax.axvline(
                float(row["t"]),
                color=str(row["color"]),
                alpha=0.65,
                linewidth=float(row["linewidth"]),
                linestyle=str(row["linestyle"]),
            )

    event_ax = axes[len(channels)]
    event_ax.set_ylim(0, 1)
    event_ax.set_yticks([])
    event_ax.set_ylabel("marcas")
    event_ax.grid(True, axis="x", alpha=0.25)
    draw_labels = args.label_mode == "all" or (
        args.label_mode == "auto" and args.duration <= 120 and len(marker_rows) <= args.max_labels
    )
    for idx, row in enumerate(marker_rows):
        marker_rel = float(row["t"])
        color = str(row["color"])
        event_ax.axvline(
            marker_rel,
            color=color,
            alpha=0.85,
            linewidth=float(row["linewidth"]),
            linestyle=str(row["linestyle"]),
        )
        if draw_labels and idx < args.max_labels:
            event_ax.text(
                marker_rel,
                0.05 + 0.18 * (idx % 5),
                _short_label(str(row["label"])),
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=7,
                color=color,
            )

    if show_phase_band:
        phase_ax = axes[-1]
        phase_ax.set_ylim(0, 1)
        phase_ax.set_yticks([])
        phase_ax.set_ylabel("fase")
        phase_ax.grid(True, axis="x", alpha=0.25)
        for phase_start, phase_end, phase in phase_rows:
            phase_ax.axvspan(phase_start, phase_end, color=_phase_color(phase), alpha=0.9)
            mid = phase_start + (phase_end - phase_start) / 2
            if phase_end - phase_start > max(0.75, args.duration * 0.04):
                phase_ax.text(mid, 0.5, phase, ha="center", va="center", fontsize=8)

    axes[-1].set_xlabel("Segundos desde el inicio del flujo EEG")
    title_extra = ""
    if args.clip_uv > 0:
        title_extra += f" | recorte visual +/-{args.clip_uv:g}"
    if args.robust_ylim_percentile > 0:
        title_extra += f" | eje Y robusto p{args.robust_ylim_percentile:g}"
    fig.suptitle(
        f"Visualizador local XDF/EEG | {args.start:.2f}-{end:.2f} s | {len(marker_rows)} marcas visibles{title_extra}"
    )
    fig.text(
        0.06,
        0.025,
        "Nota. Las lineas verticales son marcas sincronizadas de tarea; la banda inferior muestra la fase del ensayo/bloque; "
        "el panel derecho resume tipos de marcas y fases visibles en la ventana.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    has_side_panel = bool(marker_rows or phase_rows)
    if has_side_panel:
        _add_side_panel(fig, marker_rows, phase_rows)
        fig.subplots_adjust(left=0.06, right=0.73, top=0.93, bottom=0.12, hspace=0.20)
    else:
        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.12, hspace=0.20)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot: {out}")
    else:
        _enable_click_details(fig, axes, marker_rows, phase_rows, args.duration)
        plt.show()


if __name__ == "__main__":
    main()
