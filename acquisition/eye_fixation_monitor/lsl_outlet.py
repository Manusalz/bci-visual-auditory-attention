from __future__ import annotations

import logging
import time
from dataclasses import dataclass

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock

    PYLSL_AVAILABLE = True
except Exception:
    StreamInfo = None  # type: ignore[assignment]
    StreamOutlet = None  # type: ignore[assignment]
    local_clock = None  # type: ignore[assignment]
    PYLSL_AVAILABLE = False


@dataclass
class LslSettings:
    enabled: bool
    marker_stream_name: str
    marker_stream_type: str
    marker_source_id: str
    sample_stream_name: str
    sample_stream_type: str
    sample_source_id: str
    nominal_srate: float


EYE_SAMPLE_CHANNELS = [
    "local_ts",
    "gaze_x",
    "gaze_y",
    "confidence",
    "face_found",
    "calibrated",
    "fixbreak",
    "over_threshold",
    "saccade_candidate",
    "deviation",
    "deviation_x",
    "deviation_y",
    "threshold",
    "deviation_px",
    "yaw_proxy",
    "nose_x",
    "nose_y",
    "eye_mid_x",
    "eye_mid_y",
    "iris_distance_px",
    "nose_to_eye_mid_px",
    "fixbreak_deviation",
    "head_guard_active",
    "head_yaw_margin",
    "gaze_speed_norm_s",
    "gaze_speed_deg_s",
]


class LslOutletManager:
    def __init__(self, settings: LslSettings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.enabled = bool(settings.enabled)
        self.marker_outlet = None
        self.sample_outlet = None

        if not self.enabled:
            self.logger.info("LSL disabled by config.")
            return

        if not PYLSL_AVAILABLE:
            self.enabled = False
            self.logger.error("pylsl not available. LSL output disabled.")
            return

        marker_info = StreamInfo(
            name=settings.marker_stream_name,
            type=settings.marker_stream_type,
            channel_count=1,
            nominal_srate=0.0,
            channel_format="string",
            source_id=settings.marker_source_id,
        )
        self.marker_outlet = StreamOutlet(marker_info)

        sample_info = StreamInfo(
            name=settings.sample_stream_name,
            type=settings.sample_stream_type,
            channel_count=len(EYE_SAMPLE_CHANNELS),
            nominal_srate=float(settings.nominal_srate),
            channel_format="float32",
            source_id=settings.sample_source_id,
        )
        desc = sample_info.desc()
        channels = desc.append_child("channels")
        for label in EYE_SAMPLE_CHANNELS:
            chan = channels.append_child("channel")
            chan.append_child_value("label", label)
            chan.append_child_value("unit", "mixed")
        self.sample_outlet = StreamOutlet(sample_info)
        self.logger.info(
            "LSL outlets ready: marker='%s', sample='%s' channels=%d",
            settings.marker_stream_name,
            settings.sample_stream_name,
            len(EYE_SAMPLE_CHANNELS),
        )

    @staticmethod
    def lsl_now() -> float:
        if PYLSL_AVAILABLE and local_clock is not None:
            return float(local_clock())
        return float(time.time())

    def push_marker(self, marker: str, timestamp: float | None = None) -> None:
        if not self.enabled or self.marker_outlet is None:
            return
        ts = self.lsl_now() if timestamp is None else float(timestamp)
        self.marker_outlet.push_sample([marker], timestamp=ts)

    def push_sample(
        self,
        timestamp: float | None = None,
        **sample_values: float,
    ) -> None:
        if not self.enabled or self.sample_outlet is None:
            return
        ts = self.lsl_now() if timestamp is None else float(timestamp)
        self.sample_outlet.push_sample(
            [float(sample_values.get(label, 0.0)) for label in EYE_SAMPLE_CHANNELS],
            timestamp=ts,
        )

    def close(self) -> None:
        self.marker_outlet = None
        self.sample_outlet = None
