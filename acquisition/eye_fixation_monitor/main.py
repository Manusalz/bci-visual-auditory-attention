from __future__ import annotations

import argparse
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from calibration import (
    AutoCalibrationConfig,
    CalibrationConfig,
    CalibrationManager,
    apply_autocalibration_to_config,
    run_guided_autocalibration,
    _play_tone_pattern,
    _beep,
    _beep_async,
    _maybe_beep_for_cross,
)
from gaze_tracker import MEDIAPIPE_AVAILABLE, GazeEstimate, MediaPipeGazeTracker, MouseGazeTracker, SimulatedGazeTracker
from logger_utils import SafeCsvWriter, create_output_dir, setup_session_logger
from lsl_outlet import LslOutletManager, LslSettings


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def iso_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone coarse eye fixation monitor (webcam + MediaPipe + LSL).")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--mode", choices=["preview", "headless", "minimal"], default=None)
    parser.add_argument("--demo-mode", choices=["auto", "off", "simulated", "mouse"], default=None)
    parser.add_argument("--duration-s", type=float, default=0.0, help="Auto-stop after N seconds (0 = unlimited).")
    parser.add_argument("--camera-index", type=int, default=None, help="Override camera index from config.")
    parser.add_argument("--no-lsl", action="store_true", help="Disable all LSL streams.")
    parser.add_argument("--save-video", action="store_true", help="Force video recording on.")
    parser.add_argument("--no-save-video", action="store_true", help="Force video recording off.")
    parser.add_argument("--output-root", default=None, help="Override output root folder.")
    parser.add_argument("--ready-file", default=None, help="Write this file after calibration is ready.")
    parser.add_argument("--stop-file", default=None, help="Exit cleanly when this file appears.")
    parser.add_argument("--mirror-preview", action="store_true", help="Force mirrored preview window.")
    parser.add_argument("--no-mirror-preview", action="store_true", help="Force non-mirrored preview window.")
    parser.add_argument("--autocalibrate", action="store_true", help="Run guided center/right/left autocalibration at start.")
    parser.add_argument("--no-autocalibrate", action="store_true", help="Disable guided autocalibration at start.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def save_config(path: Path, cfg: dict, logger: Any | None = None) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    if logger is not None:
        logger.info("Config saved: %s", path)


def build_autocalibration_config(auto_cfg: dict, threshold_cfg: dict) -> AutoCalibrationConfig:
    side_angle_raw = auto_cfg.get("side_angle_deg", None)
    if side_angle_raw in (None, "", 0, 0.0):
        side_angle_deg = float(threshold_cfg.get("degree_radius", 4.0))
    else:
        side_angle_deg = float(side_angle_raw)

    return AutoCalibrationConfig(
        enabled=bool(auto_cfg.get("enabled", False)),
        center_duration_s=float(auto_cfg.get("center_duration_s", 3.0)),
        side_duration_s=float(auto_cfg.get("side_duration_s", 2.5)),
        min_confidence=float(auto_cfg.get("min_confidence", 0.35)),
        min_phase_samples=int(auto_cfg.get("min_phase_samples", 12)),
        side_angle_deg=float(side_angle_deg),
        fullscreen=bool(auto_cfg.get("fullscreen", True)),
        window_name=str(auto_cfg.get("window_name", "Eye AutoCalibration")),
        target_radius_px=int(auto_cfg.get("target_radius_px", 18)),
        show_webcam_thumbnail=bool(auto_cfg.get("show_webcam_thumbnail", True)),
        beep_enabled=bool(auto_cfg.get("beep_enabled", True)),
        beep_hz=int(auto_cfg.get("beep_hz", 1100)),
        beep_short_ms=int(auto_cfg.get("beep_short_ms", 120)),
        beep_long_ms=int(auto_cfg.get("beep_long_ms", 650)),
        beep_gap_ms=int(auto_cfg.get("beep_gap_ms", 90)),
        beep_start_ms=int(auto_cfg.get("beep_start_ms", 70)),
        beep_start_interval_s=float(auto_cfg.get("beep_start_interval_s", 0.45)),
        beep_start_window_s=float(auto_cfg.get("beep_start_window_s", 0.9)),
        beep_cross_ms=int(auto_cfg.get("beep_cross_ms", 90)),
        beep_cross_interval_s=float(auto_cfg.get("beep_cross_interval_s", 0.22)),
        beep_end_guard_s=float(auto_cfg.get("beep_end_guard_s", 0.15)),
        gain_x_min=float(auto_cfg.get("gain_x_min", 0.7)),
        gain_x_max=float(auto_cfg.get("gain_x_max", 6.0)),
        vertical_angle_deg=float(auto_cfg.get("vertical_angle_deg", 4.0)),
        gain_y_min=float(auto_cfg.get("gain_y_min", 0.5)),
        gain_y_max=float(auto_cfg.get("gain_y_max", 4.0)),
        head_yaw_angle_deg=float(auto_cfg.get("head_yaw_angle_deg", 45.0)),
        pose_gain_x_min=float(auto_cfg.get("pose_gain_x_min", -0.2)),
        pose_gain_x_max=float(auto_cfg.get("pose_gain_x_max", 0.2)),
        prepare_s=float(auto_cfg.get("prepare_s", 3.0)),
        initial_instruction_s=float(auto_cfg.get("initial_instruction_s", 10.0)),
        head_instruction_s=float(auto_cfg.get("head_instruction_s", 7.0)),
        show_phase_instructions=bool(auto_cfg.get("show_phase_instructions", False)),
        persist_to_yaml=bool(auto_cfg.get("persist_to_yaml", True)),
    )


def compute_deviation(
    gaze_x: float,
    gaze_y: float,
    center_x: float,
    center_y: float,
    threshold_cfg: dict,
    geometry_cfg: dict,
    frame_w: int,
    frame_h: int,
) -> dict[str, float | str]:
    gain_x = float(threshold_cfg.get("gain_x", 1.0))
    gain_y = float(threshold_cfg.get("gain_y", 1.0))
    x_sign = -1.0 if bool(threshold_cfg.get("invert_x", False)) else 1.0
    dx_n = (gaze_x - center_x) * gain_x * x_sign
    dy_n = (gaze_y - center_y) * gain_y
    dev_px = math.hypot(dx_n * frame_w, dy_n * frame_h)
    mode = str(threshold_cfg.get("mode", "degrees")).lower()

    if mode == "normalized":
        deviation = math.hypot(dx_n, dy_n)
        threshold = float(threshold_cfg.get("normalized_radius", 0.08))
        return {
            "deviation": deviation,
            "threshold": threshold,
            "deviation_px": dev_px,
            "threshold_mode": "normalized",
            "deviation_x": dx_n,
            "deviation_y": dy_n,
        }

    if mode == "pixels":
        threshold = float(threshold_cfg.get("pixel_radius", 150.0))
        return {
            "deviation": dev_px,
            "threshold": threshold,
            "deviation_px": dev_px,
            "threshold_mode": "pixels",
            "deviation_x": dx_n * frame_w,
            "deviation_y": dy_n * frame_h,
        }

    screen_width_cm = float(geometry_cfg.get("screen_width_cm", 53.0))
    screen_width_px = float(geometry_cfg.get("screen_width_px", 1920))
    screen_height_px_cfg = geometry_cfg.get("screen_height_px", None)
    if screen_height_px_cfg in (None, "", 0, 0.0):
        screen_height_px = screen_width_px * (frame_h / max(frame_w, 1e-6))
    else:
        screen_height_px = float(screen_height_px_cfg)
    screen_height_cm_cfg = geometry_cfg.get("screen_height_cm", None)
    if screen_height_cm_cfg in (None, "", 0, 0.0):
        screen_height_cm = screen_width_cm * (screen_height_px / max(screen_width_px, 1e-6))
    else:
        screen_height_cm = float(screen_height_cm_cfg)
    eye_to_screen_cm = float(geometry_cfg.get("eye_to_screen_cm", 60.0))

    dx_screen_px = dx_n * screen_width_px
    dy_screen_px = dy_n * screen_height_px
    dx_cm = dx_screen_px * (screen_width_cm / max(screen_width_px, 1e-6))
    dy_cm = dy_screen_px * (screen_height_cm / max(screen_height_px, 1e-6))
    dx_deg = math.degrees(math.atan2(dx_cm, max(eye_to_screen_cm, 1e-6)))
    dy_deg = math.degrees(math.atan2(dy_cm, max(eye_to_screen_cm, 1e-6)))
    dev_deg = math.hypot(dx_deg, dy_deg)
    threshold = float(threshold_cfg.get("degree_radius", 4.0))
    return {
        "deviation": dev_deg,
        "threshold": threshold,
        "deviation_px": math.hypot(dx_screen_px, dy_screen_px),
        "threshold_mode": "degrees",
        "deviation_x": dx_deg,
        "deviation_y": dy_deg,
    }


def apply_head_yaw_guard(
    *,
    deviation: float,
    deviation_x: float,
    deviation_y: float,
    threshold_mode: str,
    yaw_proxy: float,
    threshold_cfg: dict,
) -> dict[str, float | bool]:
    guard_cfg = threshold_cfg.get("head_yaw_guard", {}) or {}
    if not bool(guard_cfg.get("enabled", False)):
        return {
            "fixbreak_deviation": float(deviation),
            "head_guard_active": False,
            "head_yaw_margin": 0.0,
        }
    if str(threshold_mode).lower() != "degrees":
        return {
            "fixbreak_deviation": float(deviation),
            "head_guard_active": False,
            "head_yaw_margin": 0.0,
        }

    yaw_abs = abs(float(yaw_proxy))
    yaw_dead_zone = max(0.0, float(guard_cfg.get("yaw_dead_zone", 0.03)))
    if yaw_abs <= yaw_dead_zone:
        return {
            "fixbreak_deviation": float(deviation),
            "head_guard_active": False,
            "head_yaw_margin": 0.0,
        }

    dx_abs = abs(float(deviation_x))
    dy_abs = abs(float(deviation_y))
    dominance_ratio = max(0.0, float(guard_cfg.get("horizontal_dominance_ratio", 1.15)))
    if dx_abs < dy_abs * dominance_ratio:
        return {
            "fixbreak_deviation": float(deviation),
            "head_guard_active": False,
            "head_yaw_margin": 0.0,
        }

    margin_per_yaw = max(0.0, float(guard_cfg.get("margin_deg_per_yaw", 30.0)))
    margin = max(0.0, yaw_abs - yaw_dead_zone) * margin_per_yaw
    max_margin = float(guard_cfg.get("max_margin_deg", 10.0))
    if max_margin > 0.0:
        margin = min(margin, max_margin)

    guarded_dx = max(0.0, dx_abs - margin)
    guarded_deviation = float(math.hypot(guarded_dx, dy_abs))
    return {
        "fixbreak_deviation": guarded_deviation,
        "head_guard_active": bool(margin > 0.0 and guarded_deviation < float(deviation)),
        "head_yaw_margin": float(margin),
    }


def open_camera(camera_cfg: dict, override_index: int | None):
    cam_index = int(camera_cfg.get("index", 0) if override_index is None else override_index)
    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    fps = float(camera_cfg.get("fps", 30))
    backend = str(camera_cfg.get("backend", "dshow")).lower()

    if backend == "dshow":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(cam_index)
    else:
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def draw_overlay(
    frame: np.ndarray,
    estimate: GazeEstimate,
    center_x: float,
    center_y: float,
    calibrated: bool,
    calibration_progress: float,
    fixbreak: bool,
    over_threshold: bool,
    deviation: float,
    deviation_x: float,
    deviation_y: float,
    threshold: float,
    threshold_unit: str,
    mirror_preview: bool,
    show_landmarks: bool,
    draw_iris: bool,
    draw_pupil: bool,
    iris_draw_scale: float,
    pupil_draw_radius: int,
    show_text: bool,
    yaw_proxy: float = 0.0,
    cal_preparing: bool = False,
) -> np.ndarray:
    out = frame
    h, w = out.shape[:2]
    color = (0, 0, 255) if over_threshold else (0, 200, 0)

    if show_landmarks:
        for contour in estimate.eye_contours:
            if contour.size > 0:
                cv2.polylines(out, [contour], isClosed=True, color=color, thickness=1)
        if draw_iris:
            for idx, (px, py) in enumerate(estimate.iris_points):
                if idx < len(estimate.iris_circles):
                    cx, cy, r = estimate.iris_circles[idx]
                    r_draw = int(max(1, round(r * iris_draw_scale)))
                    cv2.circle(out, (int(cx), int(cy)), r_draw, (255, 255, 0), 1)
                cv2.circle(out, (px, py), 2, (255, 255, 0), -1)
        if draw_pupil:
            for px, py in estimate.pupil_points:
                pr = int(max(1, pupil_draw_radius))
                cv2.circle(out, (px, py), pr, (0, 255, 255), -1)
                cv2.circle(out, (px, py), pr + 1, (0, 255, 255), 1)
        if estimate.nose_tip is not None and estimate.eye_midpoint is not None:
            nx, ny = estimate.nose_tip
            mx, my = estimate.eye_midpoint
            cv2.circle(out, (nx, ny), 3, (0, 180, 255), -1)
            cv2.line(out, (nx, ny), (mx, my), (0, 180, 255), 1)
            cv2.circle(out, (mx, my), 3, (160, 255, 160), 1)
            if len(estimate.iris_points) >= 2:
                cv2.line(out, estimate.iris_points[0], estimate.iris_points[1], (160, 255, 160), 1)

    gaze_px = (int(np.clip(estimate.gaze_x, 0.0, 1.0) * w), int(np.clip(estimate.gaze_y, 0.0, 1.0) * h))
    center_px = (int(np.clip(center_x, 0.0, 1.0) * w), int(np.clip(center_y, 0.0, 1.0) * h))
    cv2.circle(out, center_px, 9, (255, 180, 0), 1)
    cv2.circle(out, gaze_px, 5, color, -1)
    cv2.line(out, center_px, gaze_px, color, 1)

    if show_text:
        status = "FIXBREAK" if fixbreak else ("OVER-THR" if over_threshold else "FIXATING")
        cal_text = "CALIBRATING" if not calibrated else "CALIBRATED"
        cv2.putText(out, f"{status} | {cal_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            out,
            f"gaze=({estimate.gaze_x:.3f},{estimate.gaze_y:.3f}) conf={estimate.confidence:.2f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            out,
            f"dev={deviation:.3f} thr={threshold:.3f} [{threshold_unit}]",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            out,
            f"dX={deviation_x:.3f} dY={deviation_y:.3f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
        )
        yaw_color = (0, 140, 255) if abs(yaw_proxy) > 0.15 else (160, 220, 160)
        cv2.putText(
            out,
            f"yaw_proxy={yaw_proxy:+.3f}  {'<-- cabeza girada' if abs(yaw_proxy) > 0.15 else 'cabeza al frente'}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            yaw_color,
            1,
        )
        cv2.putText(
            out,
            f"nose-eye={estimate.nose_to_eye_mid_px:.1f}px  eye_dist={estimate.iris_distance_px:.1f}px",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 220, 160),
            1,
        )
        if not calibrated:
            YELLOW = (0, 255, 255)
            if cal_preparing:
                msg = "A continuacion mantenga la mirada fija en la cruz amarilla"
                (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                cv2.putText(out, msg, (max(10, (w - mw) // 2), center_px[1] - 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, YELLOW, 2)
            else:
                # Cruz amarilla grande en el centro
                arm = 40
                cv2.line(out, (center_px[0] - arm, center_px[1]), (center_px[0] + arm, center_px[1]), YELLOW, 3)
                cv2.line(out, (center_px[0], center_px[1] - arm), (center_px[0], center_px[1] + arm), YELLOW, 3)
                cv2.circle(out, center_px, 20, YELLOW, 2)
                cv2.circle(out, center_px, 5, YELLOW, -1)
        mirror_text = "MIRROR ON" if mirror_preview else "MIRROR OFF"
        cv2.putText(out, mirror_text, (20, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 255), 1)
        cv2.putText(
            out,
            "Keys: q/ESC=quit, m=manual marker, c=recalibrate, a=autocal, v=toggle mirror",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )
    return out


def mirror_estimate_for_display(estimate: GazeEstimate, frame_width: int) -> GazeEstimate:
    mirrored_contours: list[np.ndarray] = []
    for contour in estimate.eye_contours:
        if contour.size == 0:
            mirrored_contours.append(contour)
            continue
        mirrored = contour.copy()
        mirrored[:, 0] = (frame_width - 1) - mirrored[:, 0]
        mirrored_contours.append(mirrored)

    mirrored_iris = [((frame_width - 1) - int(px), int(py)) for px, py in estimate.iris_points]
    mirrored_iris_circles = [((frame_width - 1) - int(cx), int(cy), int(r)) for cx, cy, r in estimate.iris_circles]
    mirrored_pupil = [((frame_width - 1) - int(px), int(py)) for px, py in estimate.pupil_points]
    mirrored_nose = None
    if estimate.nose_tip is not None:
        mirrored_nose = ((frame_width - 1) - int(estimate.nose_tip[0]), int(estimate.nose_tip[1]))
    mirrored_eye_mid = None
    if estimate.eye_midpoint is not None:
        mirrored_eye_mid = ((frame_width - 1) - int(estimate.eye_midpoint[0]), int(estimate.eye_midpoint[1]))
    return GazeEstimate(
        gaze_x=float(1.0 - estimate.gaze_x),
        gaze_y=float(estimate.gaze_y),
        confidence=float(estimate.confidence),
        face_found=bool(estimate.face_found),
        eye_contours=mirrored_contours,
        iris_points=mirrored_iris,
        iris_circles=mirrored_iris_circles,
        pupil_points=mirrored_pupil,
        yaw_proxy=-float(estimate.yaw_proxy),
        nose_tip=mirrored_nose,
        eye_midpoint=mirrored_eye_mid,
        iris_distance_px=float(estimate.iris_distance_px),
        nose_to_eye_mid_px=float(estimate.nose_to_eye_mid_px),
    )


def select_tracking_mode(demo_mode: str, webcam_ok: bool) -> str:
    demo_mode = demo_mode.lower()
    if demo_mode in {"simulated", "mouse"}:
        return demo_mode
    if demo_mode == "off":
        return "webcam"
    if webcam_ok and MEDIAPIPE_AVAILABLE:
        return "webcam"
    return "simulated"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mirror_preview and args.no_mirror_preview:
        parser.error("Choose either --mirror-preview or --no-mirror-preview, not both.")
    if args.autocalibrate and args.no_autocalibrate:
        parser.error("Choose either --autocalibrate or --no-autocalibrate, not both.")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    mode = args.mode or str(cfg.get("runtime", {}).get("mode", "preview"))
    mode = "headless" if mode == "minimal" else mode
    demo_mode = args.demo_mode or str(cfg.get("runtime", {}).get("demo_mode", "auto"))

    camera_cfg = dict(cfg.get("camera", {}))
    runtime_cfg = dict(cfg.get("runtime", {}))
    calibration_cfg = dict(cfg.get("calibration", {}))
    auto_cfg_raw = dict(cfg.get("auto_calibration", {}))
    threshold_cfg = dict(cfg.get("threshold", {}))
    geometry_cfg = dict(cfg.get("geometry", {}))
    video_cfg = dict(cfg.get("video", {}))
    vis_cfg = dict(cfg.get("visualization", {}))
    mediapipe_cfg = dict(cfg.get("mediapipe", {}))
    head_pose_cfg = dict(cfg.get("head_pose", {}))
    lsl_cfg = dict(cfg.get("lsl", {}))
    output_cfg = dict(cfg.get("output", {}))

    output_root = Path(args.output_root) if args.output_root else Path(runtime_cfg.get("output_root", "outputs"))
    output_dir = create_output_dir(output_root=output_root, mode=mode)
    logger = setup_session_logger(output_dir / "runtime.log", level=args.log_level)
    logger.info("Output folder: %s", output_dir.resolve())
    ready_file = Path(args.ready_file).resolve() if args.ready_file else None
    ready_written = False

    def write_ready_file(reason: str) -> None:
        nonlocal ready_written
        if ready_file is None or ready_written:
            return
        try:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.write_text(
                f"ready\nreason={reason}\noutput_dir={output_dir.resolve()}\n",
                encoding="utf-8",
            )
            ready_written = True
            logger.info("Ready file written: %s", ready_file)
        except Exception as exc:
            logger.warning("Could not write ready file %s (%s)", ready_file, exc)

    save_config(output_dir / "config_used.yaml", cfg)

    continuous_csv = SafeCsvWriter(
        output_dir / str(output_cfg.get("continuous_csv", "gaze_samples.csv")),
        [
            "local_ts",
            "iso_ts",
            "lsl_ts",
            "source_mode",
            "calibrated",
            "gaze_x",
            "gaze_y",
            "center_x",
            "center_y",
            "confidence",
            "face_found",
            "fixbreak",
            "over_threshold",
            "deviation",
            "deviation_x",
            "deviation_y",
            "threshold",
            "threshold_mode",
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
            "saccade_candidate",
        ],
    )
    events_csv = SafeCsvWriter(
        output_dir / str(output_cfg.get("events_csv", "events.csv")),
        [
            "local_ts",
            "iso_ts",
            "event",
            "marker",
            "fixbreak",
            "gaze_x",
            "gaze_y",
            "confidence",
            "deviation",
            "deviation_x",
            "deviation_y",
            "threshold",
            "threshold_mode",
            "fixbreak_deviation",
            "head_guard_active",
            "head_yaw_margin",
            "gaze_speed_norm_s",
            "gaze_speed_deg_s",
            "saccade_candidate",
            "note",
        ],
    )

    lsl_settings = LslSettings(
        enabled=bool(lsl_cfg.get("enabled", True) and not args.no_lsl),
        marker_stream_name=str(lsl_cfg.get("marker_stream_name", "EyeFix_Markers")),
        marker_stream_type=str(lsl_cfg.get("marker_stream_type", "Markers")),
        marker_source_id=str(lsl_cfg.get("marker_source_id", "eye_fix_marker_01")),
        sample_stream_name=str(lsl_cfg.get("sample_stream_name", "EyeFix_Gaze")),
        sample_stream_type=str(lsl_cfg.get("sample_stream_type", "EyeGaze")),
        sample_source_id=str(lsl_cfg.get("sample_source_id", "eye_fix_gaze_01")),
        nominal_srate=float(lsl_cfg.get("nominal_srate", 30.0)),
    )
    lsl_manager = LslOutletManager(lsl_settings, logger)

    cap = None
    tracker = None
    video_writer = None
    video_path = output_dir / str(video_cfg.get("filename", "webcam_overlay.mp4"))
    stop_file = Path(args.stop_file) if args.stop_file else None
    save_video = bool(video_cfg.get("save", False))
    if args.save_video:
        save_video = True
    if args.no_save_video:
        save_video = False

    auto_cfg = build_autocalibration_config(auto_cfg_raw, threshold_cfg)
    auto_enabled = bool(auto_cfg.enabled)
    if args.autocalibrate:
        auto_enabled = True
    if args.no_autocalibrate:
        auto_enabled = False

    lsl_manager.push_marker("eye/monitor/start")
    events_csv.write_row(
        {
            "local_ts": time.time(),
            "iso_ts": iso_ts(time.time()),
            "event": "monitor_start",
            "marker": "eye/monitor/start",
            "note": "",
        }
    )

    try:
        demo_mode_norm = demo_mode.lower()
        if demo_mode_norm in {"simulated", "mouse"}:
            cap = None
            webcam_ok = False
        else:
            cap = open_camera(camera_cfg, args.camera_index)
            webcam_ok = cap is not None

        tracking_mode = select_tracking_mode(demo_mode_norm, webcam_ok=webcam_ok)

        if tracking_mode == "webcam" and not webcam_ok:
            msg = "Webcam unavailable and demo_mode=off. Exiting cleanly."
            logger.error(msg)
            events_csv.write_row(
                {
                    "local_ts": time.time(),
                    "iso_ts": iso_ts(time.time()),
                    "event": "fatal",
                    "marker": "",
                    "note": msg,
                }
            )
            return 2

        if tracking_mode == "webcam" and not MEDIAPIPE_AVAILABLE:
            msg = "mediapipe not installed. Use demo_mode=simulated/mouse or install requirements."
            logger.error(msg)
            events_csv.write_row(
                {
                    "local_ts": time.time(),
                    "iso_ts": iso_ts(time.time()),
                    "event": "fatal",
                    "marker": "",
                    "note": msg,
                }
            )
            return 3

        if tracking_mode == "webcam":
            task_model_path_cfg = mediapipe_cfg.get("task_model_path", "models/face_landmarker.task")
            task_model_path = Path(task_model_path_cfg)
            if not task_model_path.is_absolute():
                task_model_path = (config_path.parent / task_model_path).resolve()
            tracker = MediaPipeGazeTracker(
                backend=str(mediapipe_cfg.get("preferred_backend", "auto")),
                task_model_path=str(task_model_path),
                task_model_url=str(
                    mediapipe_cfg.get(
                        "task_model_url",
                        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                        "face_landmarker/float16/1/face_landmarker.task",
                    )
                ),
                auto_download_task_model=bool(mediapipe_cfg.get("auto_download_task_model", True)),
                head_pose_correction=bool(head_pose_cfg.get("enabled", False)),
                pose_gain_x=float(head_pose_cfg.get("gain_x", 0.03)),
                pose_dead_zone=float(head_pose_cfg.get("dead_zone", 0.0)),
            )
            logger.info("Tracking mode: webcam + MediaPipe.")
        elif tracking_mode == "mouse":
            if cap is not None:
                cap.release()
                cap = None
            tracker = MouseGazeTracker()
            logger.info("Tracking mode: mouse demo.")
        else:
            if cap is not None:
                cap.release()
                cap = None
            tracker = SimulatedGazeTracker()
            logger.info("Tracking mode: simulated demo.")

        calibration_manager = CalibrationManager(
            CalibrationConfig(
                duration_s=float(calibration_cfg.get("duration_s", 7.0)),
                min_confidence=float(calibration_cfg.get("min_confidence", 0.35)),
                min_samples=int(calibration_cfg.get("min_samples", 60)),
            )
        )
        center_x, center_y = 0.5, 0.5
        calibrated = False

        def run_and_apply_autocalibration(reason: str) -> tuple[bool, bool]:
            nonlocal center_x, center_y, calibrated
            if cap is None or tracking_mode != "webcam":
                logger.warning("Autocalibration requested but webcam tracking is not active.")
                return False, False

            t0 = time.time()
            lsl_manager.push_marker("eye/autocalibration/start")
            events_csv.write_row(
                {
                    "local_ts": t0,
                    "iso_ts": iso_ts(t0),
                    "event": "autocalibration_start",
                    "marker": "eye/autocalibration/start",
                    "note": reason,
                }
            )
            result = run_guided_autocalibration(
                cap=cap,
                tracker=tracker,
                cfg=auto_cfg,
                geometry_cfg=geometry_cfg,
                logger=logger,
                marker_callback=lsl_manager.push_marker,
            )
            t1 = time.time()
            lsl_manager.push_marker("eye/autocalibration/end")

            if result.aborted:
                events_csv.write_row(
                    {
                        "local_ts": t1,
                        "iso_ts": iso_ts(t1),
                        "event": "autocalibration_cancelled",
                        "marker": "eye/autocalibration/end",
                        "note": result.message,
                    }
                )
                logger.info(result.message)
                return False, True

            if not result.success:
                events_csv.write_row(
                    {
                        "local_ts": t1,
                        "iso_ts": iso_ts(t1),
                        "event": "autocalibration_failed",
                        "marker": "eye/autocalibration/end",
                        "note": result.message,
                    }
                )
                logger.warning(result.message)
                return False, False

            center_x, center_y = float(result.center_x), float(result.center_y)
            threshold_cfg["gain_x"] = float(result.gain_x)
            threshold_cfg["invert_x"] = bool(result.invert_x)
            threshold_cfg["gain_y"] = float(result.gain_y)
            calibrated = True
            # Aplicar pose_gain_x calibrado al tracker en vivo
            if hasattr(tracker, "pose_gain_x"):
                tracker.pose_gain_x = float(result.pose_gain_x)

            events_csv.write_row(
                {
                    "local_ts": t1,
                    "iso_ts": iso_ts(t1),
                    "event": "autocalibration_ok",
                    "marker": "eye/autocalibration/end",
                    "gaze_x": center_x,
                    "gaze_y": center_y,
                    "threshold": float(result.gain_x),
                    "threshold_mode": "gain_x",
                    "note": result.message,
                }
            )
            logger.info(result.message)
            write_ready_file("autocalibration_ok")

            if auto_cfg.persist_to_yaml:
                try:
                    apply_autocalibration_to_config(cfg, result)
                    save_config(config_path, cfg, logger=logger)
                    save_config(output_dir / "config_used.yaml", cfg, logger=None)
                except Exception as exc:
                    logger.warning("Could not persist autocalibration to YAML: %s", exc)

            return True, False

        if auto_enabled:
            if tracking_mode != "webcam":
                logger.warning("Autocalibration requested but tracking_mode=%s. Skipping.", tracking_mode)
            else:
                ok_auto, aborted_auto = run_and_apply_autocalibration(reason="startup")
                if aborted_auto:
                    return 0
                if ok_auto:
                    logger.info(
                        "Autocalibration applied on startup. gain_x=%.3f invert_x=%s",
                        float(threshold_cfg.get("gain_x", 1.0)),
                        bool(threshold_cfg.get("invert_x", False)),
                    )

        _center_cal_prepare_s = float(auto_cfg_raw.get("prepare_s", 3.0))
        _cal_last_beep = -999.0
        _center_cal_cross_start = 0.0
        if not calibrated:
            _center_cal_preparing = True
            _center_cal_prep_end = time.time() + _center_cal_prepare_s
            lsl_manager.push_marker("eye/calibration/prepare")
            events_csv.write_row(
                {
                    "local_ts": time.time(),
                    "iso_ts": iso_ts(time.time()),
                    "event": "calibration_prepare",
                    "marker": "eye/calibration/prepare",
                    "note": "Showing center instruction before calibration.",
                }
            )
        else:
            _center_cal_preparing = False
            _center_cal_prep_end = 0.0
            write_ready_file("calibration_ready")
            events_csv.write_row(
                {
                    "local_ts": time.time(),
                    "iso_ts": iso_ts(time.time()),
                    "event": "calibration_ready",
                    "marker": "",
                    "note": "Using autocalibrated center and gain_x/invert_x.",
                }
            )

        min_fixbreak_s = float(threshold_cfg.get("min_fixbreak_ms", 120)) / 1000.0
        min_recover_s = float(threshold_cfg.get("min_recover_ms", 80)) / 1000.0
        no_face_behavior = str(runtime_cfg.get("no_face_behavior", "ignore")).lower()
        loop_hz = float(runtime_cfg.get("loop_hz", 30))
        loop_dt = 1.0 / max(loop_hz, 1.0)
        mirror_preview = bool(runtime_cfg.get("mirror_preview", False))
        if args.mirror_preview:
            mirror_preview = True
        if args.no_mirror_preview:
            mirror_preview = False
        manual_marker_name = str(lsl_cfg.get("manual_test_marker", "eye/manual/test"))
        logger.info("Preview mirror: %s", "ON" if mirror_preview else "OFF")

        frame_w = int(camera_cfg.get("width", 640))
        frame_h = int(camera_cfg.get("height", 480))
        win_name = "Eye Fixation Monitor"
        _preview_centered = False

        fixbreak = False
        saccade_active = False
        prev_motion_ts: float | None = None
        prev_gaze_x: float | None = None
        prev_gaze_y: float | None = None
        prev_deviation_x: float | None = None
        prev_deviation_y: float | None = None
        over_since = None
        under_since = None
        start_ts = time.time()
        frame_failures = 0

        while True:
            tick = time.time()
            lsl_ts = LslOutletManager.lsl_now()

            if stop_file is not None and stop_file.exists():
                logger.info("Stop file detected, shutting down cleanly: %s", stop_file)
                break

            if cap is not None:
                ok, raw_frame = cap.read()
                if not ok:
                    frame_failures += 1
                    if frame_failures > 20:
                        raise RuntimeError("Repeated webcam frame read failure.")
                    time.sleep(0.01)
                    continue
                frame_failures = 0
            else:
                if mode == "preview" or save_video:
                    raw_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                else:
                    raw_frame = None

            if raw_frame is not None:
                frame_h, frame_w = raw_frame.shape[:2]
            else:
                frame_w = int(camera_cfg.get("width", frame_w))
                frame_h = int(camera_cfg.get("height", frame_h))
            estimate = tracker.estimate(raw_frame, now_s=tick)  # type: ignore[union-attr]

            if not calibrated:
                if _center_cal_preparing:
                    if tick >= _center_cal_prep_end:
                        _center_cal_preparing = False
                        _center_cal_cross_start = tick
                        _cal_last_beep = -999.0
                        calibration_manager.start(tick)
                        lsl_manager.push_marker("eye/calibration/start")
                        events_csv.write_row(
                            {
                                "local_ts": tick,
                                "iso_ts": iso_ts(tick),
                                "event": "calibration_start",
                                "marker": "eye/calibration/start",
                                "note": "Look at screen center.",
                            }
                        )
                else:
                    calibration_manager.add_sample(estimate.gaze_x, estimate.gaze_y, estimate.confidence)
                # Beep solo cuando la cruz esta visible y se estan tomando muestras.
                if not _center_cal_preparing:
                    cal_elapsed = max(0.0, tick - float(_center_cal_cross_start))
                    cal_remaining = max(0.0, float(calibration_manager.cfg.duration_s) - cal_elapsed)
                    _cal_last_beep = _maybe_beep_for_cross(
                        cfg=auto_cfg,
                        elapsed_s=cal_elapsed,
                        remaining_s=cal_remaining,
                        last_beep_s=float(_cal_last_beep),
                    )

                if not _center_cal_preparing and calibration_manager.is_done(tick):
                    cal = calibration_manager.finalize()
                    center_x, center_y = cal.center_x, cal.center_y
                    calibrated = True
                    logger.info(cal.message)
                    lsl_manager.push_marker("eye/calibration/end")
                    write_ready_file("center_calibration_end")
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "calibration_end",
                            "marker": "eye/calibration/end",
                            "gaze_x": center_x,
                            "gaze_y": center_y,
                            "confidence": estimate.confidence,
                            "note": cal.message,
                        }
                    )

            deviation_metrics = compute_deviation(
                estimate.gaze_x,
                estimate.gaze_y,
                center_x,
                center_y,
                threshold_cfg=threshold_cfg,
                geometry_cfg=geometry_cfg,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            deviation = float(deviation_metrics["deviation"])
            threshold = float(deviation_metrics["threshold"])
            deviation_px = float(deviation_metrics["deviation_px"])
            threshold_mode = str(deviation_metrics["threshold_mode"])
            deviation_x = float(deviation_metrics["deviation_x"])
            deviation_y = float(deviation_metrics["deviation_y"])
            fixbreak_guard = apply_head_yaw_guard(
                deviation=deviation,
                deviation_x=deviation_x,
                deviation_y=deviation_y,
                threshold_mode=threshold_mode,
                yaw_proxy=estimate.yaw_proxy,
                threshold_cfg=threshold_cfg,
            )
            fixbreak_deviation = float(fixbreak_guard["fixbreak_deviation"])
            head_guard_active = bool(fixbreak_guard["head_guard_active"])
            head_yaw_margin = float(fixbreak_guard["head_yaw_margin"])
            gaze_speed_norm_s = 0.0
            gaze_speed_deg_s = 0.0
            if prev_motion_ts is not None:
                dt_motion = max(1e-6, float(tick - prev_motion_ts))
                if prev_gaze_x is not None and prev_gaze_y is not None:
                    gaze_speed_norm_s = float(
                        math.hypot(estimate.gaze_x - prev_gaze_x, estimate.gaze_y - prev_gaze_y) / dt_motion
                    )
                if prev_deviation_x is not None and prev_deviation_y is not None and threshold_mode == "degrees":
                    gaze_speed_deg_s = float(
                        math.hypot(deviation_x - prev_deviation_x, deviation_y - prev_deviation_y) / dt_motion
                    )
            saccade_cfg = threshold_cfg.get("saccade_candidate", {}) or {}
            saccade_enabled = bool(saccade_cfg.get("enabled", True))
            saccade_speed_threshold = float(saccade_cfg.get("speed_deg_s", 80.0))
            saccade_min_confidence = float(saccade_cfg.get("min_confidence", 0.35))
            saccade_candidate = bool(
                saccade_enabled
                and calibrated
                and estimate.face_found
                and estimate.confidence >= saccade_min_confidence
                and threshold_mode == "degrees"
                and gaze_speed_deg_s >= saccade_speed_threshold
            )
            over_threshold = False

            if calibrated:
                if estimate.face_found:
                    is_over = fixbreak_deviation > threshold
                else:
                    is_over = no_face_behavior == "fixbreak"
                over_threshold = bool(is_over)

                entered_fixbreak = False
                ended_fixbreak = False

                if is_over:
                    under_since = None
                    if not fixbreak:
                        if over_since is None:
                            over_since = tick
                        elif (tick - over_since) >= min_fixbreak_s:
                            fixbreak = True
                            entered_fixbreak = True
                else:
                    over_since = None
                    if fixbreak:
                        if under_since is None:
                            under_since = tick
                        elif (tick - under_since) >= min_recover_s:
                            fixbreak = False
                            ended_fixbreak = True
                    else:
                        under_since = None

                if entered_fixbreak:
                    lsl_manager.push_marker("eye/fixbreak/start", timestamp=lsl_ts)
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "fixbreak_start",
                            "marker": "eye/fixbreak/start",
                            "fixbreak": 1,
                            "gaze_x": estimate.gaze_x,
                            "gaze_y": estimate.gaze_y,
                            "confidence": estimate.confidence,
                            "deviation": deviation,
                            "deviation_x": deviation_x,
                            "deviation_y": deviation_y,
                            "threshold": threshold,
                            "threshold_mode": threshold_mode,
                            "fixbreak_deviation": fixbreak_deviation,
                            "head_guard_active": int(head_guard_active),
                            "head_yaw_margin": head_yaw_margin,
                            "note": "",
                        }
                    )
                if ended_fixbreak:
                    lsl_manager.push_marker("eye/fixbreak/end", timestamp=lsl_ts)
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "fixbreak_end",
                            "marker": "eye/fixbreak/end",
                            "fixbreak": 0,
                            "gaze_x": estimate.gaze_x,
                            "gaze_y": estimate.gaze_y,
                            "confidence": estimate.confidence,
                            "deviation": deviation,
                            "deviation_x": deviation_x,
                            "deviation_y": deviation_y,
                            "threshold": threshold,
                            "threshold_mode": threshold_mode,
                            "fixbreak_deviation": fixbreak_deviation,
                            "head_guard_active": int(head_guard_active),
                            "head_yaw_margin": head_yaw_margin,
                            "note": "",
                        }
                    )
            else:
                fixbreak = False

            if calibrated:
                if saccade_candidate and not saccade_active:
                    saccade_active = True
                    lsl_manager.push_marker("eye/saccade/candidate/start", timestamp=lsl_ts)
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "saccade_candidate_start",
                            "marker": "eye/saccade/candidate/start",
                            "fixbreak": int(fixbreak),
                            "gaze_x": estimate.gaze_x,
                            "gaze_y": estimate.gaze_y,
                            "confidence": estimate.confidence,
                            "deviation": deviation,
                            "deviation_x": deviation_x,
                            "deviation_y": deviation_y,
                            "threshold": threshold,
                            "threshold_mode": threshold_mode,
                            "fixbreak_deviation": fixbreak_deviation,
                            "head_guard_active": int(head_guard_active),
                            "head_yaw_margin": head_yaw_margin,
                            "gaze_speed_norm_s": gaze_speed_norm_s,
                            "gaze_speed_deg_s": gaze_speed_deg_s,
                            "saccade_candidate": 1,
                            "note": "Coarse webcam speed threshold crossed.",
                        }
                    )
                elif (not saccade_candidate) and saccade_active:
                    saccade_active = False
                    lsl_manager.push_marker("eye/saccade/candidate/end", timestamp=lsl_ts)
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "saccade_candidate_end",
                            "marker": "eye/saccade/candidate/end",
                            "fixbreak": int(fixbreak),
                            "gaze_x": estimate.gaze_x,
                            "gaze_y": estimate.gaze_y,
                            "confidence": estimate.confidence,
                            "deviation": deviation,
                            "deviation_x": deviation_x,
                            "deviation_y": deviation_y,
                            "threshold": threshold,
                            "threshold_mode": threshold_mode,
                            "fixbreak_deviation": fixbreak_deviation,
                            "head_guard_active": int(head_guard_active),
                            "head_yaw_margin": head_yaw_margin,
                            "gaze_speed_norm_s": gaze_speed_norm_s,
                            "gaze_speed_deg_s": gaze_speed_deg_s,
                            "saccade_candidate": 0,
                            "note": "",
                        }
                    )
            else:
                saccade_active = False

            lsl_manager.push_sample(
                local_ts=tick,
                gaze_x=estimate.gaze_x,
                gaze_y=estimate.gaze_y,
                confidence=estimate.confidence,
                face_found=1 if estimate.face_found else 0,
                calibrated=1 if calibrated else 0,
                fixbreak=1 if fixbreak else 0,
                over_threshold=1 if over_threshold else 0,
                saccade_candidate=1 if saccade_candidate else 0,
                deviation=deviation,
                deviation_x=deviation_x,
                deviation_y=deviation_y,
                threshold=threshold,
                deviation_px=deviation_px,
                yaw_proxy=estimate.yaw_proxy,
                nose_x=float("nan") if estimate.nose_tip is None else int(estimate.nose_tip[0]),
                nose_y=float("nan") if estimate.nose_tip is None else int(estimate.nose_tip[1]),
                eye_mid_x=float("nan") if estimate.eye_midpoint is None else int(estimate.eye_midpoint[0]),
                eye_mid_y=float("nan") if estimate.eye_midpoint is None else int(estimate.eye_midpoint[1]),
                iris_distance_px=estimate.iris_distance_px,
                nose_to_eye_mid_px=estimate.nose_to_eye_mid_px,
                fixbreak_deviation=fixbreak_deviation,
                head_guard_active=1 if head_guard_active else 0,
                head_yaw_margin=head_yaw_margin,
                gaze_speed_norm_s=gaze_speed_norm_s,
                gaze_speed_deg_s=gaze_speed_deg_s,
                timestamp=lsl_ts,
            )
            continuous_csv.write_row(
                {
                    "local_ts": tick,
                    "iso_ts": iso_ts(tick),
                    "lsl_ts": lsl_ts,
                    "source_mode": tracking_mode,
                    "calibrated": int(calibrated),
                    "gaze_x": estimate.gaze_x,
                    "gaze_y": estimate.gaze_y,
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": estimate.confidence,
                    "face_found": int(estimate.face_found),
                    "fixbreak": int(fixbreak),
                    "over_threshold": int(over_threshold),
                    "deviation": deviation,
                    "deviation_x": deviation_x,
                    "deviation_y": deviation_y,
                    "threshold": threshold,
                    "threshold_mode": threshold_mode,
                    "deviation_px": deviation_px,
                    "yaw_proxy": estimate.yaw_proxy,
                    "nose_x": "" if estimate.nose_tip is None else int(estimate.nose_tip[0]),
                    "nose_y": "" if estimate.nose_tip is None else int(estimate.nose_tip[1]),
                    "eye_mid_x": "" if estimate.eye_midpoint is None else int(estimate.eye_midpoint[0]),
                    "eye_mid_y": "" if estimate.eye_midpoint is None else int(estimate.eye_midpoint[1]),
                    "iris_distance_px": estimate.iris_distance_px,
                    "nose_to_eye_mid_px": estimate.nose_to_eye_mid_px,
                    "fixbreak_deviation": fixbreak_deviation,
                    "head_guard_active": int(head_guard_active),
                    "head_yaw_margin": head_yaw_margin,
                    "gaze_speed_norm_s": gaze_speed_norm_s,
                    "gaze_speed_deg_s": gaze_speed_deg_s,
                    "saccade_candidate": int(saccade_candidate),
                }
            )
            prev_motion_ts = tick
            prev_gaze_x = float(estimate.gaze_x)
            prev_gaze_y = float(estimate.gaze_y)
            prev_deviation_x = float(deviation_x)
            prev_deviation_y = float(deviation_y)

            if mode == "preview" or (save_video and raw_frame is not None):
                if raw_frame is None:
                    raw_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                preview_frame = raw_frame.copy()
                overlay_estimate = estimate
                overlay_center_x = center_x
                overlay_dx = deviation_x
                if mirror_preview:
                    preview_frame = cv2.flip(preview_frame, 1)
                    overlay_estimate = mirror_estimate_for_display(estimate, frame_width=frame_w)
                    overlay_center_x = float(1.0 - center_x)
                    overlay_dx = float(-deviation_x)
                cal_prog = calibration_manager.progress(tick) if not calibrated else 1.0
                preview = draw_overlay(
                    frame=preview_frame,
                    estimate=overlay_estimate,
                    center_x=overlay_center_x,
                    center_y=center_y,
                    calibrated=calibrated,
                    calibration_progress=cal_prog,
                    fixbreak=fixbreak,
                    over_threshold=over_threshold,
                    deviation=deviation,
                    deviation_x=overlay_dx,
                    deviation_y=deviation_y,
                    threshold=threshold,
                    threshold_unit=threshold_mode,
                    mirror_preview=mirror_preview,
                    show_landmarks=bool(vis_cfg.get("show_landmarks", True)),
                    draw_iris=bool(vis_cfg.get("draw_iris", True)),
                    draw_pupil=bool(vis_cfg.get("draw_pupil", True)),
                    iris_draw_scale=float(vis_cfg.get("iris_draw_scale", 0.85)),
                    pupil_draw_radius=int(vis_cfg.get("pupil_draw_radius", 3)),
                    show_text=bool(vis_cfg.get("show_text", True)),
                    yaw_proxy=overlay_estimate.yaw_proxy,
                    cal_preparing=_center_cal_preparing,
                )
                if mode == "preview":
                    cv2.imshow(win_name, preview)
                    if not _preview_centered:
                        try:
                            import ctypes as _ct
                            _sw = _ct.windll.user32.GetSystemMetrics(0)
                            _sh = _ct.windll.user32.GetSystemMetrics(1)
                            cv2.moveWindow(win_name, max(0, (_sw - frame_w) // 2), max(0, (_sh - frame_h) // 2))
                        except Exception:
                            pass
                        _preview_centered = True
            else:
                preview = raw_frame

            if save_video:
                if video_writer is None:
                    codec = str(video_cfg.get("codec", "mp4v"))
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer_fps = float(camera_cfg.get("fps", loop_hz))
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, writer_fps, (frame_w, frame_h))
                    if not video_writer.isOpened():
                        logger.warning("Could not initialize VideoWriter, disabling video output.")
                        video_writer.release()
                        video_writer = None
                        save_video = False
                if video_writer is not None:
                    if preview is None:
                        preview = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                    video_writer.write(preview)

            if mode == "preview":
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("m"):
                    lsl_manager.push_marker(manual_marker_name)
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "manual_marker",
                            "marker": manual_marker_name,
                            "fixbreak": int(fixbreak),
                            "gaze_x": estimate.gaze_x,
                            "gaze_y": estimate.gaze_y,
                            "confidence": estimate.confidence,
                            "deviation": deviation,
                            "deviation_x": deviation_x,
                            "deviation_y": deviation_y,
                            "threshold": threshold,
                            "threshold_mode": threshold_mode,
                            "note": "Pressed key m in preview.",
                        }
                    )
                if key == ord("c"):
                    calibration_manager.start(tick)
                    calibrated = False
                    center_x, center_y = 0.5, 0.5
                    fixbreak = False
                    over_since = None
                    under_since = None
                    lsl_manager.push_marker("eye/calibration/start")
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "calibration_restart",
                            "marker": "eye/calibration/start",
                            "note": "Manual recalibration requested.",
                        }
                    )
                if key == ord("a"):
                    if tracking_mode != "webcam" or cap is None:
                        note = "Autocalibration key ignored: webcam mode is not active."
                        logger.warning(note)
                        events_csv.write_row(
                            {
                                "local_ts": tick,
                                "iso_ts": iso_ts(tick),
                                "event": "autocalibration_skipped",
                                "marker": "",
                                "note": note,
                            }
                        )
                    else:
                        ok_auto, aborted_auto = run_and_apply_autocalibration(reason="manual_key_a")
                        if aborted_auto:
                            break
                        if ok_auto:
                            fixbreak = False
                            over_since = None
                            under_since = None
                if key == ord("v"):
                    mirror_preview = not mirror_preview
                    logger.info("Preview mirror toggled: %s", "ON" if mirror_preview else "OFF")
                    events_csv.write_row(
                        {
                            "local_ts": tick,
                            "iso_ts": iso_ts(tick),
                            "event": "preview_mirror_toggle",
                            "marker": "",
                            "note": f"mirror_preview={'ON' if mirror_preview else 'OFF'}",
                        }
                    )
                try:
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
            else:
                time.sleep(max(0.0, loop_dt - (time.time() - tick)))

            if args.duration_s > 0 and (tick - start_ts) >= args.duration_s:
                logger.info("Duration limit reached: %.2fs", args.duration_s)
                break

        if fixbreak:
            lsl_manager.push_marker("eye/fixbreak/end")
            events_csv.write_row(
                {
                    "local_ts": time.time(),
                    "iso_ts": iso_ts(time.time()),
                    "event": "fixbreak_end_on_shutdown",
                    "marker": "eye/fixbreak/end",
                    "fixbreak": 0,
                    "note": "Forced end marker at shutdown.",
                }
            )
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 0
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        logger.error(traceback.format_exc())
        events_csv.write_row(
            {
                "local_ts": time.time(),
                "iso_ts": iso_ts(time.time()),
                "event": "fatal_exception",
                "marker": "",
                "note": repr(exc),
            }
        )
        return 1
    finally:
        try:
            lsl_manager.push_marker("eye/monitor/end")
        except Exception:
            pass
        try:
            events_csv.write_row(
                {
                    "local_ts": time.time(),
                    "iso_ts": iso_ts(time.time()),
                    "event": "monitor_end",
                    "marker": "eye/monitor/end",
                    "note": "",
                }
            )
        except Exception:
            pass

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if tracker is not None:
                tracker.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if video_writer is not None:
                video_writer.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            continuous_csv.close()
        except Exception:
            pass
        try:
            events_csv.close()
        except Exception:
            pass
        try:
            lsl_manager.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
