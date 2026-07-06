from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    import winsound

    HAS_WINSOUND = True
except Exception:
    winsound = None  # type: ignore[assignment]
HAS_WINSOUND = False


_BEEP_LOCK = threading.Lock()


@dataclass
class CalibrationConfig:
    duration_s: float = 7.0
    min_confidence: float = 0.35
    min_samples: int = 60


@dataclass
class CalibrationResult:
    center_x: float
    center_y: float
    samples_used: int
    success: bool
    message: str


class CalibrationManager:
    def __init__(self, cfg: CalibrationConfig) -> None:
        self.cfg = cfg
        self._started_at: float | None = None
        self._samples: list[tuple[float, float]] = []

    def start(self, ts: float) -> None:
        self._started_at = ts
        self._samples.clear()

    def started(self) -> bool:
        return self._started_at is not None

    def add_sample(self, gaze_x: float, gaze_y: float, confidence: float) -> None:
        if confidence >= self.cfg.min_confidence:
            self._samples.append((gaze_x, gaze_y))

    def progress(self, ts: float) -> float:
        if self._started_at is None:
            return 0.0
        elapsed = ts - self._started_at
        return float(np.clip(elapsed / max(self.cfg.duration_s, 1e-6), 0.0, 1.0))

    def is_done(self, ts: float) -> bool:
        if self._started_at is None:
            return False
        return (ts - self._started_at) >= self.cfg.duration_s

    def finalize(self) -> CalibrationResult:
        n = len(self._samples)
        if n == 0:
            return CalibrationResult(
                center_x=0.5,
                center_y=0.5,
                samples_used=n,
                success=False,
                message="No valid calibration samples. Fallback to center=(0.5,0.5).",
            )

        arr = np.asarray(self._samples, dtype=np.float64)
        center = np.median(arr, axis=0)
        if n < self.cfg.min_samples:
            return CalibrationResult(
                center_x=float(center[0]),
                center_y=float(center[1]),
                samples_used=n,
                success=False,
                message=(
                    f"Calibration partial ({n}/{self.cfg.min_samples} samples). "
                    f"Using partial center=({float(center[0]):.3f},{float(center[1]):.3f})."
                ),
            )
        return CalibrationResult(
            center_x=float(center[0]),
            center_y=float(center[1]),
            samples_used=n,
            success=True,
            message=f"Calibration OK with {n} samples.",
        )


@dataclass
class AutoCalibrationConfig:
    enabled: bool = False
    center_duration_s: float = 3.0
    side_duration_s: float = 2.5
    min_confidence: float = 0.35
    min_phase_samples: int = 12
    side_angle_deg: float = 6.0
    fullscreen: bool = True
    window_name: str = "Eye AutoCalibration"
    target_radius_px: int = 18
    show_webcam_thumbnail: bool = True
    beep_enabled: bool = True
    beep_hz: int = 1100
    beep_short_ms: int = 120
    beep_long_ms: int = 650
    beep_gap_ms: int = 90
    beep_start_ms: int = 70
    beep_start_interval_s: float = 0.45
    beep_start_window_s: float = 0.9
    beep_cross_ms: int = 90
    beep_cross_interval_s: float = 0.22
    beep_end_guard_s: float = 0.15
    gain_x_min: float = 0.7
    gain_x_max: float = 6.0
    vertical_angle_deg: float = 4.0
    gain_y_min: float = 0.5
    gain_y_max: float = 4.0
    head_yaw_angle_deg: float = 30.0
    pose_gain_x_min: float = -0.2
    pose_gain_x_max: float = 0.2
    prepare_s: float = 3.0
    initial_instruction_s: float = 10.0
    head_instruction_s: float = 7.0
    show_phase_instructions: bool = False
    persist_to_yaml: bool = True


@dataclass
class AutoCalibrationPhaseResult:
    name: str
    target_x_px: int
    target_y_px: int
    median_x: float
    median_y: float
    median_yaw_proxy: float = 0.0
    samples: int = 0
    success: bool = False
    message: str = ""


@dataclass
class AutoCalibrationResult:
    success: bool
    aborted: bool
    center_x: float
    center_y: float
    gain_x: float
    invert_x: bool
    side_angle_deg: float
    side_offset_cm: float
    side_offset_px: int
    center_samples: int
    right_samples: int
    left_samples: int
    top_samples: int = 0
    bottom_samples: int = 0
    gain_y: float = 1.0
    pose_gain_x: float = 0.03
    message: str = ""


def apply_autocalibration_to_config(cfg: dict[str, Any], result: AutoCalibrationResult) -> None:
    threshold_cfg = cfg.setdefault("threshold", {})
    threshold_cfg["gain_x"] = float(round(result.gain_x, 4))
    threshold_cfg["invert_x"] = bool(result.invert_x)
    threshold_cfg["gain_y"] = float(round(result.gain_y, 4))

    head_pose_cfg = cfg.setdefault("head_pose", {})
    head_pose_cfg["gain_x"] = float(round(result.pose_gain_x, 4))

    auto_cfg = cfg.setdefault("auto_calibration", {})
    auto_cfg["last_center_x"] = float(round(result.center_x, 6))
    auto_cfg["last_center_y"] = float(round(result.center_y, 6))
    auto_cfg["last_gain_x"] = float(round(result.gain_x, 4))
    auto_cfg["last_invert_x"] = bool(result.invert_x)
    auto_cfg["last_gain_y"] = float(round(result.gain_y, 4))
    auto_cfg["last_pose_gain_x"] = float(round(result.pose_gain_x, 4))
    auto_cfg["last_side_angle_deg"] = float(round(result.side_angle_deg, 3))
    auto_cfg["last_side_offset_cm"] = float(round(result.side_offset_cm, 3))


def _play_tone_pattern(cfg: AutoCalibrationConfig) -> None:
    if not cfg.beep_enabled:
        return
    freq = int(max(80, cfg.beep_hz))
    short_ms = int(max(40, cfg.beep_short_ms))
    long_ms = int(max(120, cfg.beep_long_ms))
    gap_s = max(0.0, float(cfg.beep_gap_ms) / 1000.0)
    _beep(freq, short_ms)
    time.sleep(gap_s)
    _beep(freq, short_ms)
    time.sleep(gap_s)
    _beep(freq, long_ms)


def _beep_async(hz: int, duration_ms: int) -> None:
    if not _BEEP_LOCK.acquire(blocking=False):
        return

    def _worker() -> None:
        try:
            _beep(hz, duration_ms)
        finally:
            try:
                _BEEP_LOCK.release()
            except RuntimeError:
                pass

    threading.Thread(target=_worker, daemon=True).start()


def _maybe_beep_for_cross(cfg: AutoCalibrationConfig, elapsed_s: float, remaining_s: float, last_beep_s: float) -> float:
    if not cfg.beep_enabled:
        return last_beep_s
    start_window_s = max(0.0, float(cfg.beep_start_window_s))
    in_start_pattern = elapsed_s < start_window_s
    duration_ms = int(max(20, cfg.beep_start_ms if in_start_pattern else cfg.beep_cross_ms))
    interval_s = max(0.05, float(cfg.beep_start_interval_s if in_start_pattern else cfg.beep_cross_interval_s))
    finish_guard_s = float(duration_ms) / 1000.0 + max(0.0, float(cfg.beep_end_guard_s))
    if remaining_s <= finish_guard_s:
        return last_beep_s
    if elapsed_s - last_beep_s >= interval_s:
        _beep_async(cfg.beep_hz, duration_ms)
        return elapsed_s
    return last_beep_s


def _safe_wait_key(delay_ms: int) -> int:
    try:
        return int(cv2.waitKey(delay_ms) & 0xFF)
    except cv2.error:
        return -1


def _phase_median(samples: list[tuple[float, float]]) -> tuple[float, float]:
    if not samples:
        return 0.5, 0.5
    arr = np.asarray(samples, dtype=np.float64)
    med = np.median(arr, axis=0)
    return float(med[0]), float(med[1])


_YELLOW = (0, 255, 255)
_YELLOW_DIM = (0, 130, 130)
_CYAN = (0, 210, 255)


def _draw_target(canvas: np.ndarray, tx: int, ty: int, arm: int, bright: bool) -> None:
    color = _YELLOW if bright else _YELLOW_DIM
    thick = 3 if bright else 2
    cv2.line(canvas, (tx - arm, ty), (tx + arm, ty), color, thick)
    cv2.line(canvas, (tx, ty - arm), (tx, ty + arm), color, thick)
    cv2.circle(canvas, (tx, ty), max(7, arm // 2), color, 2)
    cv2.circle(canvas, (tx, ty), 5, color, -1)


def _draw_face_geometry(frame: np.ndarray, estimate: Any | None) -> np.ndarray:
    out = frame.copy()
    if estimate is None:
        return out
    try:
        for contour in getattr(estimate, "eye_contours", []) or []:
            if getattr(contour, "size", 0) > 0:
                cv2.polylines(out, [contour], isClosed=True, color=(80, 220, 80), thickness=1)
        for idx, (px, py) in enumerate(getattr(estimate, "iris_points", []) or []):
            circles = getattr(estimate, "iris_circles", []) or []
            if idx < len(circles):
                cx, cy, r = circles[idx]
                cv2.circle(out, (int(cx), int(cy)), int(max(1, r)), (255, 255, 0), 1)
            cv2.circle(out, (int(px), int(py)), 2, (255, 255, 0), -1)
        nose_tip = getattr(estimate, "nose_tip", None)
        eye_midpoint = getattr(estimate, "eye_midpoint", None)
        if nose_tip is not None and eye_midpoint is not None:
            nx, ny = int(nose_tip[0]), int(nose_tip[1])
            mx, my = int(eye_midpoint[0]), int(eye_midpoint[1])
            cv2.circle(out, (nx, ny), 3, (0, 180, 255), -1)
            cv2.line(out, (nx, ny), (mx, my), (0, 180, 255), 1)
            cv2.circle(out, (mx, my), 3, (160, 255, 160), 1)
            iris_points = getattr(estimate, "iris_points", []) or []
            if len(iris_points) >= 2:
                cv2.line(out, iris_points[0], iris_points[1], (160, 255, 160), 1)
            label = (
                f"nariz-ojos={float(getattr(estimate, 'nose_to_eye_mid_px', 0.0)):.1f}px "
                f"ojos={float(getattr(estimate, 'iris_distance_px', 0.0)):.1f}px"
            )
            cv2.putText(out, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 160), 1)
    except Exception:
        return frame
    return out


def _thumb_overlay(canvas: np.ndarray, frame: np.ndarray, screen_w: int, screen_h: int, estimate: Any | None = None) -> None:
    if frame is None or frame.size == 0:
        return
    frame = _draw_face_geometry(frame, estimate)
    thumb_w = min(200, screen_w // 6)
    thumb_h = int(round(thumb_w * (frame.shape[0] / max(frame.shape[1], 1))))
    if thumb_h <= 0 or thumb_w <= 0:
        return
    thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    thumb = cv2.flip(thumb, 1)
    y0, x0 = 24, max(0, screen_w - thumb_w - 24)
    y1, x1 = min(screen_h, y0 + thumb_h), min(screen_w, x0 + thumb_w)
    canvas[y0:y1, x0:x1] = thumb[: y1 - y0, : x1 - x0]


def _wrap_instruction_text(text: str, max_chars: int = 66) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        words = [w for w in paragraph.strip().split() if w]
        if not words:
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _draw_instruction_canvas(
    screen_w: int,
    screen_h: int,
    target_xy: tuple[int, int],
    instruction: str,
    phase_label: str,
    frame: np.ndarray | None = None,
    estimate: Any | None = None,
    show_webcam_thumbnail: bool = True,
) -> np.ndarray:
    """Pantalla de preparacion: solo la frase; la cruz aparece despues."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (16, 16, 16)

    lines = _wrap_instruction_text(
        instruction or "A continuacion mantenga la mirada fija en la cruz amarilla"
    )
    font_scale, thick, line_gap = 0.95, 2, 44
    total_h = len(lines) * line_gap
    y0 = screen_h // 2 - total_h // 2 + 20
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ix = max(20, (screen_w - tw) // 2)
        color = _CYAN if i == 0 else (220, 220, 160)
        cv2.putText(canvas, line, (ix, y0 + i * line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)

    if show_webcam_thumbnail and frame is not None:
        _thumb_overlay(canvas, frame, screen_w, screen_h, estimate=estimate)
    return canvas


def _draw_autocalibration_canvas(
    frame: np.ndarray,
    screen_w: int,
    screen_h: int,
    center_xy: tuple[int, int],
    target_xy: tuple[int, int],
    target_radius_px: int,
    remaining_s: float,
    samples: int,
    min_samples: int,
    show_webcam_thumbnail: bool,
    phase_hint: str = "",
    estimate: Any | None = None,
) -> np.ndarray:
    """Pantalla de coleccion: solo la cruz amarilla brillante + info minima."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (16, 16, 16)

    cx, cy = center_xy
    tx, ty = target_xy

    # Guia sutil del centro
    cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), (38, 38, 38), 1)
    cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), (38, 38, 38), 1)

    # Cruz amarilla brillante en el target
    arm = max(32, target_radius_px + 14)
    _draw_target(canvas, tx, ty, arm=arm, bright=True)

    # Info minima en la esquina superior izquierda
    info = f"Calibrando  {max(0.0, remaining_s):.1f}s   muestras: {samples}/{min_samples}"
    cv2.putText(canvas, info, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    if phase_hint:
        cv2.putText(canvas, phase_hint, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 170, 90), 1)
    cv2.putText(canvas, "ESC o q: cancelar", (20, screen_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)

    if show_webcam_thumbnail and frame is not None and frame.size > 0:
        _thumb_overlay(canvas, frame, screen_w, screen_h, estimate=estimate)
    return canvas


def _cancelled_phase(phase_name: str, target_xy: tuple[int, int], n: int) -> AutoCalibrationPhaseResult:
    return AutoCalibrationPhaseResult(
        name=phase_name, target_x_px=target_xy[0], target_y_px=target_xy[1],
        median_x=0.5, median_y=0.5, samples=n, success=False,
        message="Autocalibration cancelled by user.",
    )


def _collect_phase(
    cap: Any,
    tracker: Any,
    cfg: AutoCalibrationConfig,
    window_name: str,
    phase_name: str,
    instruction: str,
    target_xy: tuple[int, int],
    center_xy: tuple[int, int],
    duration_s: float,
    screen_w: int,
    screen_h: int,
    marker_callback: Any = None,
    show_instruction: bool = True,
    phase_hint: str = "",
) -> tuple[AutoCalibrationPhaseResult, bool]:
    phase_label = f"Punto {phase_name}"
    last_frame: np.ndarray = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    last_estimate: Any | None = None

    if show_instruction:
        # --- Fase 1: instruccion sin sonido; la senal auditiva empieza con la cruz. ---
        prep_end = time.time() + max(0.5, cfg.prepare_s)
        while time.time() < prep_end:
            now1 = time.time()
            ok, frame = cap.read()
            if ok:
                last_frame = frame
                try:
                    last_estimate = tracker.estimate(frame, now_s=now1)
                except Exception:
                    last_estimate = None
            canvas = _draw_instruction_canvas(
                screen_w=screen_w, screen_h=screen_h,
                target_xy=target_xy, instruction=instruction,
                phase_label=phase_label, frame=last_frame,
                estimate=last_estimate,
                show_webcam_thumbnail=cfg.show_webcam_thumbnail,
            )
            cv2.imshow(window_name, canvas)
            key = _safe_wait_key(1)
            if key in (27, ord("q")):
                return _cancelled_phase(phase_name, target_xy, 0), True

    if marker_callback is not None:
        try:
            marker_callback(f"eye/calibration/phase/{phase_name}/start")
        except Exception:
            pass

    # --- Fase 2: coleccion con sonido continuo ---
    start_ts = time.time()
    samples: list[tuple[float, float]] = []
    yaw_samples: list[float] = []
    frame_failures = 0
    _col_result: tuple[AutoCalibrationPhaseResult, bool] | None = None
    _last_beep2 = -999.0

    while True:
        now = time.time()
        elapsed = now - start_ts
        if elapsed >= duration_s:
            break

        # Beep solo durante coleccion con cruz visible; nunca se dispara si puede pisar el texto siguiente.
        _last_beep2 = _maybe_beep_for_cross(
            cfg=cfg,
            elapsed_s=float(elapsed),
            remaining_s=float(duration_s - elapsed),
            last_beep_s=float(_last_beep2),
        )

        ok, frame = cap.read()
        if not ok:
            frame_failures += 1
            if frame_failures > 20:
                _col_result = (
                    AutoCalibrationPhaseResult(
                        name=phase_name, target_x_px=target_xy[0], target_y_px=target_xy[1],
                        median_x=0.5, median_y=0.5, samples=0, success=False,
                        message="Webcam frame read failed during autocalibration.",
                    ),
                    False,
                )
                break
            time.sleep(0.01)
            continue

        frame_failures = 0
        last_frame = frame
        estimate = tracker.estimate(frame, now_s=now)
        if bool(getattr(estimate, "face_found", False)) and float(getattr(estimate, "confidence", 0.0)) >= cfg.min_confidence:
            samples.append((float(getattr(estimate, "gaze_x", 0.5)), float(getattr(estimate, "gaze_y", 0.5))))
            yaw_samples.append(float(getattr(estimate, "yaw_proxy", 0.0)))

        canvas = _draw_autocalibration_canvas(
            frame=last_frame,
            screen_w=screen_w, screen_h=screen_h,
            center_xy=center_xy, target_xy=target_xy,
            target_radius_px=cfg.target_radius_px,
            remaining_s=max(0.0, duration_s - elapsed),
            samples=len(samples), min_samples=cfg.min_phase_samples,
            show_webcam_thumbnail=cfg.show_webcam_thumbnail,
            phase_hint=phase_hint,
            estimate=estimate,
        )
        cv2.imshow(window_name, canvas)
        key = _safe_wait_key(1)
        if key in (27, ord("q")):
            if marker_callback is not None:
                try:
                    marker_callback(f"eye/calibration/phase/{phase_name}/cancelled")
                except Exception:
                    pass
            _col_result = _cancelled_phase(phase_name, target_xy, len(samples)), True
            break

    if _col_result is not None:
        return _col_result

    if marker_callback is not None:
        try:
            marker_callback(f"eye/calibration/phase/{phase_name}/end")
        except Exception:
            pass
    median_x, median_y = _phase_median(samples)
    median_yaw_proxy = float(np.median(yaw_samples)) if yaw_samples else 0.0
    enough_samples = len(samples) >= cfg.min_phase_samples
    msg = f"{phase_name}: {len(samples)} valid samples."
    if not enough_samples:
        msg = f"{phase_name}: insufficient samples ({len(samples)}/{cfg.min_phase_samples})."
    return (
        AutoCalibrationPhaseResult(
            name=phase_name,
            target_x_px=target_xy[0],
            target_y_px=target_xy[1],
            median_x=median_x,
            median_y=median_y,
            median_yaw_proxy=median_yaw_proxy,
            samples=len(samples),
            success=enough_samples,
            message=msg,
        ),
        False,
    )


def _show_instruction_for_duration(
    *,
    cap: Any,
    tracker: Any,
    cfg: AutoCalibrationConfig,
    window_name: str,
    screen_w: int,
    screen_h: int,
    target_xy: tuple[int, int],
    instruction: str,
    duration_s: float,
) -> bool:
    end_ts = time.time() + max(0.5, float(duration_s))
    last_frame: np.ndarray = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    last_estimate: Any | None = None
    while time.time() < end_ts:
        now = time.time()
        ok, frame = cap.read()
        if ok:
            last_frame = frame
            try:
                last_estimate = tracker.estimate(frame, now_s=now)
            except Exception:
                last_estimate = None
        canvas = _draw_instruction_canvas(
            screen_w=screen_w,
            screen_h=screen_h,
            target_xy=target_xy,
            instruction=instruction,
            phase_label="calibracion",
            frame=last_frame,
            estimate=last_estimate,
            show_webcam_thumbnail=cfg.show_webcam_thumbnail,
        )
        cv2.imshow(window_name, canvas)
        key = _safe_wait_key(1)
        if key in (27, ord("q")):
            return True
    return False


def _beep(hz: int, duration_ms: int) -> None:
    """Beep sincrono corto para calibracion, sin abrir PortAudio si Windows puede emitirlo."""
    hz = int(max(80, hz))
    duration_ms = int(max(20, duration_ms))
    if HAS_WINSOUND and winsound is not None:
        try:
            winsound.Beep(hz, duration_ms)
            return
        except Exception:
            pass
    try:
        import ctypes

        ctypes.windll.kernel32.Beep(hz, duration_ms)
        return
    except Exception:
        pass
    try:
        import sounddevice as sd  # type: ignore

        try:
            device_info = sd.query_devices(None, "output")
            sample_rate = int(round(float(device_info.get("default_samplerate", 44100))))
        except Exception:
            sample_rate = 44100
        duration_s = max(0.01, float(duration_ms) / 1000.0)
        n_samples = max(1, int(round(duration_s * sample_rate)))
        t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
        wave = (0.18 * np.sin(2.0 * np.pi * float(hz) * t)).astype(np.float32)
        sd.play(wave, samplerate=sample_rate, blocking=True)
        sd.stop()
    except Exception:
        pass


def _yaw_fraction_label(deg: float) -> str:
    d = round(float(deg))
    if d == 45:
        return "la MITAD"
    if d == 30:
        return "un TERCIO"
    if d == 60:
        return "dos TERCIOS"
    if d == 20:
        return "un CUARTO aprox."
    return f"{d} grados"


def _setup_fullscreen_window(window_name: str, fallback_w: int, fallback_h: int) -> tuple[int, int]:
    """Ventana sin bordes a pantalla completa, compatible con el rendering de OpenCV.
    Remueve solo WS_CAPTION + WS_THICKFRAME (NO WS_POPUP, que rompe cv2.imshow).
    Posiciona la ventana en (0,0) del monitor primario con SetWindowPos en coords lógicas."""
    # Paso 1: mover a (0,0) antes de cualquier otra operación
    cv2.moveWindow(window_name, 0, 0)
    cv2.waitKey(1)

    try:
        import ctypes
        u32 = ctypes.windll.user32

        # Encontrar handle de la ventana
        hwnd = None
        for _ in range(15):
            hwnd = u32.FindWindowW(None, window_name)
            if hwnd:
                break
            time.sleep(0.05)

        if not hwnd:
            return fallback_w, fallback_h

        # Dimensiones del monitor primario en coordenadas lógicas (consistente con OpenCV)
        screen_w = u32.GetSystemMetrics(0)  # SM_CXSCREEN
        screen_h = u32.GetSystemMetrics(1)  # SM_CYSCREEN
        if screen_w <= 0 or screen_h <= 0:
            return fallback_w, fallback_h

        # Remover solo title bar y borde redimensionable — NO WS_POPUP (rompería cv2.imshow)
        GWL_STYLE = -16
        WS_CAPTION = 0x00C00000    # title bar (WS_BORDER + WS_DLGFRAME)
        WS_THICKFRAME = 0x00040000  # borde redimensionable
        style = u32.GetWindowLongW(hwnd, GWL_STYLE)
        u32.SetWindowLongW(hwnd, GWL_STYLE, style & ~WS_CAPTION & ~WS_THICKFRAME)

        # Cubrir toda la pantalla primaria y poner encima de todo
        HWND_TOPMOST = -1
        SWP_FRAMECHANGED = 0x0020
        SWP_SHOWWINDOW = 0x0040
        u32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, screen_w, screen_h,
                         SWP_FRAMECHANGED | SWP_SHOWWINDOW)
        u32.SetForegroundWindow(hwnd)
        u32.BringWindowToTop(hwnd)

        return screen_w, screen_h

    except Exception:
        return fallback_w, fallback_h


def run_guided_autocalibration(
    cap: Any,
    tracker: Any,
    cfg: AutoCalibrationConfig,
    geometry_cfg: dict[str, Any],
    logger: logging.Logger | None = None,
    marker_callback: Any = None,
) -> AutoCalibrationResult:
    if cap is None:
        return AutoCalibrationResult(
            success=False,
            aborted=False,
            center_x=0.5,
            center_y=0.5,
            gain_x=1.0,
            invert_x=False,
            side_angle_deg=cfg.side_angle_deg,
            side_offset_cm=0.0,
            side_offset_px=0,
            center_samples=0,
            right_samples=0,
            left_samples=0,
            message="Autocalibration requires an active webcam.",
        )

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    screen_w = int(geometry_cfg.get("screen_width_px", cam_w) or cam_w)
    screen_h_cfg = geometry_cfg.get("screen_height_px", None)
    if screen_h_cfg in (None, "", 0, 0.0):
        screen_h = int(round(screen_w * (cam_h / max(cam_w, 1e-6))))
    else:
        screen_h = int(screen_h_cfg)
    screen_w = int(max(640, screen_w))
    screen_h = int(max(360, screen_h))

    # Usar resolución real del monitor en Windows para evitar canvas desajustado
    try:
        import ctypes as _ctypes
        _u32 = _ctypes.windll.user32
        real_w = int(_u32.GetSystemMetrics(0))
        real_h = int(_u32.GetSystemMetrics(1))
        if real_w > 0 and real_h > 0:
            screen_w, screen_h = real_w, real_h
    except Exception:
        pass

    screen_width_cm = float(geometry_cfg.get("screen_width_cm", 53.0))
    eye_to_screen_cm = float(geometry_cfg.get("eye_to_screen_cm", 60.0))

    side_offset_cm = float(eye_to_screen_cm * math.tan(math.radians(max(0.1, cfg.side_angle_deg))))
    px_per_cm = float(screen_w / max(screen_width_cm, 1e-6))
    side_offset_px = int(round(side_offset_cm * px_per_cm))

    cx = screen_w // 2
    cy = screen_h // 2
    margin = max(40, cfg.target_radius_px * 2)
    right_x = int(np.clip(cx + side_offset_px, margin, screen_w - margin))
    left_x = int(np.clip(cx - side_offset_px, margin, screen_w - margin))
    side_offset_px_eff = int(min(right_x - cx, cx - left_x))

    window_name = str(cfg.window_name)

    # Mostrar ventana PRIMERO (necesario antes de poder hacer fullscreen en Windows)
    _blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.putText(_blank, "Preparando calibracion...",
                (max(0, screen_w // 2 - 280), screen_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 2)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, _blank)
    cv2.waitKey(1)

    # Forzar pantalla completa sin bordes con Windows API
    screen_w, screen_h = _setup_fullscreen_window(window_name, screen_w, screen_h)
    cv2.waitKey(200)

    original_head_pose_correction = getattr(tracker, "head_pose_correction", None)
    if original_head_pose_correction is not None:
        try:
            tracker.head_pose_correction = False
        except Exception:
            original_head_pose_correction = None

    # Recalcular todas las posiciones con las dimensiones reales del monitor
    cx = screen_w // 2
    cy = screen_h // 2
    margin = max(40, cfg.target_radius_px * 2)
    side_offset_px = int(round(side_offset_cm * px_per_cm))
    right_x = int(np.clip(cx + side_offset_px, margin, screen_w - margin))
    left_x = int(np.clip(cx - side_offset_px, margin, screen_w - margin))
    side_offset_px_eff = int(min(right_x - cx, cx - left_x))

    # Posiciones verticales para los puntos arriba/abajo
    screen_height_cm_est = float(geometry_cfg.get("screen_height_cm") or
                                  float(geometry_cfg.get("screen_width_cm", 53.0)) * screen_h / max(screen_w, 1))
    vert_offset_cm = float(eye_to_screen_cm * math.tan(math.radians(max(0.1, cfg.vertical_angle_deg))))
    px_per_cm_y = float(screen_h / max(screen_height_cm_est, 1.0))
    vert_offset_px = int(round(vert_offset_cm * px_per_cm_y))
    top_y = int(np.clip(cy - vert_offset_px, margin, screen_h - margin))
    bottom_y = int(np.clip(cy + vert_offset_px, margin, screen_h - margin))

    phase_specs = [
        ("central",    "Mire al centro",                                       (cx,      cy),       float(cfg.center_duration_s), ""),
        ("derecha",    "Mire a la derecha  -->",                               (right_x, cy),       float(cfg.side_duration_s), ""),
        ("izquierda",  "<--  Mire a la izquierda",                             (left_x,  cy),       float(cfg.side_duration_s), ""),
        ("arriba",     "Mire hacia arriba  ^",                                 (cx,      top_y),    float(cfg.side_duration_s), ""),
        ("abajo",      "Mire hacia abajo  v",                                  (cx,      bottom_y), float(cfg.side_duration_s), ""),
        ("yaw_derecha",   "Gire la cabeza 30 grados a la DERECHA\ncomo mirar a alguien adelante y a su derecha\npero siga mirando la cruz amarilla del centro", (cx, cy), float(cfg.side_duration_s), "Gire cabeza a la derecha; mire la cruz central"),
        ("yaw_izquierda", "Gire la cabeza 30 grados a la IZQUIERDA\ncomo mirar a alguien adelante y a su izquierda\npero siga mirando la cruz amarilla del centro", (cx, cy), float(cfg.side_duration_s), "Gire cabeza a la izquierda; mire la cruz central"),
    ]

    results: dict[str, AutoCalibrationPhaseResult] = {}
    aborted = False
    try:
        if not cfg.show_phase_instructions:
            initial_instruction = (
                "A continuacion calibraremos el software.\n"
                "Vera aparecer una cruz amarilla, primero en el centro de la pantalla "
                "y luego en otros lados.\n"
                "Mirela fija sin mover la cabeza."
            )
            aborted = _show_instruction_for_duration(
                cap=cap,
                tracker=tracker,
                cfg=cfg,
                window_name=window_name,
                screen_w=screen_w,
                screen_h=screen_h,
                target_xy=(cx, cy),
                instruction=initial_instruction,
                duration_s=max(10.0, float(cfg.initial_instruction_s)),
            )

        for phase_name, instruction, target_xy, duration_s, phase_hint in phase_specs:
            if aborted:
                break
            if not cfg.show_phase_instructions and phase_name == "yaw_derecha":
                aborted = _show_instruction_for_duration(
                    cap=cap,
                    tracker=tracker,
                    cfg=cfg,
                    window_name=window_name,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    target_xy=(cx, cy),
                    instruction="Ahora gire la cabeza a la derecha,\npero siga mirando la cruz central.",
                    duration_s=max(7.0, float(cfg.head_instruction_s)),
                )
                if aborted:
                    break
            if not cfg.show_phase_instructions and phase_name == "yaw_izquierda":
                aborted = _show_instruction_for_duration(
                    cap=cap,
                    tracker=tracker,
                    cfg=cfg,
                    window_name=window_name,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    target_xy=(cx, cy),
                    instruction="Ahora gire la cabeza a la izquierda,\npero siga mirando la cruz central.",
                    duration_s=max(7.0, float(cfg.head_instruction_s)),
                )
                if aborted:
                    break
            phase_result, phase_aborted = _collect_phase(
                cap=cap,
                tracker=tracker,
                cfg=cfg,
                window_name=window_name,
                phase_name=phase_name,
                instruction=instruction,
                target_xy=target_xy,
                center_xy=(cx, cy),
                duration_s=max(0.3, duration_s),
                screen_w=screen_w,
                screen_h=screen_h,
                marker_callback=marker_callback,
                show_instruction=cfg.show_phase_instructions,
                phase_hint=phase_hint,
            )
            results[phase_name] = phase_result
            if logger is not None:
                logger.info("Autocal phase %s: %s", phase_name, phase_result.message)
            if phase_aborted:
                aborted = True
                break
    finally:
        if original_head_pose_correction is not None:
            try:
                tracker.head_pose_correction = original_head_pose_correction
            except Exception:
                pass
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    center_res = results.get("central")
    right_res = results.get("derecha")
    left_res = results.get("izquierda")
    top_res = results.get("arriba")
    bot_res = results.get("abajo")
    yaw_r_res = results.get("yaw_derecha")
    yaw_l_res = results.get("yaw_izquierda")
    center_samples = 0 if center_res is None else center_res.samples
    right_samples = 0 if right_res is None else right_res.samples
    left_samples = 0 if left_res is None else left_res.samples
    top_samples = 0 if top_res is None else top_res.samples
    bottom_samples = 0 if bot_res is None else bot_res.samples

    if aborted:
        return AutoCalibrationResult(
            success=False, aborted=True,
            center_x=0.5, center_y=0.5, gain_x=1.0, gain_y=1.0, invert_x=False,
            side_angle_deg=cfg.side_angle_deg, side_offset_cm=side_offset_cm,
            side_offset_px=side_offset_px_eff,
            center_samples=center_samples, right_samples=right_samples,
            left_samples=left_samples, top_samples=top_samples, bottom_samples=bottom_samples,
            message="Autocalibration cancelled by user.",
        )

    if center_res is None or right_res is None or left_res is None:
        return AutoCalibrationResult(
            success=False, aborted=False,
            center_x=0.5, center_y=0.5, gain_x=1.0, gain_y=1.0, invert_x=False,
            side_angle_deg=cfg.side_angle_deg, side_offset_cm=side_offset_cm,
            side_offset_px=side_offset_px_eff,
            center_samples=center_samples, right_samples=right_samples,
            left_samples=left_samples, top_samples=top_samples, bottom_samples=bottom_samples,
            message="Autocalibration did not complete all phases.",
        )

    if not (center_res.success and right_res.success and left_res.success):
        return AutoCalibrationResult(
            success=False, aborted=False,
            center_x=center_res.median_x, center_y=center_res.median_y,
            gain_x=1.0, gain_y=1.0, invert_x=False,
            side_angle_deg=cfg.side_angle_deg, side_offset_cm=side_offset_cm,
            side_offset_px=side_offset_px_eff,
            center_samples=center_samples, right_samples=right_samples,
            left_samples=left_samples, top_samples=top_samples, bottom_samples=bottom_samples,
            message="Autocalibration had insufficient samples in one or more phases.",
        )

    # Blend center phase with side midpoint to reduce drift.
    side_mid_x = 0.5 * (right_res.median_x + left_res.median_x)
    center_x = float(np.clip(0.7 * center_res.median_x + 0.3 * side_mid_x, 0.0, 1.0))
    side_mid_y = (
        0.5 * (top_res.median_y + bot_res.median_y)
        if top_res is not None and bot_res is not None
        else center_res.median_y
    )
    center_y = float(np.clip(0.7 * center_res.median_y + 0.3 * side_mid_y, 0.0, 1.0))

    invert_x = bool(right_res.median_x < left_res.median_x)
    observed_half_norm = float(np.median([abs(right_res.median_x - center_x), abs(left_res.median_x - center_x)]))
    expected_half_norm = float(side_offset_px_eff / max(screen_w, 1))
    if observed_half_norm < 1e-4:
        return AutoCalibrationResult(
            success=False, aborted=False,
            center_x=center_x, center_y=center_y, gain_x=1.0, gain_y=1.0, invert_x=invert_x,
            side_angle_deg=cfg.side_angle_deg, side_offset_cm=side_offset_cm,
            side_offset_px=side_offset_px_eff,
            center_samples=center_samples, right_samples=right_samples,
            left_samples=left_samples, top_samples=top_samples, bottom_samples=bottom_samples,
            message="Observed horizontal range was too small to fit gain_x.",
        )

    gain_x = float(np.clip(expected_half_norm / observed_half_norm, cfg.gain_x_min, cfg.gain_x_max))

    # gain_y: calibrar si los puntos arriba/abajo tienen suficientes muestras
    gain_y = 1.0
    if (top_res is not None and bot_res is not None
            and top_res.success and bot_res.success):
        observed_half_y = abs(top_res.median_y - bot_res.median_y) / 2.0
        expected_half_y = float(abs(top_y - cy) / max(screen_h, 1))
        if observed_half_y > 1e-4:
            gain_y = float(np.clip(expected_half_y / observed_half_y, cfg.gain_y_min, cfg.gain_y_max))

    # pose_gain_x: cuanto corregir gaze_x por giro de cabeza.
    # Formula: cuando cabeza gira manteniendo ojos al centro,
    #   gaze_x_raw + pose_gain_x * (yaw_proxy - yaw_central) debe = center_x
    pose_gain_x = 0.03  # default si no hay datos de head yaw
    pose_estimates: list[float] = []
    center_yaw = float(center_res.median_yaw_proxy)
    center_x_for_pose = float(center_res.median_x)
    for yaw_res in (yaw_r_res, yaw_l_res):
        if yaw_res is not None and yaw_res.success:
            dyaw = float(yaw_res.median_yaw_proxy - center_yaw)
            if abs(dyaw) <= 0.02:
                continue
            pg = (center_x_for_pose - float(yaw_res.median_x)) / dyaw
            pose_estimates.append(pg)
    if pose_estimates:
        pose_gain_x = float(np.clip(np.median(pose_estimates), cfg.pose_gain_x_min, cfg.pose_gain_x_max))

    msg = (
        "Autocalibration OK (7 puntos). "
        f"center=({center_x:.4f},{center_y:.4f}), "
        f"gain_x={gain_x:.3f}, gain_y={gain_y:.3f}, invert_x={invert_x}, "
        f"pose_gain_x={pose_gain_x:.3f}."
    )
    return AutoCalibrationResult(
        success=True, aborted=False,
        center_x=center_x, center_y=center_y,
        gain_x=gain_x, gain_y=gain_y, invert_x=invert_x, pose_gain_x=pose_gain_x,
        side_angle_deg=cfg.side_angle_deg, side_offset_cm=side_offset_cm,
        side_offset_px=side_offset_px_eff,
        center_samples=center_samples, right_samples=right_samples,
        left_samples=left_samples, top_samples=top_samples, bottom_samples=bottom_samples,
        message=msg,
    )
